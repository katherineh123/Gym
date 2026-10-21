# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One native hook pipeline must compose interfaces, plants and literal policy."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from environment_servers.agentic_if.drivers import opencode_profile as profile
from environment_servers.agentic_if.drivers.opencode_hooks import compile_hooks
from environment_servers.agentic_if.drivers.opencode_profile import profile_sha256
from environment_servers.agentic_if.model_boundary import ModelBoundary
from environment_servers.agentic_if.tests.support import Provider, context
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence, SessionInfo, SessionStep
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


def hook_config(spec: RuntimeSpec) -> dict:
    """The hook configuration that stage_hooks uploads for an attempt without a continuation."""
    config = compile_hooks(spec)
    return config | {"profile_sha256": profile_sha256(config)}


def test_gateway_owns_literal_presentation_and_hooks_keep_installed_delegation():
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={
            "tool_names": {"read": "ReadRange", "task": "Delegate"},
            "tool_interfaces": {"read": "cw/zed@3/read", "task": "cw/zed@3/task"},
            "tool_descriptions": {"read": "Read exactly these lines."},
            "tool_parameter_descriptions": {"read": {"start_line": "First line, inclusive."}},
        },
    )
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "system", "content": "Keep {{cwd}} literal."},
            {"role": "user", "content": "Fix the bug."},
        ]
    )
    result = compile_hooks(spec)
    assert result["tools"]["task"]["fixed"] == {"subagent_type": "general"}
    assert result["tools"]["read"]["args_map"] == {"filePath": "path", "offset": "start_line", "limit": "line_count"}
    assert "description" not in result["tools"]["read"]
    # Parameter descriptions are presentation too: hooks keep the installed execution schema unchanged.
    installed = installed_interfaces({"read": "cw/zed@3/read"})["read"]
    assert result["tools"]["read"]["parameters"] == installed["parameters"]
    # Aliases are model-boundary labels; hooks keep native implementations.
    assert set(result["tools"]) == {"read", "task"}
    request = {
        "messages": [{"role": "system", "content": "Native context"}, {"role": "user", "content": "Fix the bug."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Native",
                    "parameters": tool["parameters"],
                },
            }
            for name, tool in result["tools"].items()
        ],
    }
    visible = ModelBoundary(spec, params).prepare(request).payload
    assert visible["messages"][0]["content"] == "Keep {{cwd}} literal."
    read = next(tool["function"] for tool in visible["tools"] if tool["function"]["name"] == "ReadRange")
    assert read["description"] == "Read exactly these lines."
    assert read["parameters"]["properties"]["start_line"]["description"] == "First line, inclusive."


def test_event_receipt_is_correlated_to_the_exact_session_and_call():
    from environment_servers.agentic_if.drivers.opencode_hooks import attach_events

    evidence = SessionEvidence(
        status="complete",
        sessions=[SessionInfo(id="root"), SessionInfo(id="child", parent_id="root")],
        steps=[
            SessionStep(
                id="a", session_id="root", role="tool", tool="read", call_id="same", invocation_id="root", output="ok"
            ),
            SessionStep(
                id="b",
                session_id="child",
                role="tool",
                tool="read",
                call_id="same",
                invocation_id="child",
                output="NOTICE",
                error=True,
            ),
        ],
    )
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        events=[{"id": "notice", "match": {"tool": "read"}, "action": "fail", "text": "NOTICE"}],
    )
    records = [
        {"kind": "loaded", "plants": 1},
        {
            "kind": "fired",
            "event_id": "notice",
            "index": 0,
            "tool": "read",
            "action": "fail",
            "session_id": "child",
            "call_id": "same",
            "call_number": 2,
        },
    ]
    result = attach_events(evidence, spec.events, records)
    assert result.status == "complete"
    assert result.events[0].id == "notice"
    assert result.events[0].step_index == 1
    assert evidence.events == []
    assert attach_events(evidence, spec.events, []).status == "incomplete"
    bad = [records[0], records[1] | {"call_id": "missing"}]
    assert attach_events(evidence, spec.events, bad).status == "incomplete"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute the native plugin")
def test_native_error_presentation_uses_actor_labels_without_rewriting_raw_output(tmp_path):
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={
            "tool_names": {"read": "ReadSource", "bash": "RunCommand", "edit": "EditSource", "write": "WriteSource"},
            "tool_interfaces": {"read": "cw/zed@3/read", "edit": "cw/pi@2/edit"},
        },
        events=[{"id": "literal", "match": {"tool": "read"}, "action": "fail", "text": "PLANTED read timeout"}],
    )
    config = tmp_path / "profile.json"
    # Isolate argument/error transforms; alias provenance is tested with its HTTP endpoint separately.
    config.write_text(json.dumps(hook_config(spec) | {"check_aliases": False}))
    plugin = tmp_path / "profile.mjs"
    plugin.write_text(Path(profile.__file__).with_name("if-profile.js").read_text())
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("opencode_errors.test.mjs"))],
        env=os.environ
        | {
            "IF_PROFILE_PLUGIN": str(plugin),
            "IF_TEST_ROOT": str(tmp_path),
            "NEMO_GYM_OPENCODE_PROFILE": str(config),
            "NEMO_GYM_OPENCODE_PROFILE_APPLIED": str(tmp_path / "applied.jsonl"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "error presentation passed" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute the native plugin")
def test_combined_hook_execution_and_binary_workspace_receipts(tmp_path):
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={
            "tool_names": {"read": "ReadRange", "task": "Delegate", "bash": "Shell"},
            "tool_interfaces": {"read": "cw/zed@3/read", "task": "cw/zed@3/task"},
            "tool_parameter_descriptions": {"read": {"start_line": "First line, inclusive."}},
        },
    )
    config = tmp_path / "profile.json"
    config.write_text(json.dumps(hook_config(spec) | {"check_aliases": False}))
    plugin = tmp_path / "profile.mjs"
    plugin.write_text(Path(profile.__file__).with_name("if-profile.js").read_text())
    script = Path(__file__).with_name("opencode_hooks.test.mjs")
    env = os.environ | {
        "IF_PROFILE_PLUGIN": str(plugin),
        "NEMO_GYM_OPENCODE_PROFILE": str(config),
        "NEMO_GYM_OPENCODE_PROFILE_APPLIED": str(tmp_path / "applied.jsonl"),
        "NEMO_GYM_OPENCODE_PLANTS": str(tmp_path / "plants.json"),
        "NEMO_GYM_OPENCODE_PLANTS_FIRED": str(tmp_path / "fired.jsonl"),
        "IF_TEST_ROOT": str(tmp_path),
    }
    result = subprocess.run(["node", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "combined hooks passed" in result.stdout


@pytest.mark.asyncio
async def test_staged_continuation_schedule_is_every_declared_event(tmp_path):
    from dataclasses import replace

    from environment_servers.agentic_if.bootstrap import BinarySandbox
    from environment_servers.agentic_if.drivers.opencode_hooks import stage_hooks
    from resources_servers.agentic_if.task_variants.runtime import ToolEvent
    from resources_servers.agentic_if.tests.task_variants.test_runtime_contract import planted_continuation

    spec = RuntimeSpec(harness="opencode", task_backend="swe", **planted_continuation())
    provider = Provider(child=False)
    ctx = replace(context(tmp_path, provider), spec=spec)
    hooks = await stage_hooks(ctx, BinarySandbox(ctx.sandbox, "/tmp/private-run"))
    # A continuation's events all fire in the live run, staged in the form the plugin reads.
    live = json.loads(provider.files[hooks.environment()["NEMO_GYM_OPENCODE_PLANTS"]])
    assert live == [
        {"id": "timeout", "match": {"tool": "read"}, "nth": 1, "action": "fail", "text": "Read timed out."}
    ]
    assert [ToolEvent.model_validate(event) for event in live] == spec.events
    config = json.loads(provider.files[hooks.environment()["NEMO_GYM_OPENCODE_PROFILE"]])
    assert config["literal_errors"] == ["Read timed out."]


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to execute the native plugin")
async def test_staged_request_kind_plugin_wraps_upstream_assistant_header(tmp_path):
    from environment_servers.agentic_if.bootstrap import BinarySandbox
    from environment_servers.agentic_if.drivers.opencode_hooks import stage_hooks

    provider = Provider(child=False)
    ctx = context(tmp_path, provider)
    hooks = await stage_hooks(ctx, BinarySandbox(ctx.sandbox, "/tmp/private-run"))
    # Upstream's plugin is staged for the wrapper's relative import, but only the wrapper is registered.
    assert hooks.plugins == ["file:///tmp/private-run/if-profile.mjs", "file:///tmp/private-run/if-request-kind.mjs"]
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in ("assistant-message-header.mjs", "if-request-kind.mjs"):
        (staged / name).write_bytes(provider.files["/tmp/private-run/" + name])
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("opencode_request_kind.test.mjs"))],
        env=os.environ | {"IF_REQUEST_KIND_PLUGIN": str(staged / "if-request-kind.mjs")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
