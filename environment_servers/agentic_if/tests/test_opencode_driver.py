# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OpenCode execution must collect every child and keep the task identity."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from environment_servers.agentic_if.tests.support import (
    REASONING,
    Provider,
    call,
    context,
    driver,
    export,
    message,
    observation,
    text,
    tool_observation,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


@pytest.mark.asyncio
async def test_native_opencode_collects_children_and_keeps_private_run_state(tmp_path):
    provider = Provider()
    ctx = context(tmp_path, provider)
    result = await driver(tmp_path).execute(ctx)
    assert result.status == "completed"
    assert result.evidence.status == "complete"
    assert [(session.id, session.parent_id) for session in result.evidence.sessions] == [
        ("root", None),
        ("child", "root"),
    ]
    assert REASONING not in result.evidence.model_dump_json()
    launches = [call for call in provider.calls if "opencode run " in call["command"]]
    assert len(launches) == 1
    assert launches[0]["user"] == "agent"
    assert launches[0]["cwd"] == "/app/repo"
    config = json.loads(launches[0]["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["nemo_gym"]["options"]["baseURL"] == "http://gym/attempt/oc/v1"
    assert config["agent"]["build"]["steps"] == 5
    assert launches[0]["env"]["XDG_DATA_HOME"].startswith("/tmp/")
    assert launches[0]["env"]["XDG_STATE_HOME"].startswith("/tmp/")
    assert any(content == b"Fix 'quoted' {literal}" for content in provider.files.values())
    assert any(path.endswith("/assistant-message-header.mjs") for path in provider.files)
    assert any(plugin.endswith("/if-request-kind.mjs") for plugin in config["plugin"])
    assert not any("python" in call["command"] for call in provider.calls)
    assert next(item for item in result.response.output if item.type == "message").content[0].text == "Done"


@pytest.mark.asyncio
async def test_driver_marks_child_capture_mismatch_incomplete(tmp_path):
    provider = Provider()
    provider.observations[1]["response"]["choices"][0]["message"]["content"] = "Child actually said this"
    result = await driver(tmp_path).execute(context(tmp_path, provider))
    assert result.status == "error"
    assert result.evidence.status == "incomplete"
    assert any("training_capture:child:" in issue for issue in result.evidence.issues)
    assert not any("training_capture:root:" in issue for issue in result.evidence.issues)


@pytest.mark.asyncio
async def test_failed_child_export_is_not_reported_as_complete(tmp_path):
    provider = Provider(missing=True)
    result = await driver(tmp_path).execute(context(tmp_path, provider))
    assert result.status == "error"
    assert result.evidence.status == "incomplete"
    assert any("child" in issue for issue in result.evidence.issues)


@pytest.mark.asyncio
async def test_open_code_never_starts_if_the_binary_pin_is_wrong(tmp_path):
    provider = Provider()
    executor = driver(tmp_path)
    executor.binary_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        await executor.execute(context(tmp_path, provider))
    assert not provider.calls


@pytest.mark.asyncio
async def test_driver_installs_one_hook_pipeline_for_interface_and_literal_system(tmp_path):
    provider = Provider(child=False)
    ctx = context(tmp_path, provider)
    ctx = replace(
        ctx,
        spec=RuntimeSpec(
            harness="opencode",
            task_backend="swe",
            tools={"tool_names": {"task": "Delegate"}, "tool_interfaces": {"task": "cw/zed@3/task"}},
        ),
        params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "system", "content": "Literal {{cwd}}"}, {"role": "user", "content": "Fix"}]
        ),
    )
    result = await driver(tmp_path).execute(ctx)
    assert result.status == "completed"
    launch = next(call for call in provider.calls if "opencode run " in call["command"])
    config = json.loads(launch["env"]["OPENCODE_CONFIG_CONTENT"])
    assert sum(plugin.endswith("/if-profile.mjs") for plugin in config["plugin"]) == 1
    native_hook = provider.files[launch["env"]["NEMO_GYM_OPENCODE_PROFILE"]]
    native_hook = json.loads(native_hook)
    assert native_hook["tools"]["task"]["fixed"]["subagent_type"] == "general"
    assert "hook_applied" in result.artifacts


@pytest.mark.asyncio
async def test_driver_returns_actor_tool_names_and_raw_arguments_not_native_dispatch_labels(tmp_path):
    class CallingProvider(Provider):
        async def download(self, source, destination):
            if "export-root" in source:
                document = export(
                    message(
                        "a", "assistant", text("Inspect"), call("c1", "bash", input={"command": "pwd"}, output="/app")
                    ),
                    message("z", "assistant", text("Done"), created=5),
                )
                Path(destination).write_text(json.dumps(document))
                return
            await super().download(source, destination)

    observed = observation()
    observed["execution_tools"] = {"RunCommand": "bash"}
    ctx = replace(
        context(tmp_path, CallingProvider(child=False)),
        spec=RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": {"bash": "RunCommand"}}),
        model_observations=[observed, tool_observation()],
    )
    result = await driver(tmp_path).execute(ctx)
    assert result.status == "completed"
    actual = next(item for item in result.response.output if item.type == "function_call")
    assert actual.name == "RunCommand"
    assert actual.arguments == '{ "command": "pwd" }'
    assert any(step.role == "system" and step.text == "Explain each command." for step in result.evidence.steps)
