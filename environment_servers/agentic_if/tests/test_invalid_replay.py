# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rejected-call continuations retain what the model actually sent, not native repair text."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from environment_servers.agentic_if.drivers.opencode_hooks import StagedHooks
from environment_servers.agentic_if.drivers.opencode_observations import prefix_model_calls, preserve_rejected_calls
from environment_servers.agentic_if.model_gateway import ModelGateway
from environment_servers.agentic_if.tests.support import ModelServer, gateway
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


def native_export(native_tool="invalid"):
    result = {
        "info": {"id": "source"},
        "messages": [
            {"info": {"id": "u", "role": "user"}, "parts": [{"type": "text", "text": "Fix it"}]},
            {
                "info": {"id": "a", "role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": "invalid",
                        "callID": "rejected",
                        "state": {
                            "status": "completed",
                            "input": {"tool": "Missing", "error": "Available tools: read"},
                            "output": "Available tools: read",
                            "metadata": {"nemo_gym_profile": {"workspace_diff": ""}},
                        },
                    }
                ],
            },
        ],
    }
    if native_tool == "read":
        part = result["messages"][1]["parts"][0]
        part["tool"] = "read"
        error = "read is unavailable. Available tools: ReadSource."
        part["state"] = {
            "status": "error",
            "input": {"filePath": "source"},
            "error": error,
            "metadata": {
                "nemo_gym_profile": {
                    "dispatch_rejected": True,
                    "model_name": "read",
                    "error": error,
                }
            },
        }
    return result


def history(native_tool="invalid"):
    part = native_export(native_tool)["messages"][1]["parts"][0]
    return {
        "messages": [
            {"role": "user", "content": "Continue"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "rejected",
                        "type": "function",
                        "function": {
                            "name": native_tool,
                            "arguments": json.dumps(part["state"]["input"]),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "rejected",
                "content": part["state"].get("output", part["state"].get("error")),
            },
        ],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_tool,arguments",
    [
        ("invalid", '{ "path": "source" }'),
        ("invalid", "{"),
        ("read", '{ "filePath": "source" }'),
    ],
)
async def test_captured_rejected_call_survives_export_and_continuation(tmp_path, monkeypatch, native_tool, arguments):
    original = {"name": "read" if native_tool == "read" else "Missing", "arguments": arguments}
    server = ModelServer(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "rejected", "type": "function", "function": original},
                        ],
                    }
                }
            ]
        }
    )
    source = gateway(server, tools={"tool_names": {"read": "ReadSource"}})
    request = history()
    request["messages"] = [{"role": "user", "content": "Fix it"}]
    await source.forward(request, headers={"x-session-id": "source"})
    assert source.tool_call(session_id="source", call_id="rejected", native_tool=native_tool)["allowed"] is False

    async def no_hook_records(self, context):
        return []

    monkeypatch.setattr(StagedHooks, "records", no_hook_records)
    raw = {"source": native_export(native_tool)}
    saved = deepcopy(raw)
    context = SimpleNamespace(
        spec=source.boundary.spec, model_observations=source.observations, artifacts_dir=tmp_path
    )
    enriched, _ = await StagedHooks("/private/run", "hash").enrich_exports(context, raw, {"source": "raw.json"})
    receipt = enriched["source"]["messages"][1]["parts"][0]["state"]["metadata"]["nemo_gym_model_call"]
    assert receipt == {"function": original, "available_names": ["ReadSource"]}
    assert raw == saved

    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={"tool_names": {"read": "ReadSource"}},
        continuation={
            "mode": "restore",
            "prefix_export": enriched["source"],
            "workspace_diff": "",
            "source_task_sha256": "a" * 64,
            "source_tools_sha256": "b" * 64,
        },
    )
    resumed_server = ModelServer({"choices": [{"message": {"role": "assistant", "content": "Done"}}]})
    resumed = ModelGateway(
        spec=spec,
        params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Continue"}]),
        client=resumed_server,
        model_server="actor",
        run_id="attempt-b",
        timeout_s=30,
    )
    await resumed.forward(history(native_tool), headers={"x-session-id": "new-root"})
    messages = resumed_server.requests[-1]["json"]["messages"]
    assert messages[-2]["tool_calls"][0]["function"] == original
    assert messages[-1]["content"] == (
        "read is unavailable. Available tools: ReadSource."
        if native_tool == "read"
        else "Missing is unavailable or its arguments are invalid. Available tools: ReadSource."
    )
    # A child cannot borrow a root's historical call even with an identical ID, and nothing is fabricated or sent.
    child = history()
    with pytest.raises(ValueError, match="unobserved rejected"):
        await resumed.forward(child, headers={"x-session-id": "child"})
    assert child == history()
    assert len(resumed_server.requests) == 1

    # Another exported continuation must retain the same source evidence.
    restored = native_export(native_tool)
    restored["info"]["id"] = "new-root"
    context = SimpleNamespace(spec=spec, model_observations=resumed.observations, artifacts_dir=tmp_path)
    again, _ = await StagedHooks("/private/run", "hash").enrich_exports(
        context, {"new-root": restored}, {"new-root": "raw-2.json"}
    )
    assert again["new-root"]["messages"][1]["parts"][0]["state"]["metadata"]["nemo_gym_model_call"] == receipt


@pytest.mark.parametrize("case", ["duplicate_call", "duplicate_names", "wrong_arguments"])
@pytest.mark.parametrize("native_tool", ["invalid", "read"])
def test_invalid_prefix_receipts_fail_closed(case, native_tool):
    export = native_export(native_tool)
    part = export["messages"][1]["parts"][0]
    receipt = {"function": {"name": "Missing", "arguments": "{"}, "available_names": ["ReadSource"]}
    part["state"]["metadata"]["nemo_gym_model_call"] = receipt
    if case == "duplicate_call":
        export["messages"][1]["parts"].append(deepcopy(part))
    elif case == "duplicate_names":
        receipt["available_names"].append("ReadSource")
    else:
        receipt["function"]["arguments"] = {}
    with pytest.raises(ValueError, match="invalid or duplicate"):
        prefix_model_calls(SimpleNamespace(prefix_export=export))


@pytest.mark.parametrize("native_tool", ["invalid", "read"])
def test_export_does_not_infer_lost_model_arguments(native_tool):
    with pytest.raises(ValueError, match="unobserved rejected"):
        preserve_rejected_calls({"source": native_export(native_tool)}, [], None)


def test_ordinary_tool_error_does_not_require_rejection_receipts():
    export = native_export("read")
    state = export["messages"][1]["parts"][0]["state"]
    state["metadata"] = {"nemo_gym_profile": {"workspace_diff": ""}}
    state["error"] = "File not found"
    saved = deepcopy(export)
    preserve_rejected_calls({"source": export}, [], None)
    assert export == saved


@pytest.mark.parametrize("native_tool", ["invalid", "read"])
def test_rejected_prefix_without_captured_call_fails_before_native_start(native_tool):
    server = ModelServer({"choices": [{"message": {"role": "assistant", "content": "Done"}}]})
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={"tool_names": {"read": "ReadSource"}},
        continuation={
            "mode": "restore",
            "prefix_export": native_export(native_tool),
            "workspace_diff": "",
            "source_task_sha256": "a" * 64,
            "source_tools_sha256": "b" * 64,
        },
    )
    with pytest.raises(ValueError, match="requires captured call evidence"):
        ModelGateway(
            spec=spec,
            params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Continue"}]),
            client=server,
            model_server="actor",
            run_id="resumed",
            timeout_s=30,
        )
    assert server.requests == []


@pytest.mark.asyncio
async def test_native_default_replay_preserves_stock_invalid_history_without_custom_receipts():
    server = ModelServer({"choices": [{"message": {"role": "assistant", "content": "Done"}}]})
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        continuation={
            "mode": "restore",
            "prefix_export": native_export(),
            "workspace_diff": "",
            "source_task_sha256": "a" * 64,
            "source_tools_sha256": "b" * 64,
        },
    )
    route = ModelGateway(
        spec=spec,
        params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Continue"}]),
        client=server,
        model_server="actor",
        run_id="native-default",
        timeout_s=30,
    )
    request = history()
    await route.forward(request, headers={"x-session-id": "root"})
    actual = server.requests[-1]["json"]["messages"]
    assert actual[-2]["tool_calls"] == request["messages"][-2]["tool_calls"]
    assert actual[-1]["content"] == request["messages"][-1]["content"]
