# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native errors belong to OpenCode, not the IF model transport."""

import json

import pytest

from environment_servers.agentic_if.drivers.opencode_evidence import opencode_evidence
from environment_servers.agentic_if.drivers.opencode_observations import bind_observations
from environment_servers.agentic_if.model_boundary import ModelBoundary
from environment_servers.agentic_if.tests.support import (
    ModelServer,
    call,
    definition,
    export,
    gateway,
    message,
    params,
)
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


def reply(name):
    function = {"name": name, "arguments": '{ "path": "/tmp/x" }'}
    return {"choices": [{"message": {"tool_calls": [{"id": "call-1", "type": "function", "function": function}]}}]}


def function(response):
    return response["choices"][0]["message"]["tool_calls"][0]["function"]


@pytest.mark.parametrize("name", ["READ", "made_up_tool"])
def test_native_default_calls_are_not_repaired_or_rejected_by_transport(name):
    payload = {"messages": [{"role": "user", "content": "Task"}], "tools": []}
    boundary = ModelBoundary(RuntimeSpec(harness="opencode", task_backend="swe"), params())
    prepared = boundary.prepare(payload)
    original = reply(name)
    assert prepared.native_reply(original) == original


def test_aliases_follow_the_native_lowercase_repair():
    payload = {"messages": [{"role": "user", "content": "Task"}], "tools": [definition("read")]}
    boundary = ModelBoundary(
        RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": {"read": "inspect"}}), params()
    )
    prepared = boundary.prepare(payload)
    output = function(prepared.native_reply(reply("INSPECT")))
    # Leave the capitalization repair itself to OpenCode, after translating the alias.
    assert output["name"] == "READ"
    assert prepared.execution_tools["INSPECT"] == "read"
    assert output["arguments"] == '{ "path": "/tmp/x" }'


def test_opencode_unknown_call_keeps_native_invalid_receipt_and_original_model_attempt():
    source = export(
        message(
            "a",
            "assistant",
            call("call-1", "invalid", input={"tool": "made_up_tool", "error": "unavailable"}, output="Native error"),
        )
    )
    observed = {
        "session_id": "root",
        "assistant_message_id": "a",
        "execution_tools": {},
        "request": {"messages": []},
        "response": reply("made_up_tool"),
    }
    evidence = bind_observations(opencode_evidence(source), [observed])
    assert evidence.status == "complete"
    attempted = evidence.steps[0].calls[0]
    assert attempted.model_tool == "made_up_tool"
    assert attempted.model_args == {"path": "/tmp/x"}
    assert attempted.rejected is True
    assert evidence.steps[1].output == "Native error"


@pytest.mark.asyncio
async def test_opencode_native_case_repair_remains_visible_to_evidence():
    response = reply("READ")
    response["id"] = "r1"
    response["choices"][0]["message"]["role"] = "assistant"
    response["choices"][0]["finish_reason"] = "tool_calls"
    route = gateway(ModelServer(response))
    result = await route.forward(
        {"messages": [{"role": "user", "content": "Fix it"}], "tools": [definition("read")]},
        headers={"x-session-id": "root", "x-opencode-assistant-message-id": "a"},
    )
    assert function(json.loads(result.body))["name"] == "READ"
    assert route.observations[0]["execution_tools"]["READ"] == "read"
