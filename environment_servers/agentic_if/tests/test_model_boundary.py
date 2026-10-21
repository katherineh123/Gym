# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared boundary changes presentation, not implementations or tool arguments."""

from copy import deepcopy

import pytest

from environment_servers.agentic_if.model_boundary import ModelBoundary
from environment_servers.agentic_if.tests.support import definition, params
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, seal_runtime_row


def test_ingestion_rejects_a_second_ignored_system_prompt_channel():
    with pytest.raises(ValueError, match="instructions"):
        seal_runtime_row(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "Task"}],
                    "instructions": "A second prompt",
                }
            },
            RuntimeSpec(harness="opencode", task_backend="swe"),
        )


def chat():
    return {
        "messages": [{"role": "system", "content": "Native policy"}, {"role": "user", "content": "Task"}],
        "tools": [definition("read"), definition("bash")],
    }


def test_literal_task_survives_native_text_blocks_and_framing():
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    payload = {
        "messages": [
            {"role": "system", "content": "Native policy"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Native task framing:\n"},
                    {"type": "text", "text": "Task", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "\nTerminal state: /app $"},
                ],
            },
        ],
        "tools": [],
    }
    original = deepcopy(payload)
    prepared = ModelBoundary(spec, params()).prepare(payload)
    assert prepared.payload["messages"] == original["messages"]
    assert payload == original


def test_wrapped_multiple_user_messages_keep_native_framing_and_authored_boundaries():
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    authored = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "user", "content": "Task"},
            {"role": "user", "content": "Extra instruction"},
        ]
    )
    payload = {
        "messages": [{"role": "user", "content": "Task:\nTask\n\nExtra instruction\nTerminal: ready"}],
        "tools": [],
    }
    prepared = ModelBoundary(spec, authored).prepare(payload)
    assert prepared.payload["messages"] == [
        {"role": "user", "content": "Task:\nTask"},
        {"role": "user", "content": "Extra instruction\nTerminal: ready"},
    ]


def test_selected_tools_keep_native_interfaces_and_only_expose_selected_names():
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={
            "tool_names": {"read": "ReadSource", "bash": "RunCommand"},
            "tool_descriptions": {"read": "Read {{literal}}"},
            "tool_parameter_descriptions": {"read": {"/properties/path": "Path exactly"}},
        },
    )
    original = chat()
    before = deepcopy(original)
    prepared = ModelBoundary(spec, params()).prepare(original)
    tools = prepared.payload["tools"]
    assert [tool["function"]["name"] for tool in tools] == ["ReadSource", "RunCommand"]
    assert tools[0]["function"]["description"] == "Read {{literal}}"
    assert tools[0]["function"]["parameters"]["properties"]["path"]["description"] == "Path exactly"
    assert tools[0]["function"]["parameters"]["required"] == ["path"]
    assert original == before
    reply = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {"name": "ReadSource", "arguments": '{"path":"/app/x"}'},
                        }
                    ]
                }
            }
        ]
    }
    returned = prepared.native_reply(reply)
    call = returned["choices"][0]["message"]["tool_calls"][0]
    assert call["function"] == {"name": "read", "arguments": '{"path":"/app/x"}'}
    assert reply["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ReadSource"
    assert prepared.execution_tools == {"ReadSource": "read", "RunCommand": "bash"}


def test_empty_selection_exposes_nothing_and_omission_preserves_native_defaults():
    request = chat()
    default = ModelBoundary(RuntimeSpec(harness="opencode", task_backend="swe"), params())
    assert default.prepare(request).payload == request
    none = ModelBoundary(RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": {}}), params())
    with pytest.raises(ValueError, match="unexpected.*bash|unexpected.*read"):
        none.prepare(request)
    assert none.prepare({**request, "tools": []}).payload["tools"] == []


def test_child_permissions_are_intersected_not_expanded():
    boundary = ModelBoundary(
        RuntimeSpec(
            harness="opencode", task_backend="swe", tools={"tool_names": {"read": "Read", "task": "Delegate"}}
        ),
        params(),
    )
    request = chat()
    request["tools"] = request["tools"][:1]
    with pytest.raises(ValueError, match="unavailable"):
        boundary.prepare(request)
    child = boundary.prepare(request, child_session=True)
    assert [tool["function"]["name"] for tool in child.payload["tools"]] == ["Read"]


@pytest.mark.parametrize("selection", [None, {}, {"read": "Read"}])
def test_opencode_accepts_omitted_tools_when_native_registry_is_empty(selection):
    boundary = ModelBoundary(
        RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": selection}), params()
    )
    prepared = boundary.prepare({"messages": [{"role": "user", "content": "Task"}]}, child_session=True)
    assert prepared.payload["tools"] == []


@pytest.mark.parametrize("system", [None, "Only {{literal}}", ""])
@pytest.mark.parametrize("child_session", [False, True])
def test_one_system_contract_preserves_defaults_or_replaces_all_system_input(system, child_session):
    history = [
        {"role": "user", "content": "Native framing: Task"},
        {"role": "assistant", "content": "Earlier turn"},
        {"role": "user", "content": "Terminal observation"},
    ]
    request = {
        "messages": [
            {"role": "system", "content": "Native policy"},
            {"role": "developer", "content": "Native environment and project instructions"},
            *history,
        ],
        "instructions": "Separate native system instructions",
        "tools": [],
    }
    original = deepcopy(request)
    boundary = ModelBoundary(RuntimeSpec(harness="opencode", task_backend="swe"), params(system))
    prepared = boundary.prepare(request, child_session=child_session).payload
    if system is None:
        assert prepared == original
    else:
        assert prepared == {"messages": [{"role": "system", "content": system}, *history], "tools": []}
    assert request == original


def test_system_and_multiple_user_messages_are_literal():
    boundary = ModelBoundary(
        RuntimeSpec(harness="opencode", task_backend="swe"),
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": "Only {{literal}}"},
                {"role": "user", "content": "One"},
                {"role": "user", "content": "Two"},
            ]
        ),
    )
    request = {
        "messages": [
            {"role": "system", "content": "Native"},
            {"role": "system", "content": "Environment"},
            {"role": "user", "content": "One\n\nTwo"},
        ],
        "tools": [definition("read")],
    }
    prepared = boundary.prepare(request)
    assert prepared.payload["messages"] == [
        {"role": "system", "content": "Only {{literal}}"},
        {"role": "user", "content": "One"},
        {"role": "user", "content": "Two"},
    ]
