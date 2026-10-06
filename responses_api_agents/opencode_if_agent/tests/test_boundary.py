# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json


def test_system_start_is_before_native_prompt_and_end_after_it():
    from responses_api_agents.opencode_if_agent.boundary import model_request

    result = model_request(
        {"messages": [{"role": "system", "content": "NATIVE"}]},
        tool_names={},
        system_text="AFTER",
        system_prefix="BEFORE",
    )
    assert result["messages"][0]["content"] == "BEFORE\n\nNATIVE\n\nAFTER"


def test_tool_description_instruction_does_not_change_arguments_or_dispatch_name():
    from responses_api_agents.opencode_if_agent.boundary import model_request

    payload = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a command.",
                    "parameters": {"type": "object"},
                },
            }
        ]
    }
    result = model_request(
        payload,
        tool_names={"bash": "shell"},
        system_text="",
        instructions=[
            {
                "instruction_text": "Explain the purpose of each shell call.",
                "placement": {"surface": "tool_description", "position": "end", "tool": "bash"},
            }
        ],
    )
    function = result["tools"][0]["function"]
    assert function["name"] == "shell"
    assert function["description"] == "Execute a command.\n\nExplain the purpose of each shell call."
    assert function["parameters"] == {"type": "object"}
    assert payload["tools"][0]["function"]["description"] == "Execute a command."


from copy import deepcopy

import pytest


def payload():
    return {
        "model": "dummy_model",
        "stream": True,
        "messages": [
            {"role": "system", "content": "Native system."},
            {"role": "user", "content": "Find bash in the repository."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"echo bash"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "bash"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "bash"}},
    }


def test_boundary_renames_schema_history_and_tool_choice_without_rewriting_data():
    from responses_api_agents.opencode_if_agent.boundary import model_request

    source = payload()
    before = deepcopy(source)
    actual = model_request(source, tool_names={"bash": "shell"}, system_text="Always explain commands.")
    assert source == before
    assert actual["tools"][0]["function"]["name"] == "shell"
    assert actual["tools"][0]["function"]["parameters"] == source["tools"][0]["function"]["parameters"]
    assert actual["messages"][0]["role"] == "system"
    assert "Always explain commands." in actual["messages"][0]["content"]
    assert actual["messages"][1] == source["messages"][1]
    assert actual["messages"][2]["tool_calls"][0]["function"] == {
        "name": "shell",
        "arguments": '{"command":"echo bash"}',
    }
    assert actual["messages"][3] == source["messages"][3]
    assert actual["tool_choice"]["function"]["name"] == "shell"


def test_absent_or_colliding_native_tool_is_a_configuration_error():
    from responses_api_agents.opencode_if_agent.boundary import model_request

    with pytest.raises(ValueError, match="unavailable"):
        model_request(payload(), tool_names={"read": "read_file"}, system_text="")
    source = payload()
    source["tools"].append({"type": "function", "function": {"name": "shell"}})
    with pytest.raises(ValueError, match="collision"):
        model_request(source, tool_names={"bash": "shell"}, system_text="")


def test_nonstream_tool_response_is_mapped_back_to_native_execution():
    from responses_api_agents.opencode_if_agent.boundary import native_response

    source = {
        "choices": [
            {
                "message": {
                    "content": "shell is useful",
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"command":"echo shell"}'},
                        }
                    ],
                }
            }
        ]
    }
    actual = native_response(source, {"bash": "shell"})
    assert actual["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert source["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "shell"
    assert actual["choices"][0]["message"]["content"] == "shell is useful"


def event(delta, finish=None):
    return {"id": "c", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def call(name=None, arguments=None, index=0):
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    return {"tool_calls": [{"index": index, "id": "call_a", "type": "function", "function": function}]}


def test_stream_fragmented_names_are_reassembled_before_reverse_mapping():
    from responses_api_agents.opencode_if_agent.boundary import ToolStreamRewriter

    writer = ToolStreamRewriter({"bash": "shell"})
    output = []
    for chunk in [
        event(call("sh")),
        event(call("ell")),
        event(call(arguments='{"command":')),
        event(call(arguments='"echo shell"}')),
        event({}, "tool_calls"),
    ]:
        output.extend(writer.feed(chunk))
    calls = [call for chunk in output for choice in chunk["choices"] for call in choice["delta"].get("tool_calls", [])]
    assert "".join(c["function"].get("name", "") for c in calls) == "bash"
    assert "".join(c["function"].get("arguments", "") for c in calls) == '{"command":"echo shell"}'
    assert output[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_sse_split_network_frames_preserve_usage_and_done():
    from responses_api_agents.opencode_if_agent.boundary import rewrite_sse

    chunks = [event(call("shell", "{}")), event({}, "tool_calls"), {"choices": [], "usage": {"total_tokens": 7}}]
    wire = "".join("data: " + json.dumps(chunk) + "\r\n\r\n" for chunk in chunks) + "data: [DONE]\r\n\r\n"

    async def stream():
        for i in range(0, len(wire), 7):
            yield wire[i : i + 7].encode()

    actual = b"".join([part async for part in rewrite_sse(stream(), {"bash": "shell"})]).decode()
    assert '"name": "bash"' in actual
    assert '"total_tokens": 7' in actual
    assert actual.endswith("data: [DONE]\n\n")


def test_parallel_stream_tools_do_not_share_names():
    from responses_api_agents.opencode_if_agent.boundary import ToolStreamRewriter

    writer = ToolStreamRewriter({"bash": "shell", "read": "read_file"})
    output = []
    for chunk in [
        event(call("shell", index=0)),
        event(call("read_file", index=1)),
        event(call(arguments="{}", index=1)),
        event(call(arguments="{}", index=0)),
        event({}, "tool_calls"),
    ]:
        output.extend(writer.feed(chunk))
    names = {
        c["index"]: c["function"]["name"]
        for chunk in output
        for choice in chunk["choices"]
        for c in choice["delta"].get("tool_calls", [])
        if "name" in c["function"]
    }
    assert names == {0: "bash", 1: "read"}
