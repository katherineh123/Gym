# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from environment_servers.agentic_if.tests.support import ModelServer, gateway


def request(*, stream=False):
    return {
        "messages": [{"role": "user", "content": "Fix it"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        "stream": stream,
    }


def reply(name, *, call_id="call-a"):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name,allowed", [("ReadSource", True), ("read", False), ("READ", False), ("made_up", False)])
async def test_only_exposed_alias_authorizes_the_native_executor(name, allowed):
    route = gateway(ModelServer(reply(name)), tools={"tool_names": {"read": "ReadSource"}})
    response = await route.forward(request(), headers={"x-session-id": "root"})
    actual = json.loads(response.body)["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    assert actual == ("read" if allowed else name)
    verdict = route.tool_call(session_id="root", call_id="call-a", native_tool="read")
    assert verdict["allowed"] is allowed
    assert verdict["model_name"] == name
    assert verdict["available_names"] == ["ReadSource"]


@pytest.mark.asyncio
async def test_parent_and_child_with_identical_call_ids_cannot_borrow_authorization():
    server = ModelServer(reply("ReadSource"))
    route = gateway(server, tools={"tool_names": {"read": "ReadSource"}})
    await route.forward(request(), headers={"x-session-id": "root"})
    server.result = reply("read")
    await route.forward(request(), headers={"x-session-id": "child"})
    assert route.tool_call(session_id="root", call_id="call-a", native_tool="read")["allowed"]
    assert not route.tool_call(session_id="child", call_id="call-a", native_tool="read")["allowed"]
    with pytest.raises(ValueError, match="unobserved"):
        route.tool_call(session_id="absent", call_id="call-a", native_tool="read")
    assert not route.tool_call(session_id="root", call_id="call-a", native_tool="bash")["allowed"]


@pytest.mark.asyncio
async def test_authorization_exists_before_the_stream_terminal_event():
    events = [
        {
            "id": "response-a",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-a",
                                "type": "function",
                                "function": {"name": "Read", "arguments": ""},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "id": "response-a",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"name": "Source", "arguments": "{}"}}]},
                }
            ],
        },
        {"id": "response-a", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    route = gateway(ModelServer(events, stream=True), tools={"tool_names": {"read": "ReadSource"}})
    response = await route.forward(request(stream=True), headers={"x-session-id": "root"})
    checked = False
    async for chunk in response.body_iterator:
        if b'"arguments": "{}"' in chunk:
            assert "response" not in route.observations[0]
            assert route.tool_call(session_id="root", call_id="call-a", native_tool="read")["allowed"]
            checked = True
    assert checked


@pytest.mark.asyncio
async def test_duplicate_call_identity_fails_closed_instead_of_replacing_prior_provenance():
    server = ModelServer(reply("ReadSource"))
    route = gateway(server, tools={"tool_names": {"read": "ReadSource"}})
    await route.forward(request(), headers={"x-session-id": "root"})
    server.result = reply("read")
    with pytest.raises(ValueError, match="duplicate"):
        await route.forward(request(), headers={"x-session-id": "root"})
    with pytest.raises(ValueError, match="ambiguous"):
        route.tool_call(session_id="root", call_id="call-a", native_tool="read")


@pytest.mark.asyncio
async def test_rejected_native_spelling_is_not_relabelled_as_a_valid_alias_in_history():
    server = ModelServer(reply("read"))
    route = gateway(server, tools={"tool_names": {"read": "ReadSource"}})
    await route.forward(request(), headers={"x-session-id": "root"})
    history = request()
    history["messages"] += [
        reply("read")["choices"][0]["message"],
        {"role": "tool", "tool_call_id": "call-a", "content": "read unavailable"},
    ]
    server.result = {"choices": [{"message": {"role": "assistant", "content": "Retry"}}]}
    await route.forward(history, headers={"x-session-id": "root"})
    assert server.requests[-1]["json"]["messages"][1]["tool_calls"][0]["function"]["name"] == "read"
