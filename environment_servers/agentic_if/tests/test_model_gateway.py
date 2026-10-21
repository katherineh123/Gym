# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest
from aiohttp import ClientResponseError, ClientSession, web
from aiohttp.test_utils import TestServer

from environment_servers.agentic_if.tests.support import ModelServer, gateway


async def streamed_body(response):
    return b"".join(
        [part.encode() if isinstance(part, str) else part async for part in response.body_iterator]
    ).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_native_compaction_bypasses_actor_presentation_but_is_recorded(stream):
    message = {"role": "assistant", "content": "Native summary"}
    if stream:
        server = ModelServer(
            [
                {"id": "summary", "choices": [{"index": 0, "delta": message}]},
                {"id": "summary", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ],
            stream=True,
        )
    else:
        server = ModelServer({"choices": [{"message": message}]})
    route = gateway(server, tools={"tool_names": {"read": "Read"}})
    route.boundary.system = "The task's literal system prompt must not replace native summarization."
    payload = {
        "messages": [
            {"role": "system", "content": "Summarize natively"},
            {"role": "user", "content": "History without the original task"},
        ],
        "tools": [],
    } | ({"stream": True} if stream else {})
    before = deepcopy(payload)
    response = await route.forward(
        payload,
        headers={
            "x-session-id": "root",
            "x-opencode-assistant-message-id": "summary",
            "x-opencode-request-kind": "compaction",
        },
    )
    if stream:
        await streamed_body(response)
    assert server.requests[0]["json"] == before
    assert payload == before
    assert route.observations[0]["source"] == "compaction"
    assert route.observations[0]["response"]["choices"][0]["message"]["content"] == "Native summary"
    assert route.root_session is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,tools",
    [
        ({"x-opencode-request-kind": "compaction"}, []),
        (
            {"x-opencode-request-kind": "compaction", "x-opencode-assistant-message-id": "summary"},
            [{"type": "function", "function": {"name": "read"}}],
        ),
        ({"x-opencode-request-kind": "unknown", "x-opencode-assistant-message-id": "a"}, []),
    ],
)
async def test_invalid_native_request_kind_is_not_a_bypass(headers, tools):
    server = ModelServer({})
    route = gateway(server)
    with pytest.raises(ValueError, match="native"):
        await route.forward({"messages": [], "tools": tools}, headers=headers)
    assert not server.requests
    assert not route.observations


@pytest.mark.asyncio
async def test_opencode_can_continue_after_native_context_compaction():
    server = ModelServer({"choices": [{"message": {"role": "assistant", "content": "Continue"}}]})
    route = gateway(server)
    initial = {"messages": [{"role": "user", "content": "Fix it"}]}
    compacted = {"messages": [{"role": "user", "content": "Native summary and continuation"}]}
    with pytest.raises(ValueError, match="literal task"):
        await route.forward(compacted, headers={"x-session-id": "root", "x-opencode-assistant-message-id": "bad"})
    await route.forward(initial, headers={"x-session-id": "root", "x-opencode-assistant-message-id": "initial"})
    await route.forward(compacted, headers={"x-session-id": "root", "x-opencode-assistant-message-id": "after"})
    assert server.requests[-1]["json"]["messages"] == compacted["messages"]
    assert len(server.requests) == 2


@pytest.mark.asyncio
async def test_http_rejection_is_retained_without_poisoning_a_native_retry():
    """The model endpoint rejects one request; the native harness retries it."""
    calls = 0

    async def endpoint(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.json_response({"error": "rate limited"}, status=429)
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "Done"}}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", endpoint)
    async with TestServer(app) as server, ClientSession() as session:

        class Client:
            async def post(self, *, server_name, url_path, _max_num_tries, **kwargs):
                return await session.post(server.make_url(url_path), **kwargs)

        route = gateway(Client())
        payload = {"messages": [{"role": "user", "content": "Fix it"}]}
        headers = {"x-session-id": "root", "x-opencode-assistant-message-id": "a"}
        with pytest.raises(ClientResponseError, match="429"):
            await route.forward(payload, headers=headers)
        # No gateway retry; the second call is made by the harness itself.
        assert calls == 1
        reply = await route.forward(payload, headers=headers)
        assert json.loads(reply.body)["choices"][0]["message"]["content"] == "Done"
        assert not route.issues
        assert len(route.observations) == 2
        assert route.observations[0]["transport_error"] == {"phase": "http_rejected", "status": 429}
        assert "response" not in route.observations[0]
        assert route.observations[1]["response"]["choices"][0]["message"]["content"] == "Done"


@pytest.mark.asyncio
@pytest.mark.parametrize("child", [False, True])
async def test_invalid_tool_recovery_lists_only_the_tools_offered_for_that_call(child):
    server = ModelServer({"choices": [{"message": {"role": "assistant", "content": "Continue"}}]})
    route = gateway(server, tools={"tool_names": {"read": "Read", "task": "Delegate"}})
    request = {
        "messages": [{"role": "user", "content": "Fix it"}],
        "tools": [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("read", "task")
        ],
    }
    await route.forward(request, headers={"x-session-id": "root"})
    session = "child" if child else "root"
    if child:
        request["tools"] = request["tools"][:1]
    original = {"name": "Missing", "arguments": '{ "path": "source" }'}
    server.result = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "unknown", "type": "function", "function": original}],
                }
            }
        ]
    }
    await route.forward(request, headers={"x-session-id": session})
    request["messages"] += [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "unknown",
                    "type": "function",
                    "function": {
                        "name": "invalid",
                        "arguments": json.dumps({"tool": "Missing", "error": "Available tools: read, task"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "unknown", "content": "Available tools: read, task"},
    ]
    server.result = {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    await route.forward(request, headers={"x-session-id": session})
    messages = server.requests[-1]["json"]["messages"]
    assert messages[-2]["tool_calls"][0]["function"] == original
    expected = "Read" if child else "Delegate, Read"
    assert messages[-1]["content"] == (
        f"Missing is unavailable or its arguments are invalid. Available tools: {expected}."
    )
    assert request["messages"][-1]["content"] == "Available tools: read, task"


@pytest.mark.asyncio
@pytest.mark.parametrize("captured", [True, False])
async def test_successful_invalid_alias_keeps_its_result_in_current_and_imported_history(captured):
    original = {"name": "invalid", "arguments": '{"filePath":"/app/source.py"}'}
    server = ModelServer(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "successful-read", "type": "function", "function": original},
                        ],
                    }
                }
            ]
        }
    )
    route = gateway(server, tools={"tool_names": {"read": "invalid"}})
    request = {
        "messages": [{"role": "user", "content": "Fix it"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
    }
    if captured:
        reply = await route.forward(request, headers={"x-session-id": "root"})
        assert json.loads(reply.body)["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"
        assert route.tool_call(session_id="root", call_id="successful-read", native_tool="read")["allowed"]
    request["messages"] += [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "successful-read",
                    "type": "function",
                    "function": {"name": "read", "arguments": original["arguments"]},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "successful-read", "content": "ACTUAL_FILE_CONTENT"},
    ]
    saved = deepcopy(request)
    server.result = {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    await route.forward(request, headers={"x-session-id": "root"})
    messages = server.requests[-1]["json"]["messages"]
    assert messages[-2]["tool_calls"][0]["function"] == original
    assert messages[-1]["content"] == "ACTUAL_FILE_CONTENT"
    assert request == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [True, False])
async def test_selected_tool_json_repair_preserves_observed_arguments_without_changing_defaults(explicit):
    original = {"name": "ReadSource" if explicit else "read", "arguments": "{"}
    server = ModelServer(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "bad-json", "type": "function", "function": original}],
                    }
                }
            ]
        }
    )
    route = gateway(server, tools={"tool_names": {"read": "ReadSource"}} if explicit else None)
    request = {
        "messages": [{"role": "user", "content": "Fix it"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
    }
    reply = await route.forward(request, headers={"x-session-id": "root"})
    assert json.loads(reply.body)["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"
    repaired = {"name": "invalid", "arguments": json.dumps({"tool": "read", "error": "invalid JSON"})}
    request["messages"] += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "bad-json", "type": "function", "function": repaired}],
        },
        {"role": "tool", "tool_call_id": "bad-json", "content": "Invalid arguments"},
    ]
    server.result = {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    await route.forward(request, headers={"x-session-id": "root"})
    assert server.requests[-1]["json"]["messages"][1]["tool_calls"][0]["function"] == (
        original if explicit else repaired
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_plain_text_with_null_tool_calls_is_observed_and_returned(stream):
    message = {"role": "assistant", "content": "Done", "tool_calls": None}
    choice = {"index": 0, "finish_reason": "stop", "delta" if stream else "message": message}
    model_response = {"id": "plain-text", "choices": [choice]}
    server = ModelServer([model_response] if stream else model_response, stream=stream)
    route = gateway(server)
    result = await route.forward(
        {"messages": [{"role": "user", "content": "Fix it"}], "stream": stream},
        headers={"x-session-id": "root", "x-opencode-assistant-message-id": "a1"},
    )
    if stream:
        assert '"content": "Done"' in await streamed_body(result)
    else:
        assert json.loads(result.body) == model_response
    assert route.observations[0]["response"]["choices"][0]["message"]["content"] == "Done"
    assert not route.issues
    assert server.released == 1


@pytest.mark.asyncio
async def test_chat_stream_preserves_fragments_and_observes_model_visible_calls():
    events = [
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Inspect é",
                        "tool_calls": [{"index": 0, "id": "call-a", "type": "function", "function": {"name": "Run"}}],
                    },
                }
            ],
        },
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "Command", "arguments": '{"command":'}}]
                    },
                }
            ],
        },
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"pwd"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    server = ModelServer(events, stream=True)
    route = gateway(server, tools={"tool_names": {"bash": "RunCommand"}})
    payload = {
        "stream": True,
        "messages": [{"role": "user", "content": "Fix it"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}],
    }
    result = await route.forward(payload, headers={"x-session-id": "root", "x-opencode-assistant-message-id": "a1"})
    body = await streamed_body(result)
    parsed = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
    calls = [call for event in parsed for choice in event["choices"] for call in choice["delta"].get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "bash"
    assert "".join(call["function"].get("arguments", "") for call in calls) == '{"command":"pwd"}'
    observed = route.observations[0]["response"]["choices"][0]["message"]["tool_calls"][0]["function"]
    assert observed == {"name": "RunCommand", "arguments": '{"command":"pwd"}'}
    assert server.released == 1
    assert payload["tools"][0]["function"]["name"] == "bash"
    assert server.requests[0]["json"]["tools"][0]["function"]["name"] == "RunCommand"
    assert route.observations[0]["execution_tools"] == {"RunCommand": "bash"}


@pytest.mark.asyncio
async def test_open_code_auxiliary_calls_are_not_actor_evidence_or_prompt_overrides():
    server = ModelServer(
        {"id": "title", "choices": [{"message": {"role": "assistant", "content": "Title"}, "finish_reason": "stop"}]}
    )
    route = gateway(server, tools={"tool_names": {"bash": "Run"}})
    payload = {"messages": [{"role": "user", "content": "Generate a title"}]}
    await route.forward(payload, headers={"x-session-id": "root"})
    assert server.requests[0]["json"] == payload
    assert route.observations == []


@pytest.mark.asyncio
async def test_truncated_stream_is_not_a_completed_observation():
    server = ModelServer(
        [{"id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Partial"}}]}], stream=True
    )
    route = gateway(server)
    result = await route.forward(
        {"stream": True, "messages": [{"role": "user", "content": "Fix it"}]},
        headers={"x-session-id": "root", "x-opencode-assistant-message-id": "a1"},
    )
    with pytest.raises(ValueError, match="terminal"):
        await streamed_body(result)
    assert route.issues and server.released == 1
    assert "response" not in route.observations[0]


@pytest.mark.asyncio
async def test_tool_observations_drop_redundant_names_not_call_identity():
    server = ModelServer(
        {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}]}
    )
    route = gateway(server)
    payload = {
        "messages": [
            {"role": "user", "content": "Fix it"},
            {
                "role": "assistant",
                "content": "Checking",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "bash", "content": "/app"},
        ]
    }
    await route.forward(payload, headers={"x-session-id": "root", "x-opencode-assistant-message-id": "a2"})
    assert server.requests[0]["json"]["messages"][-1] == {"role": "tool", "tool_call_id": "call-1", "content": "/app"}
