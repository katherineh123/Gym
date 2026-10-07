# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest
from fastapi import Request
from omegaconf import OmegaConf

from nemo_gym.task_variants.builder import build_variant
from responses_api_agents.opencode_if_agent.tests.test_builder import constraint, row


def make_request(body):
    return Request({"type": "http", "headers": [], "session": {}, "path": "/run"}, receive=lambda: None)


def config():
    from responses_api_agents.opencode_if_agent.app import OpenCodeIFConfig

    return OpenCodeIFConfig(
        name="if_agent",
        host="0.0.0.0",
        port=9001,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resources"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        sandbox_provider="sandbox",
        sandbox_config={},
        sandbox_timeout=60,
        opencode_version="1.17.11",
        opencode_max_context_window=32768,
    )


def client():
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    return ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=9000), global_config_dict=OmegaConf.create({})
    )


def test_tampered_actor_input_is_rejected_before_execution():
    from responses_api_agents.opencode_if_agent.app import validate_variant

    source = build_variant(row(), {"instructions": [constraint()]})
    assert validate_variant(source)["base_task_id"] == "public-example-1"
    source["responses_create_params"]["input"][0]["content"] = "Replaced prompt"
    with pytest.raises(ValueError, match="actor input"):
        validate_variant(source)


def test_tampered_rubric_is_rejected():
    from responses_api_agents.opencode_if_agent.app import validate_variant

    source = build_variant(row(), {"instructions": [constraint()]})
    source["task_variant"]["instructions"][0]["rubric"] = "Always pass"
    with pytest.raises(ValueError, match="variant ID"):
        validate_variant(source)


def test_baseline_content_blocks_survive_request_model_normalization():
    from responses_api_agents.opencode_if_agent.app import OpenCodeIFRunRequest, validate_variant

    source = row()
    source["responses_create_params"]["input"] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Fix it."}],
        }
    ]
    generated = build_variant(source, {})
    parsed = OpenCodeIFRunRequest.model_validate(generated)
    assert validate_variant(parsed.model_dump(mode="json"))["base_task_id"] == source["instance_id"]


def test_multiple_workers_rejected_for_run_scoped_model_route():
    from responses_api_agents.opencode_if_agent.app import OpenCodeIFConfig

    fields = config().model_dump()
    fields["num_workers"] = 2
    with pytest.raises(ValueError, match="one worker"):
        OpenCodeIFConfig.model_validate(fields)


def test_model_boundary_route_is_registered():
    from responses_api_agents.opencode_if_agent.app import OpenCodeIFAgent

    server = OpenCodeIFAgent(config=config(), server_client=client())
    app = server.setup_webserver()
    assert "/if/{attempt_id}/v1/chat/completions" in {route.path for route in app.routes}


@pytest.mark.asyncio
async def test_native_handoff_keeps_grade_separate_and_cleans_up(monkeypatch):
    from responses_api_agents.opencode_if_agent import app as module

    native = module.OpenCodeSandboxedAgent
    seen = []

    async def native_run(self, request, body):
        state = request.state._ng_if_run
        assert state.attempt_id in self._variant_runs
        assert "PRIVATE" not in json.dumps(body.responses_create_params.input, default=str)
        state.request_count = 1
        state.first_request = {"messages": [{"role": "user", "content": "End with FINISH."}], "tools": []}
        seen.append(state.attempt_id)
        return SimpleNamespace(
            reward=0.0,
            opencode_failed=False,
            model_patch="diff",
            response=SimpleNamespace(output=[]),
        )

    monkeypatch.setattr(native, "run", native_run)
    agent = module.OpenCodeIFAgent(config=config(), server_client=client())
    body = module.OpenCodeIFRunRequest.model_validate(build_variant(row(), {"instructions": [constraint()]}))
    request = make_request(body)
    result = await agent.run(request, body)
    assert result.reward == 0.0
    assert result.if_result["error"] == "missing_judge_api_key"
    assert result.variant_receipt["attempt_id"] == seen[0]
    assert agent._variant_runs == {}
    assert not hasattr(request.state, "_ng_if_run")


@pytest.mark.asyncio
async def test_failed_native_run_does_not_leave_an_active_model_route(monkeypatch):
    from responses_api_agents.opencode_if_agent import app as module

    async def fail(*args, **kwargs):
        raise RuntimeError("sandbox unavailable")

    monkeypatch.setattr(module.OpenCodeSandboxedAgent, "run", fail)
    agent = module.OpenCodeIFAgent(config=config(), server_client=client())
    body = module.OpenCodeIFRunRequest.model_validate(build_variant(row(), {}))
    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        await agent.run(make_request(body), body)
    assert agent._variant_runs == {}


@pytest.mark.asyncio
async def test_export_uses_model_visible_tool_names_and_incomplete_delegation_is_not_graded(monkeypatch):
    from pydantic import BaseModel

    from responses_api_agents.opencode_if_agent import app as module

    class ToolCall(BaseModel):
        type: str = "function_call"
        name: str
        arguments: str = "{}"

    async def native_run(self, request, body):
        state = request.state._ng_if_run
        state.request_count = 1
        state.first_request = {"messages": [], "tools": []}
        return SimpleNamespace(
            reward=0,
            opencode_failed=False,
            model_patch="",
            ng_trajectory=None,
            response=SimpleNamespace(output=[ToolCall(name="task"), ToolCall(name="bash")]),
        )

    monkeypatch.setattr(module.OpenCodeSandboxedAgent, "run", native_run)
    agent = module.OpenCodeIFAgent(config=config(), server_client=client())
    body = module.OpenCodeIFRunRequest.model_validate(
        build_variant(
            row(),
            {
                "tool_names": {"bash": "shell"},
                "instructions": [constraint()],
            },
        )
    )
    result = await agent.run(make_request(body), body)
    assert result.response.output[-1].name == "shell"
    assert result.if_result["error"] == "incomplete_rollout_evidence"


@pytest.mark.asyncio
async def test_proxy_ignores_auxiliary_first_request_and_isolates_concurrent_attempts(monkeypatch):
    import asyncio

    from responses_api_agents.opencode_if_agent import app as module

    forwarded = []

    async def http(method, url, **kwargs):
        forwarded.append((url, kwargs["json"], kwargs["headers"]))

        async def response_json():
            return {"choices": [{"message": {"content": "ok"}}]}

        return SimpleNamespace(raise_for_status=lambda: None, release=lambda: None, json=response_json)

    monkeypatch.setattr(module, "http_request", http)
    agent = module.OpenCodeIFAgent(config=config(), server_client=client())
    states = [
        module.VariantRun(
            str(i), {"tool_names": {}, "instructions": []}, f"ONLY_{i}", upstream_url=f"http://upstream/{i}"
        )
        for i in range(2)
    ]
    agent._variant_runs.update({state.attempt_id: state for state in states})

    def req(payload):
        async def body():
            return payload

        return SimpleNamespace(
            json=body,
            cookies={},
            headers={
                "x-session-id": "native-session",
                "x-opencode-assistant-message-id": "native-message",
                "authorization": "DO-NOT-FORWARD",
            },
        )

    await agent.variant_chat_completions(req({"messages": [{"role": "system", "content": "Generate a title"}]}), "0")
    assert states[0].first_request is None
    payload = {
        "messages": [{"role": "user", "content": "Fix it"}],
        "tools": [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
        ],
    }
    await asyncio.gather(*(agent.variant_chat_completions(req(payload), state.attempt_id) for state in states))
    for index, state in enumerate(states):
        text = json.dumps(state.first_request)
        assert f"ONLY_{index}" in text
        assert f"ONLY_{1 - index}" not in text
    assert forwarded[-1][2]["x-session-id"] == "native-session"
    assert forwarded[-1][2]["x-opencode-assistant-message-id"] == "native-message"
    assert "DO-NOT-FORWARD" not in json.dumps(forwarded[-1][2])


def test_if_agent_declares_native_assistant_message_correlation_header():
    from responses_api_agents import opencode_if_agent, opencode_sandboxed_agent

    assert opencode_if_agent._assistant_message_header == opencode_sandboxed_agent._assistant_message_header


@pytest.mark.asyncio
async def test_multi_block_baseline_is_normalized_for_native_execution(monkeypatch):
    from responses_api_agents.opencode_if_agent import app as module

    async def native_run(self, request, body):
        assert body.responses_create_params.input[0].content == "First\nSecond"
        return SimpleNamespace(reward=0, opencode_failed=False, response=SimpleNamespace(output=[]))

    source = row()
    source["responses_create_params"]["input"][0]["content"] = [
        {"type": "input_text", "text": "First"},
        {"type": "input_text", "text": "Second"},
    ]
    body = module.OpenCodeIFRunRequest.model_validate(build_variant(source, {}))
    monkeypatch.setattr(module.OpenCodeSandboxedAgent, "run", native_run)
    agent = module.OpenCodeIFAgent(config=config(), server_client=client())
    await agent.run(make_request(body), body)


@pytest.mark.asyncio
@pytest.mark.parametrize("child_answer", [None, [{"type": "function_call", "name": "bash", "arguments": "{}"}]])
async def test_child_evidence_is_attributed_without_reasoning_and_gaps_fail_closed(monkeypatch, child_answer):
    from pydantic import BaseModel

    from responses_api_agents.opencode_if_agent import app as module

    class ToolCall(BaseModel):
        type: str = "function_call"
        name: str = "task"

    async def native_run(self, request, body):
        state = request.state._ng_if_run
        state.request_count = 2
        state.first_request = {"messages": [], "tools": []}
        return SimpleNamespace(
            reward=0,
            opencode_failed=False,
            response=SimpleNamespace(output=[ToolCall()]),
            ng_trajectory=SimpleNamespace(
                gaps=[],
                turns=[
                    SimpleNamespace(
                        invocation_id="root", turn_no=1, answer=[{"type": "function_call", "name": "task"}]
                    ),
                    SimpleNamespace(
                        invocation_id="child", turn_no=1, answer=child_answer, reasoning_content="PRIVATE"
                    ),
                ],
            ),
        )

    async def grade(instructions, evidence):
        if child_answer is None:
            assert not evidence["complete"]
        else:
            assert evidence["complete"]
            assert evidence["turns"][1] == {
                "invocation_id": "child",
                "turn_no": 1,
                "answer": [
                    {"type": "function_call", "name": "shell", "arguments": "{}"},
                ],
            }
        assert "PRIVATE" not in json.dumps(evidence)
        return {"status": "completed"}

    monkeypatch.setattr(module.OpenCodeSandboxedAgent, "run", native_run)
    agent = module.OpenCodeIFAgent(config=config(), server_client=client())
    monkeypatch.setattr(agent._judge, "grade", grade)
    body = module.OpenCodeIFRunRequest.model_validate(build_variant(row(), {"tool_names": {"bash": "shell"}}))
    await agent.run(make_request(body), body)
