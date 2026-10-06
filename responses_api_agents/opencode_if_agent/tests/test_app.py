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
