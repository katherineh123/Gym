# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lifecycle fixture: the real runner and scorer, with only remote execution and HTTP replaced."""

import asyncio
import json
import threading
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf

from environment_servers.agentic_if import app as module
from environment_servers.agentic_if.drivers.opencode import OpenCodeDriver
from environment_servers.agentic_if.runtime import RunOutcome
from environment_servers.agentic_if.tests.support import Transport, response, seeded
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.server_utils import SESSION_ID_KEY, BaseServerConfig, ServerClient
from resources_servers.agentic_if.app import AgenticIFResourcesServer, AgenticIFResourcesServerConfig
from resources_servers.agentic_if.episode_contracts import AgenticIFVerifyRequest
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence


@pytest.fixture
def setup(tmp_path, monkeypatch):
    events = []
    transports = []
    verifications = []
    model_thread = threading.get_ident()
    controls = SimpleNamespace(
        owners={},
        prepare_error=False,
        execution_error=False,
        evidence_error=False,
        reward=0.5,
        entered=asyncio.Event(),
        release=None,
        judge_entered=None,
        judge_release=None,
        seed_entered=asyncio.Event(),
        seed_release=None,
        verify_extras={},
        verify_entered=asyncio.Event(),
        verify_release=None,
        rollout_ids=[],
        actor_calls=[],
        actor_response={"id": "r-1", "choices": [{"message": {"role": "assistant", "content": "Done"}}]},
        scoring=[],
        scoring_seeds=[],
        scoring_seed_error=False,
        scoring_verify_error=False,
        timeline=[],
    )
    scorer = AgenticIFResourcesServer(
        config=AgenticIFResourcesServerConfig(
            host="127.0.0.1",
            port=9002,
            name="scoring",
            entrypoint="app.py",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            max_concurrent_grades=1,
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=9000), global_config_dict=OmegaConf.create({})
        ),
    )
    controls.scorer = scorer

    async def score(path, body, cookies):
        """The real paired Resources Server, reached in-process with cookie-keyed sessions as over HTTP."""
        session = (cookies or {}).get("session") or uuid4().hex
        caller = SimpleNamespace(session={SESSION_ID_KEY: session})
        if path == "/seed_session":
            controls.scoring.append("seed")
            controls.scoring_seeds.append(deepcopy(body))
            if controls.scoring_seed_error:
                raise HTTPException(422, "rejected task")
            reply = await scorer.seed_session(caller, ResourcesSeedSessionRequest.model_validate(body))
        elif path == "/verify":
            controls.scoring.append("verify")
            if controls.scoring_verify_error:
                raise OSError("scorer unavailable")
            reply = await scorer.verify(caller, AgenticIFVerifyRequest.model_validate(body))
        else:
            controls.scoring.append("close")
            reply = await scorer.close_resources_session(caller, ResourcesCloseSessionRequest.model_validate(body))
        value = reply.model_dump(mode="json")

        async def read():
            return json.dumps(value).encode()

        return SimpleNamespace(ok=True, read=read, cookies={"session": session})

    async def post(self, **kwargs):
        controls.timeline.append(f"{kwargs['server_name']}:{kwargs['url_path']}")
        if kwargs["server_name"] == "scoring":
            return await score(kwargs["url_path"], kwargs["json"], kwargs.get("cookies"))
        if kwargs["server_name"] == "judge":
            return await judge(kwargs["json"])
        if kwargs["server_name"] == "actor":
            controls.actor_calls.append((kwargs["url_path"], current_rollout_id()))

            async def actor_json():
                return deepcopy(controls.actor_response)

            return SimpleNamespace(ok=True, json=actor_json, release=lambda: None)
        assert kwargs["server_name"] == "tasks"
        controls.rollout_ids.append(current_rollout_id())
        if kwargs["url_path"] == "/seed_session":
            events.append("prepare")
            if controls.prepare_error:
                raise OSError("seed failed")
            controls.seed_entered.set()
            if controls.seed_release is not None:
                await controls.seed_release.wait()
            owner = kwargs["json"]["resources_session_id"]
            value = seeded(workdir="/app", session_id=owner, sandbox_id=owner)
        elif kwargs["url_path"] == "/close_session":
            owned = controls.owners.get(kwargs["json"]["resources_session_id"])
            if owned is not None:
                await owned.stop()
            else:
                events.append("cleanup")
            value = {"resources_session_id": kwargs["json"]["resources_session_id"]}
        else:
            events.append("verify")
            verifications.append(deepcopy(kwargs["json"]))
            controls.verify_entered.set()
            if controls.verify_release is not None:
                await controls.verify_release.wait()
            value = {"reward": controls.reward, "evaluation_completed": True} | controls.verify_extras

        async def read():
            return json.dumps(value).encode()

        return SimpleNamespace(ok=True, read=read, cookies={})

    async def connect(descriptor, *, provider):
        sandbox = Transport()
        transports.append(sandbox)
        controls.owners[descriptor["sandbox_id"]] = sandbox
        stop = sandbox.stop

        async def close():
            events.append("cleanup")
            await stop()

        sandbox.stop = close
        return sandbox

    async def execute(self, context):
        events.append("execute")
        controls.timeline.append("execute")
        controls.rollout_ids.append(current_rollout_id())
        controls.entered.set()
        if controls.release is not None:
            await controls.release.wait()
        if controls.execution_error:
            raise RuntimeError("native harness launch failed")
        assert context.sandbox.workdir == "/app"
        assert context.spec.harness == "opencode"
        return RunOutcome(
            response=response(),
            status="completed",
            evidence=SessionEvidence(
                status="incomplete" if controls.evidence_error else "complete",
                issues=["missing tool observation"] if controls.evidence_error else [],
                steps=[
                    {
                        "id": "u",
                        "session_id": context.run_id,
                        "role": "user",
                        "text": context.params.input[-1].content,
                    },
                    {"id": "a", "session_id": context.run_id, "role": "assistant", "text": "DONE"},
                ],
            ),
        )

    async def judge(body):
        """The scorer's judge model server, as `call_judge` reaches it: one Responses reply."""
        # Grading runs in a worker thread, but its judge request runs on the event loop, where Gym's client lives.
        assert threading.get_ident() == model_thread
        assert transports and transports[-1].stops == 1
        events.append("grade")
        if controls.judge_entered:
            controls.judge_entered.set()
        if controls.judge_release:
            await controls.judge_release.wait()
        text = '{"answer":"yes","why":"The visible answer ends in DONE."}'
        reply = {
            "id": "judge-reply",
            "object": "response",
            "created_at": 0,
            "model": body["model"],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }

        async def read():
            return json.dumps(reply).encode()

        return SimpleNamespace(ok=True, read=read)

    monkeypatch.setattr(ServerClient, "post", post)
    monkeypatch.setattr(module, "sandbox_server_url", lambda name, *, require_reachable: "http://gym:9001")
    monkeypatch.setattr(module.resources.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(module.resources, "create_provider", lambda config: SimpleNamespace())
    monkeypatch.setattr(OpenCodeDriver, "execute", execute)
    binary = tmp_path / "opencode"
    binary.write_bytes(b"pin")
    config = module.AgenticIFConfig(
        host="127.0.0.1",
        port=9001,
        name="agentic_if",
        entrypoint="app.py",
        model_server={"type": "responses_api_models", "name": "actor"},
        resources_server={"type": "resources_servers", "name": "scoring"},
        task_resources={"swe": {"server": {"type": "resources_servers", "name": "tasks"}}},
        opencode={"binary_path": binary, "binary_sha256": "0" * 64, "version": "1.17.11"},
        artifacts_dir=tmp_path / "artifacts",
        max_concurrent_episodes=1,
        queue_timeout_seconds=30,
        cleanup_timeout_seconds=30,
    )
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=9000),
        global_config_dict=OmegaConf.create({"sandbox": {"opensandbox": {}}}),
    )
    agent = module.AgenticIFEnvironment(config=config, server_client=client)
    return agent, controls, events, transports, verifications
