# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paired agentic IF Resources Server: validated seeds, one real grade per episode, IF metrics."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientConnectionError
from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.task_data import validate_jsonl_rows
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.agentic_if.app import AgenticIFResourcesServer, AgenticIFResourcesServerConfig
from resources_servers.agentic_if.episode_contracts import AgenticIFVerificationInput, AgenticIFVerifyRequest
from resources_servers.agentic_if.if_grading import judge as judge_module
from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence
from resources_servers.agentic_if.task_data import TaskData
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, seal_runtime_row, validate_runtime


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
EXACTLY_DONE = [{"id": "end", "kind": "exact", "parameters": {"value": "DONE"}, "scope": {"places": ["final"]}}]
JUDGED = [{"id": "literal-rubric", "kind": "judge", "rubric": "Finish with DONE."}]


def sealed_row(checks=EXACTLY_DONE, *, text="Fix it"):
    return seal_runtime_row(
        {"instance_id": "task-a", "responses_create_params": {"input": [{"role": "user", "content": text}]}},
        RuntimeSpec(harness="opencode", task_backend="swe", checks=checks),
    )


def identity(row):
    return TaskId(taskset="agentic_if:tests", task_id=digest(row))


def seed_body(row, *, session="resources-1", episode=EPISODE):
    return ResourcesSeedSessionRequest(
        resources_session_id=session, episode_id=episode, task_id=identity(row), task_data=row
    )


def verify_body(row, evidence, *, episode=EPISODE):
    return AgenticIFVerifyRequest(
        episode_id=episode, task_id=identity(row), verification_input=AgenticIFVerificationInput(evidence=evidence)
    )


def request(session_id="cookie-1"):
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


def evidence(answer="All set. DONE"):
    return SessionEvidence(
        status="complete",
        steps=[
            {"id": "u", "session_id": "root", "role": "user", "text": "Fix it"},
            {"id": "a", "session_id": "root", "role": "assistant", "text": answer},
        ],
    )


class JudgeModelServer:
    """The judge model server as `ServerClient.post` reaches it: one Responses reply per request."""

    def __init__(self):
        self.requests = []
        self.entered = asyncio.Event()
        self.release = None
        self.failure = None

    async def post(self, *, server_name, url_path, json, **kwargs):
        self.requests.append((server_name, url_path, json))
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.failure is not None:
            raise self.failure
        text = '{"answer": "yes", "why": "The visible answer ends in DONE."}'
        reply = {
            "id": "resp",
            "object": "response",
            "created_at": 0,
            "model": json["model"],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }

        async def read():
            return json_module.dumps(reply).encode()

        return SimpleNamespace(ok=True, read=read)


json_module = json


@pytest.fixture
def judge_server():
    return JudgeModelServer()


@pytest.fixture
def server(judge_server):
    config = AgenticIFResourcesServerConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="agentic_if",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        max_concurrent_grades=1,
    )
    client = MagicMock(spec=ServerClient)
    client.post = judge_server.post
    return AgenticIFResourcesServer(config=config, server_client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["identity", "content"])
async def test_seed_rejects_a_row_whose_identity_or_receipt_does_not_hold(server, tamper):
    row = sealed_row()
    body = seed_body(row)
    if tamper == "identity":
        body = body.model_copy(update={"task_id": TaskId(taskset="agentic_if:tests", task_id="0" * 64)})
    else:
        # Same claimed identity, altered actor input: both the digest and the sealed receipt fail.
        body.task_data["responses_create_params"]["input"][0]["content"] = "Tampered"
    with pytest.raises(HTTPException) as rejected:
        await server.seed_session(request(), body)
    assert rejected.value.status_code == 422
    assert not server._seeded


@pytest.mark.asyncio
async def test_another_episode_or_a_closed_id_conflicts_with_the_session(server):
    row = sealed_row()
    await server.seed_session(request(), seed_body(row))
    other = EpisodeId(rollout_id="other", attempt=0)
    with pytest.raises(HTTPException) as reseeded:
        await server.seed_session(request(), seed_body(row, episode=other))
    with pytest.raises(HTTPException) as mismatched:
        await server.close_resources_session(
            request(), ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=other)
        )
    await server.close_resources_session(
        request(), ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=EPISODE)
    )
    assert not server._seeded
    with pytest.raises(HTTPException) as closed:
        await server.seed_session(request("cookie-2"), seed_body(row))
    assert reseeded.value.status_code == mismatched.value.status_code == closed.value.status_code == 409


def test_session_lifecycle_meets_the_resources_session_contract(server):
    # The contract also covers repeated seeds and closes of one episode, and a seed after its close.
    check_resources_session_contract(server.setup_webserver(), seed_body(sealed_row()), keeps_state=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "if_all"), [("DONE", 1), ("All set.", 0)])
async def test_verify_grades_the_seeded_checks_against_the_recorded_evidence(server, judge_server, answer, if_all):
    row = sealed_row()
    await server.seed_session(request(), seed_body(row))
    verification = await server.verify(request(), verify_body(row, evidence(answer)))
    assert verification.if_result.if_all == if_all
    assert [check.id for check in verification.if_result.checks] == ["end"]
    assert not judge_server.requests  # a deterministic check asks no judge


@pytest.mark.asyncio
async def test_a_judged_check_asks_the_configured_judge_model_server(server, judge_server):
    row = sealed_row(JUDGED)
    await server.seed_session(request(), seed_body(row))
    verification = await server.verify(request(), verify_body(row, evidence()))
    assert verification.if_result.if_all == 1
    assert len(judge_server.requests) == 3  # the run context asks three times
    server_name, url_path, body = judge_server.requests[0]
    assert all(sent == (server_name, url_path, body) for sent in judge_server.requests)
    model = load_config()["model"]
    assert (server_name, url_path) == ("judge", "/v1/responses")
    assert body["model"] == model["name"] and body["temperature"] == model["temperature"]
    assert "Finish with DONE." in body["input"][0]["content"] and "All set. DONE" in body["input"][0]["content"]
    votes = verification.if_result.checks[0].details["units"][0]["metadata"]["votes"]
    assert [vote["status"] for vote in votes] == ["pass"] * 3


@pytest.mark.asyncio
async def test_a_judge_that_cannot_answer_makes_the_check_an_error_not_a_failure(server, judge_server, monkeypatch):
    monkeypatch.setattr(judge_module.time, "sleep", lambda seconds: None)
    judge_server.failure = ClientConnectionError("judge unreachable")
    row = sealed_row(JUDGED)
    await server.seed_session(request(), seed_body(row))
    verification = await server.verify(request(), verify_body(row, evidence()))
    [check] = verification.if_result.checks
    assert check.status == "error" and verification.if_result.if_all is None
    assert "No judge verdict after 4 attempt(s): ClientConnectionError" in check.explanation
    assert len(judge_server.requests) == 3 * 4  # each of the run context's three votes, with judge.yaml's attempts


@pytest.mark.asyncio
async def test_verify_requires_the_seeded_episode(server):
    row = sealed_row()
    with pytest.raises(HTTPException) as unseeded:
        await server.verify(request(), verify_body(row, evidence()))
    assert unseeded.value.status_code == 404
    await server.seed_session(request(), seed_body(row))
    with pytest.raises(HTTPException) as mismatch:
        await server.verify(request(), verify_body(row, evidence(), episode=EpisodeId(rollout_id="other", attempt=0)))
    assert mismatch.value.status_code == 409


@pytest.mark.asyncio
async def test_cancelled_verify_keeps_grading_capacity_until_the_judge_finishes(server, judge_server):
    judge_server.release = asyncio.Event()
    first_row, second_row = sealed_row(JUDGED), sealed_row(JUDGED, text="Other task")
    await server.seed_session(request("first"), seed_body(first_row, session="resources-first"))
    await server.seed_session(request("second"), seed_body(second_row, session="resources-second"))
    first = asyncio.create_task(server.verify(request("first"), verify_body(first_row, evidence())))
    await asyncio.wait_for(judge_server.entered.wait(), 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(
        server.verify(request("second"), verify_body(second_row, evidence("Second answer. DONE")))
    )
    await asyncio.sleep(0.05)
    assert len(judge_server.requests) == 1 and not second.done()  # The cancelled caller's job still owns the slot.
    judge_server.release.set()
    assert (await second).if_result.if_all == 1
    assert not server._grade_jobs


def test_metrics_report_if_rates_beside_the_task_reward(server):
    def rollout(reward, if_all, if_frac, statuses):
        checks = [{"id": str(index), "status": status, "explanation": ""} for index, status in enumerate(statuses)]
        return {"reward": reward, "if_result": {"checks": checks, "if_all": if_all, "if_frac": if_frac}}

    metrics = server.compute_metrics(
        [
            [rollout(1.0, 1, 1.0, ["pass"]), rollout(0.0, 1, 1.0, ["pass"])],
            [rollout(1.0, 0, 0.5, ["pass", "fail"])],
            [rollout(1.0, None, None, ["not_applicable"]), rollout(0.0, None, None, ["error"])],
        ]
    )
    assert metrics["if/graded"] == 3 and metrics["if/not_applicable"] == 1 and metrics["if/error"] == 1
    assert metrics["mean/if_all"] == pytest.approx(2 / 3)
    assert metrics["mean/if_frac"] == pytest.approx(2.5 / 3)
    assert metrics["mean/task_and_if"] == pytest.approx(1 / 3)


def test_task_data_selects_the_rows_own_backend_schema_and_requires_its_digest_identity():
    schema = TypeAdapter(TaskData)
    swe = seal_runtime_row(
        {"instance_id": "task-a", "workdir": "/testbed", "image_url": "registry/task-a:1"}
        | {"responses_create_params": {"input": [{"role": "user", "content": "Fix it"}]}},
        RuntimeSpec(harness="opencode", task_backend="swe", checks=EXACTLY_DONE),
    )
    tb21 = seal_runtime_row(
        {"task_name": "terminal-bench/a", "docker_image": "registry/a:1", "task_folder": "tasks/a"}
        | {"workdir": "/app", "responses_create_params": {"input": [{"role": "user", "content": "Fix it"}]}},
        RuntimeSpec(harness="opencode", task_backend="tb21", checks=EXACTLY_DONE),
    )
    assert type(schema.validate_python({"task_id": digest(swe)} | swe)).__name__ == "SWETask"
    assert type(schema.validate_python({"task_id": digest(tb21)} | tb21)).__name__ == "TerminalBench21Task"
    # A backend's native fields are required by that backend's schema, not by the other's.
    with pytest.raises(ValidationError, match="task_folder"):
        schema.validate_python({"task_id": digest(swe)} | swe | {"agentic_if": tb21["agentic_if"]})
    with pytest.raises(ValidationError, match="task_id"):
        schema.validate_python({"task_id": "not-a-digest"} | swe)


def test_committed_example_rows_validate_clean_and_carry_their_digest():
    example = Path(__file__).parents[1] / "data" / "example.jsonl"
    lines = example.read_text().splitlines()
    report = validate_jsonl_rows("agentic_if", TypeAdapter(TaskData), str(example), lines)
    assert report.rows == len(lines) and report.clean, report.summary()
    for row in map(json.loads, lines):
        content = {key: value for key, value in row.items() if key != "task_id"}
        assert row["task_id"] == digest(content)
        validate_runtime(content)
