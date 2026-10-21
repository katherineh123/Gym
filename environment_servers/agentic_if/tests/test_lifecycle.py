# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real runner and scorer; replace only remote execution/HTTP."""

import asyncio
import json
from copy import deepcopy

import pytest
from fastapi import Request

from environment_servers.agentic_if import app as module
from environment_servers.agentic_if.tests.support import episode, run_episode, task_row
from nemo_gym.rollout_correlation import current_rollout_id
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.materialization import validate_task_identity
from resources_servers.agentic_if.task_variants.runtime import ArgumentMatch, RuntimeSpec, seal_runtime_row


def test_raw_compaction_capture_can_be_saved_losslessly_even_when_response_view_is_incomplete(tmp_path):
    path = tmp_path / "evidence.json"
    payload = {
        "evidence": {"status": "incomplete", "issues": ["missing_tool_observation:c1"]},
        "model_observations": [{"request": {"messages": [{"role": "tool", "content": "😀\ud83d"}]}}],
    }
    before = deepcopy(payload)
    module._save(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == before
    assert payload == before
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [0.0, 0.5, 1.0])
async def test_native_episode_runs_one_ordered_lifecycle_and_keeps_task_reward_and_checks(setup, reward):
    """The collector's native task envelope reaches the lifecycle directly; grading runs after cleanup."""
    from pydantic import BaseModel, ConfigDict

    from nemo_gym.episode_types import BaseEpisodeRequest

    environment, controls, events, transports, verifications = setup
    controls.reward = reward
    row = task_row()

    class TaskInput(BaseModel):
        model_config = ConfigDict(extra="allow")

    request = BaseEpisodeRequest[TaskInput].model_validate(
        {
            "episode_id": {"rollout_id": "native-test", "attempt": 2},
            "task": {"task_id": {"taskset": "agentic_if:acceptance", "task_id": digest(row)}, "task_input": row},
        }
    )
    body = environment.request_model.model_validate(request.model_dump(mode="json"))
    endpoint = next(route.endpoint for route in environment.setup_webserver().routes if route.path == "/run")
    response = await endpoint(body=body)
    assert (response.episode_id, response.task_id) == (request.episode_id, request.task.task_id)
    assert response.failure is None
    result = response.result
    assert (result.reward, result.execution_status, result.if_result.if_all) == (reward, "completed", 1)
    assert events == ["prepare", "execute", "verify", "cleanup", "grade", "grade", "grade"]
    assert controls.rollout_ids == ["native-test-a2"] * 4
    assert current_rollout_id() is None
    assert len(verifications) == 1 and transports[0].stops == 1
    assert "agentic_if" not in verifications[0]
    assert environment._attempts == {}
    # The scoring session is seeded before any allocation, verifies after the task closes, and closes last.
    assert controls.scoring == ["seed", "verify", "close"]
    timeline = controls.timeline
    assert timeline[0] == "scoring:/seed_session" and timeline[-1] == "scoring:/close_session"
    assert timeline.index("tasks:/close_session") < timeline.index("scoring:/verify")
    assert not controls.scorer._seeded
    # A completed result retains its checks for offline rescoring.
    saved = json.loads((environment.config.artifacts_dir / result.run_id / "result.json").read_text())
    assert saved["agentic_if"] == result.model_dump(mode="json")["agentic_if"] == row["agentic_if"]


@pytest.mark.asyncio
async def test_scorer_rejection_fails_at_seed_before_allocating_task_resources(setup):
    agent, controls, events, transports, _ = setup
    controls.scoring_seed_error = True
    response = await agent.run_request(episode(task_row()))
    assert response.result is None
    assert response.failure.stage == "seed"
    assert response.failure.failure_kind == "agentic_if:invalid_task"
    assert response.failure.terminal
    assert events == [] and not transports
    assert controls.scoring == ["seed", "close"]


@pytest.mark.asyncio
async def test_collated_flat_row_keeps_its_digest_identity_and_reaches_the_scorer_sealed(setup):
    agent, controls, _, _, _ = setup
    row = task_row()
    collated = module.AgenticIFRequest.model_validate(
        {
            "episode_id": {"rollout_id": "collated", "attempt": 0},
            # Gym collation keeps the flat row's id inside task_input.
            "task": {
                "task_id": {"taskset": "agentic_if:tests", "task_id": digest(row)},
                "task_input": {"task_id": digest(row)} | row,
            },
        }
    )
    response = await agent.run_request(collated)
    assert response.failure is None, response.failure
    assert response.result.if_result.if_all == 1
    assert controls.scoring_seeds[0]["task_data"] == row


def test_materialized_content_cannot_reuse_an_unrelated_task_identity():
    content = seal_runtime_row(
        {"responses_create_params": {"input": [{"role": "user", "content": "Fix"}]}},
        RuntimeSpec(harness="opencode", task_backend="swe"),
    )
    row = {"task_id": {"taskset": "agentic_if:tests", "task_id": "0" * 64}, "task_input": content}
    with pytest.raises(ValueError, match="identity"):
        validate_task_identity(row)
    with pytest.raises(ValueError, match="identity"):
        module.AgenticIFRequest.model_validate({"task": row, "episode_id": {"rollout_id": "id", "attempt": 0}})


@pytest.mark.asyncio
async def test_unreachable_scorer_is_a_terminal_verification_failure(setup):
    agent, controls, events, transports, _ = setup
    controls.scoring_verify_error = True
    response = await agent.run_request(episode(task_row()))
    assert response.result is None
    assert response.failure.stage == "verification"
    assert response.failure.failure_kind == "agentic_if:infrastructure_error"
    assert response.failure.terminal
    assert events == ["prepare", "execute", "verify", "cleanup"]
    assert transports[0].stops == 1
    assert controls.scoring == ["seed", "verify", "close"]


@pytest.mark.asyncio
async def test_native_resource_failure_uses_typed_episode_failure_and_is_collectible(setup):
    from nemo_gym.rollout_collection import _episode_record

    environment, controls, _, transports, _ = setup
    controls.verify_extras = {"evaluation_completed": False, "failure_reason": "Verifier unavailable"}
    reply = await environment.run_request(episode(task_row()))
    assert reply.result is None
    assert reply.failure.failure_reason == "Verifier unavailable"
    assert reply.failure.terminal
    partial = reply.failure.partial_response
    assert partial.if_result.if_all == 1  # Valid IF evidence is retained independently.
    assert transports[0].stops == 1
    record = _episode_record(reply.model_dump(mode="json"))
    assert record["_ng_failure_class"] == "environment_server_failed"
    assert record["_ng_failure_partial_response"]["run_id"] == partial.run_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edit",
    [
        {"input": [{"role": "user", "content": "Tampered"}]},
        {"max_output_tokens": 1},
        {"temperature": 1.9},
        {"top_p": 0.1},
    ],
    ids=["input", "max_output_tokens", "temperature", "top_p"],
)
async def test_bad_receipt_rejected_before_allocating_resources(setup, edit):
    agent, _, events, _, _ = setup
    row = task_row()
    row["responses_create_params"].update(edit)
    response = await agent.run_request(episode(row))
    assert response.result is None
    assert "receipt" in response.failure.failure_reason
    assert response.failure.terminal
    assert events == []


def test_event_regex_rejects_python_only_syntax_before_allocation():
    with pytest.raises(ValueError, match="regex"):
        ArgumentMatch(field="command", regex="(?P<command>pwd)")


@pytest.mark.asyncio
async def test_incomplete_evidence_is_not_a_pass_and_does_not_call_judge(setup):
    agent, controls, events, _, _ = setup
    controls.evidence_error = True
    result = await run_episode(agent, task_row(), failed=True)
    assert result.if_result.if_all is None
    assert result.if_result.checks[0].status == "error"
    assert events == ["prepare", "execute", "verify", "cleanup"]


@pytest.mark.asyncio
async def test_harness_exception_cleans_up_and_cannot_be_reported_as_success(setup):
    agent, controls, events, transports, _ = setup
    controls.execution_error = True
    result = await run_episode(agent, task_row(), failed=True)
    assert result.execution_status == "error"
    assert result.if_result.if_all is None
    assert transports[0].stops == 1 and "grade" not in events
    assert not agent._attempts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entered", "release", "expected"),
    [
        ("seed_entered", "seed_release", ["prepare", "cleanup"]),
        ("entered", "release", ["prepare", "execute", "cleanup"]),
        ("verify_entered", "verify_release", ["prepare", "execute", "verify", "cleanup"]),
    ],
    ids=["allocation", "execution", "verification"],
)
async def test_cancellation_closes_the_owner_without_waiting_for_the_stage_or_running_later_ones(
    setup, entered, release, expected
):
    agent, controls, events, transports, verifications = setup
    setattr(controls, release, asyncio.Event())
    run = asyncio.create_task(run_episode(agent, task_row()))
    await getattr(controls, entered).wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    getattr(controls, release).set()
    assert events == expected  # Nothing is graded.
    assert len(verifications) == expected.count("verify")  # Verification is never reissued.
    # A sandbox that never attached needs no stop; an attached one is stopped once.
    assert [sandbox.stops for sandbox in transports] == ([] if entered == "seed_entered" else [1])
    assert not agent._attempts


@pytest.mark.asyncio
async def test_cancelled_judge_waiter_keeps_grade_capacity_until_worker_exits(setup):
    agent, controls, events, transports, _ = setup
    controls.judge_entered, controls.judge_release = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(run_episode(agent, task_row()))
    await asyncio.wait_for(controls.judge_entered.wait(), 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(run_episode(agent, task_row(text="Other task")))
    for _ in range(50):
        if len(transports) == 2 and transports[1].stops:
            break
        await asyncio.sleep(0.01)
    assert len(transports) == 2 and transports[1].stops == 1  # The sandbox is released before grading waits.
    assert events.count("grade") == 1
    assert not second.done()
    controls.judge_release.set()
    assert (await second).if_result.if_all == 1


@pytest.mark.asyncio
async def test_max_concurrent_episodes_admits_the_next_episode_only_after_the_last_one_ends(setup):
    agent, controls, events, transports, _ = setup
    assert agent.config.max_concurrent_episodes == 1
    controls.release = asyncio.Event()
    first = asyncio.create_task(run_episode(agent, task_row()))
    await controls.entered.wait()
    second = asyncio.create_task(run_episode(agent, task_row(text="Other task")))
    await asyncio.sleep(0.05)
    assert events.count("prepare") == 1 and controls.scoring.count("seed") == 1  # Not admitted: nothing allocated.
    controls.release.set()
    await asyncio.gather(first, second)
    assert events.count("prepare") == 2 and [sandbox.stops for sandbox in transports] == [1, 1]


@pytest.mark.asyncio
async def test_failed_preparation_returns_partial_evidence_in_native_failure(setup, caplog):
    agent, controls, events, _, _ = setup
    controls.prepare_error = True
    result = await run_episode(agent, task_row(), failed=True)
    assert events == ["prepare", "cleanup"]
    assert result.mask_sample and result.execution_status == "error"
    assert result.if_result.if_all is None
    assert result.if_result.checks[0].status == "error"
    assert any(record.exc_info and result.run_id in record.message for record in caplog.records)


def test_boolean_backend_identity_cannot_become_uid_one(setup):
    agent, _, events, _, _ = setup
    config = agent.config.model_dump()
    config["task_resources"]["swe"]["agent_user"] = True
    with pytest.raises(ValueError, match="agent_user"):
        module.AgenticIFConfig.model_validate(config)
    assert not events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capture", "training"),
    [
        (None, False),
        # This server calls no agent, so only capture for all agents covers its model calls.
        ({"enabled": True}, False),
        ({"enabled": True, "all_agents": True}, True),
    ],
    ids=["observability", "agent-scoped-capture", "all-agents-capture"],
)
async def test_model_callback_preserves_rollout_and_training_capture_after_crossing_http_boundary(
    setup, capture, training
):
    agent, controls, _, _, _ = setup
    if capture is not None:
        agent.server_client.global_config_dict["token_id_capture"] = capture
    else:
        agent.server_client.global_config_dict["observability_enabled"] = True
    controls.release = asyncio.Event()
    row = task_row(check=False)
    endpoint = next(route.endpoint for route in agent.setup_webserver().routes if route.path == "/run")
    run = asyncio.create_task(endpoint(body=episode(row, rollout_id="collector-episode-21")))
    try:
        await controls.entered.wait()
        attempt_id, attempt = next(iter(agent._attempts.items()))
        assert current_rollout_id() is None  # A new inbound model request has its own context.

        async def receive():
            return {
                "type": "http.request",
                "body": json.dumps({"messages": [{"role": "user", "content": "Fix it"}], "tools": []}).encode(),
            }

        callback = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/model",
                "headers": [
                    (b"authorization", ("Bearer " + attempt.context.model.api_key).encode()),
                    (b"x-opencode-assistant-message-id", b"a1"),
                ],
            },
            receive,
        )
        await agent.model_chat(callback, attempt_id)
        suffix = "/training-token-capture" if training else ""
        assert controls.actor_calls == [
            ("/ng-rollout/collector-episode-21" + suffix + "/v1/chat/completions", "collector-episode-21")
        ]
        assert current_rollout_id() is None
    finally:
        controls.release.set()
        await run
