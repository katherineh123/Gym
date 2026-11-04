# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.core import NeMoGymResponse
from nemo_gym.verification import JudgeError, exercise_verifier_fixture, judge_failsafe
from resources_servers.complex_constraints import app
from resources_servers.complex_constraints.app import (
    INSTRUCTION_TYPES,
    VERIFIER_FIXTURE,
    ComplexConstraintsResourcesServer,
    ComplexConstraintsResourcesServerConfig,
    ComplexConstraintsRunRequest,
    ComplexConstraintsVerifyRequest,
    criterion_passes,
    parse_criterion_verdict,
    render_judge_prompt,
)
from resources_servers.complex_constraints.task_data import TaskData
from resources_servers.complex_constraints.verifier_fixture import (
    create_fixture_server,
    judge_reply,
    scores_by_criterion,
)


SERVER_DIR = Path(__file__).resolve().parents[1]
PROMPT = "Plan the bake sale: three tables, no nuts, and end with a thank-you line."
RUBRICS = [
    "The plan uses exactly three tables.",
    "No item on the plan contains nuts.",
    "The plan ends with a thank-you line.",
]
ANSWER = "Table 1: cookies. Table 2: lemon bars. Table 3: muffins. Thank you, volunteers!"


def make_response(text: str = "", *, reasoning: str = "", incomplete_reason: str = "") -> NeMoGymResponse:
    output = []
    if reasoning:
        output.append({"id": "r", "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    if text:
        output.append(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    payload = {
        "id": "policy",
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }
    if incomplete_reason:
        payload.update(status="incomplete", incomplete_details={"reason": incomplete_reason})
    return NeMoGymResponse.model_validate(payload)


def make_request(text: str = ANSWER, **response) -> ComplexConstraintsVerifyRequest:
    return ComplexConstraintsVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": PROMPT}]},
        response=make_response(text, **response),
        task_id="CIF-T01",
        instruction_type="Negative",
        rubrics=RUBRICS,
    )


def judge_with_scores(server: ComplexConstraintsResourcesServer, *scores: str) -> AsyncMock:
    server.server_client.post = AsyncMock(side_effect=scores_by_criterion(RUBRICS, scores))
    return server.server_client.post


def sent_prompts(post: AsyncMock) -> list:
    return [call.kwargs["json"].messages[0]["content"] for call in post.await_args_list]


async def test_verifier_fixture() -> None:
    await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )


def test_example_rows_match_the_request_and_task_schemas() -> None:
    lines = (SERVER_DIR / "data" / "example.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        row = json.loads(line)
        ComplexConstraintsRunRequest.model_validate(row)
        TaskData.model_validate(row)
        assert [message["role"] for message in row["responses_create_params"]["input"]] == ["user"]
    assert {json.loads(line)["instruction_type"] for line in lines} == set(INSTRUCTION_TYPES)


def test_render_judge_prompt_substitutes_the_reply_then_the_criterion() -> None:
    template = "R={{ rubric.response_to_grade }} | C={{ rubric.criterion }} | {literal braces}"
    assert (
        render_judge_prompt(template, response_text="hi", criterion="be brief")
        == "R=hi | C=be brief | {literal braces}"
    )
    # As in the official scorer, a reply that contains the criterion placeholder gets the criterion substituted too.
    quirk = render_judge_prompt(template, response_text="{{ rubric.criterion }}", criterion="x")
    assert quirk == "R=x | C=x | {literal braces}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"rationale": "ok", "score": "1"}', (Decimal("1"), "ok")),
        ('{"score": 0, "rationale": "no"}', (Decimal("0"), "no")),
        ('{"score": " 0.5 ", "rationale": ""}', (Decimal("0.5"), "")),
        ('```json\n{"rationale": "fenced", "score": "1"}\n```', (Decimal("1"), "fenced")),
        ('{"score": "1"}', None),
        ('{"score": "1", "rationale": 3}', None),
        ('{"score": true, "rationale": "r"}', None),
        ('{"score": ["1"], "rationale": "r"}', None),
        ('{"score": "yes", "rationale": "r"}', None),
        ('{"score": "NaN", "rationale": "r"}', None),
        ('{"score": "Infinity", "rationale": "r"}', None),
        ('["1"]', None),
        ("The response satisfies it.", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_criterion_verdict(text, expected) -> None:
    assert parse_criterion_verdict(text) == expected


@pytest.mark.parametrize(
    ("score", "passes"),
    [("1", True), ("2", True), ("0.99995", True), ("0.99994", False), ("0", False), ("-1", False)],
)
def test_criterion_passes_after_rounding_half_up_to_four_places(score, passes) -> None:
    assert criterion_passes(Decimal(score)) is passes


async def test_verify_sends_one_judge_call_per_criterion_with_the_official_request() -> None:
    server = create_fixture_server()
    post = judge_with_scores(server, "1", "1", "1")

    result = await server.verify(make_request())

    assert result.reward == 1.0 and result.mean_criteria == 1.0 and result.criteria_passed == 3
    assert post.await_count == len(RUBRICS)
    prompts = sent_prompts(post)
    assert sorted(prompt.rsplit("Criterion:\n", 1)[1].split("\n")[0] for prompt in prompts) == sorted(RUBRICS)
    # The judge sees the reply and one criterion, never the task prompt.
    assert all(ANSWER in prompt and PROMPT not in prompt for prompt in prompts)
    sent = post.await_args.kwargs
    assert sent["server_name"] == "complex_constraints_judge_model"
    assert sent["url_path"] == "/v1/chat/completions"
    body = sent["json"].model_dump(exclude_unset=True)
    judge_file = json.loads((SERVER_DIR / "tests" / "judge_prompt.json").read_text(encoding="utf-8"))
    assert body["response_format"] == {"type": "json_schema", "json_schema": judge_file["response_schema"]}
    assert set(body) == {"messages", "response_format"}
    assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"


async def test_verify_passes_a_task_only_when_every_criterion_passes() -> None:
    server = create_fixture_server()
    judge_with_scores(server, "1", "0", "1")

    result = await server.verify(make_request())

    assert result.reward == 0.0
    assert result.mean_criteria == pytest.approx(2 / 3)
    assert result.criteria_passed == 2
    assert [grade.passed for grade in result.criterion_grades] == [True, False, True]
    assert [grade.score for grade in result.criterion_grades] == ["1", "0", "1"]


async def test_verify_asks_again_when_a_judge_reply_does_not_parse() -> None:
    server = create_fixture_server()
    valid = scores_by_criterion(RUBRICS, ["1", "1", "1"])
    replies = {"first": True}

    async def post(**kwargs):
        if RUBRICS[1] in kwargs["json"].messages[0]["content"] and replies.pop("first", False):
            return judge_reply("I think it passes.")
        return await valid(**kwargs)

    server.server_client.post = AsyncMock(side_effect=post)
    result = await server.verify(make_request())

    assert result.reward == 1.0
    assert [grade.judge_attempts for grade in result.criterion_grades] == [1, 2, 1]
    assert server.server_client.post.await_count == 4


async def test_a_criterion_that_never_parses_masks_the_row_through_the_judge_failsafe() -> None:
    server = create_fixture_server()
    valid = scores_by_criterion(RUBRICS, ["1", "1", "1"])

    async def post(**kwargs):
        if RUBRICS[2] in kwargs["json"].messages[0]["content"]:
            return judge_reply("not json")
        return await valid(**kwargs)

    server.server_client.post = AsyncMock(side_effect=post)
    with pytest.raises(JudgeError, match=r"criteria \[2\]"):
        await server.verify(make_request())
    assert server.server_client.post.await_count == 2 + server.config.judge_max_attempts

    reply = await judge_failsafe(server.verify)(body=make_request())
    row = json.loads(reply.body)
    assert row["mask_sample"] is True and row["failure_kind"] == "judge_failed"


async def test_verify_scores_zero_without_the_judge_when_there_is_no_visible_reply() -> None:
    server = create_fixture_server()
    post = judge_with_scores(server)

    reasoning_only = await server.verify(make_request("", reasoning="thinking only"))
    filtered = await server.verify(make_request("", incomplete_reason="content_filter"))
    unclosed = await server.verify(make_request("<think>still thinking"))

    for result in (reasoning_only, filtered, unclosed):
        assert (result.reward, result.mean_criteria, result.criteria_passed) == (0.0, 0.0, 0)
        assert result.extracted_answer == "" and result.criterion_grades == []
    assert filtered.policy_incomplete_reason == "content_filter"
    post.assert_not_awaited()


async def test_verify_grades_a_reply_cut_off_by_the_output_cap_as_written() -> None:
    server = create_fixture_server()
    post = judge_with_scores(server, "1", "1", "0")

    result = await server.verify(make_request("Table 1: cookies.", incomplete_reason="max_output_tokens"))

    assert result.policy_incomplete_reason == "max_output_tokens"
    assert result.reward == 0.0 and result.criteria_passed == 2
    assert post.await_count == len(RUBRICS)


async def test_verify_sends_the_judge_the_answer_without_reasoning() -> None:
    server = create_fixture_server()
    post = judge_with_scores(server, "1", "1", "1")

    result = await server.verify(make_request(f"<think>private plan</think>{ANSWER}", reasoning="hidden"))

    assert result.extracted_answer == ANSWER
    assert not any("private plan" in prompt or "hidden" in prompt for prompt in sent_prompts(post))


async def test_verify_lets_a_judge_call_failure_propagate(monkeypatch) -> None:
    server = create_fixture_server()
    monkeypatch.setattr(app, "call_judge", AsyncMock(side_effect=JudgeError("judge down")))
    with pytest.raises(JudgeError, match="judge down"):
        await server.verify(make_request())


def test_missing_judge_prompt_points_to_prepare(tmp_path) -> None:
    server = create_fixture_server()
    config = ComplexConstraintsResourcesServerConfig.model_validate(
        {**server.config.model_dump(), "judge_prompt_fpath": str(tmp_path / "missing.json")}
    )
    with pytest.raises(FileNotFoundError, match="gym eval prepare --benchmark complex_constraints"):
        ComplexConstraintsResourcesServer(config=config, server_client=server.server_client)


def test_requests_need_at_least_one_rubric() -> None:
    with pytest.raises(ValueError, match="rubrics"):
        ComplexConstraintsRunRequest.model_validate(
            {"responses_create_params": {"input": [{"role": "user", "content": PROMPT}]}, "rubrics": []}
        )


def test_metrics_report_task_pass_mean_criteria_and_instruction_types() -> None:
    def rollout(instruction_type, reward, mean_criteria, *, answer="answer", reason=None, attempts=(1,)):
        return {
            "instruction_type": instruction_type,
            "reward": reward,
            "mean_criteria": mean_criteria,
            "extracted_answer": answer,
            "policy_incomplete_reason": reason,
            "criterion_grades": [{"judge_attempts": attempt} for attempt in attempts],
        }

    tasks = [
        [rollout("Negative", 1.0, 1.0), rollout("Negative", 0.0, 0.5, attempts=(1, 2))],
        [rollout("Multistep", 0.0, 0.25), rollout("Multistep", 0.0, 0.75)],
        [rollout("Implicit", 0.0, 0.0, answer="", reason="max_output_tokens", attempts=())] * 2,
        [rollout("Conditional", 1.0, 1.0)] * 2,
    ]
    server = create_fixture_server()
    metrics = server.compute_metrics(tasks)

    assert metrics["pass@1[avg-of-2]/reward"] == pytest.approx(37.5)
    assert metrics["pass@1[avg-of-2]/mean_criteria"] == pytest.approx(56.25)
    assert metrics["Negative/pass@1[avg-of-2]/reward"] == pytest.approx(50.0)
    assert metrics["Conditional/pass@1[avg-of-2]/reward"] == pytest.approx(100.0)
    assert metrics["empty_answer_rate"] == pytest.approx(0.25)
    assert metrics["policy_max_output_tokens_rate"] == pytest.approx(0.25)
    assert metrics["policy_content_filter_rate"] == 0.0
    assert metrics["judge_retry_rate"] == pytest.approx(1 / 7)

    key = server.get_key_metrics(metrics)
    assert key["pass@1[avg-of-2]/reward"] == pytest.approx(37.5)
    assert key["pass@1[avg-of-2]/mean_criteria"] == pytest.approx(56.25)
    assert key["Implicit/pass@1[avg-of-2]/reward"] == 0.0
    assert {"empty_answer_rate", "policy_max_output_tokens_rate", "policy_content_filter_rate"} <= set(key)
