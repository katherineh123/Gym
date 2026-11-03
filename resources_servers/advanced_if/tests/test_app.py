# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.core import NeMoGymResponse
from nemo_gym.verification import JudgeError, exercise_verifier_fixture
from resources_servers.advanced_if import app
from resources_servers.advanced_if.app import (
    VERIFIER_FIXTURE,
    AdvancedIfResourcesServer,
    AdvancedIfResourcesServerConfig,
    AdvancedIfRunRequest,
    AdvancedIfVerifyRequest,
    conversation_messages,
    parse_judge_reply,
    render_judge_prompt,
)
from resources_servers.advanced_if.task_data import TaskData
from resources_servers.advanced_if.verifier_fixture import create_fixture_server, judge_reply


SERVER_DIR = Path(__file__).resolve().parents[1]
PROMPTS = {
    "if_judge_prompt": "H:{full_conversation}|U:{user_prompt_last_turn}|R:{response_text}|B:{rubrics_text}|{{}}",
    "system_steer_judge_prompt": "F:{few_shot_examples}|S:{system_prompt}|U:{user_prompt_last_turn}|R:{response_text}|"
    "B:{rubrics_text}",
    "system_steer_few_shot_examples": 'example {"SATISFIED_ALL_REQUIREMENTS": "Yes"} {not_a_field}',
}
HISTORY = [
    {"role": "user", "content": "Plan a trip."},
    {"role": "assistant", "content": "Day 1: museums."},
    {"role": "user", "content": "Make it cheaper."},
    {"role": "assistant", "content": "Day 1: parks."},
    {"role": "user", "content": "Now add a day."},
]


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


def make_request(text: str = "Day 1: parks. Day 2: beach.", *, category="carried_context", rubrics=None, **response):
    return AdvancedIfVerifyRequest(
        responses_create_params={"input": HISTORY},
        response=make_response(text, **response),
        category=category,
        rubrics=rubrics or ["Is the plan cheaper?", "Does it add a day?"],
    )


def verdict(*answers: str, satisfied: str) -> str:
    checks = {f"question_{index}": answer for index, answer in enumerate(answers, 1)}
    return json.dumps({"rubrics_check": checks, "SATISFIED_ALL_REQUIREMENTS": satisfied})


def judge_with(server: AdvancedIfResourcesServer, *contents) -> AsyncMock:
    server.server_client.post = AsyncMock(side_effect=[judge_reply(content) for content in contents])
    return server.server_client.post


async def test_verifier_fixture() -> None:
    await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )


def test_example_rows_match_the_request_and_task_schemas() -> None:
    lines = (SERVER_DIR / "data" / "example.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        row = json.loads(line)
        AdvancedIfRunRequest.model_validate(row)
        TaskData.model_validate(row)
    assert {json.loads(line)["category"] for line in lines} == set(app.CATEGORIES)


def test_render_if_prompt_numbers_turns_and_leaves_out_the_final_user_turn() -> None:
    prompt = render_judge_prompt(
        PROMPTS,
        category="carried_context",
        messages=HISTORY,
        response_text="Day 2: beach.",
        rubrics=["Cheaper?", "Café?"],
    )
    history = "user [1]: Plan a trip.\nassistant [1]: Day 1: museums.\nuser [2]: Make it cheaper.\nassistant [2]: Day 1: parks."
    rubrics = json.dumps(["Cheaper?", "Café?"], indent=4)
    assert prompt == f"H:{history}|U:Now add a day.|R:Day 2: beach.|B:{rubrics}|{{}}"
    assert "\\u00e9" in prompt  # json.dumps escapes non-ASCII, as the official scorer does

    single_turn = render_judge_prompt(
        PROMPTS, category="complex_if", messages=HISTORY[:1], response_text="x", rubrics=["a"]
    )
    assert single_turn.startswith("H:|U:Plan a trip.|")


def test_render_system_steer_prompt_shows_only_system_prompt_and_last_user_turn() -> None:
    messages = [{"role": "system", "content": "Answer in French."}, *HISTORY]
    prompt = render_judge_prompt(
        PROMPTS, category="system_steerability", messages=messages, response_text="Bonjour", rubrics=["French?"]
    )
    few_shot = PROMPTS["system_steer_few_shot_examples"]
    assert (
        prompt == f"F:{few_shot}|S:Answer in French.|U:Now add a day.|R:Bonjour|B:{json.dumps(['French?'], indent=4)}"
    )
    assert "museums" not in prompt

    no_system = render_judge_prompt(
        PROMPTS, category="system_steerability", messages=HISTORY, response_text="x", rubrics=["a"]
    )
    assert "|S:|" in no_system


def test_conversation_messages_reads_strings_and_content_parts() -> None:
    assert conversation_messages("hi") == [{"role": "user", "content": "hi"}]
    parts = [{"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]}]
    assert conversation_messages(parts) == [{"role": "user", "content": "ab"}]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"SATISFIED_ALL_REQUIREMENTS": "YES"}', {"SATISFIED_ALL_REQUIREMENTS": "YES"}),
        ('```json\n{"SATISFIED_ALL_REQUIREMENTS": "NO"}\n```', {"SATISFIED_ALL_REQUIREMENTS": "NO"}),
        ("not json", None),
        ("[1, 2]", None),
        ('{"rubrics_check": {}}', None),
        ('{"SATISFIED_ALL_REQUIREMENTS": true}', None),
        ("", None),
        (None, None),
    ],
)
def test_parse_judge_reply(text, expected) -> None:
    assert parse_judge_reply(text) == expected


async def test_verify_sends_the_official_judge_request_and_scores_its_overall_field() -> None:
    server = create_fixture_server()
    post = judge_with(server, verdict("Yes", "Yes", satisfied="YES"))

    result = await server.verify(make_request())

    assert result.reward == 1.0
    assert result.satisfied_all_requirements == "YES"
    assert result.judge_attempts == 1
    sent = post.await_args.kwargs
    assert sent["server_name"] == "advanced_if_judge_model"
    assert sent["url_path"] == "/v1/chat/completions"
    body = sent["json"].model_dump(exclude_unset=True)
    assert body["max_completion_tokens"] == 32768
    assert body["response_format"] == {"type": "json_object"}
    assert "temperature" not in body
    assert len(body["messages"]) == 1 and "Day 1: museums." in body["messages"][0]["content"]


async def test_verify_follows_the_overall_field_not_the_rubric_answers() -> None:
    server = create_fixture_server()
    judge_with(server, verdict("Yes", "Yes", satisfied="NO"))
    assert (await server.verify(make_request())).reward == 0.0

    server = create_fixture_server()
    judge_with(server, verdict("The tone is pleasant.", "No emojis are used.", satisfied=" Yes "))
    assert (await server.verify(make_request())).reward == 1.0


async def test_verify_asks_again_when_the_judge_reply_is_not_a_verdict() -> None:
    server = create_fixture_server()
    judge_with(server, "I think it passes.", verdict("Yes", "No", satisfied="NO"))

    result = await server.verify(make_request())

    assert result.judge_attempts == 2
    assert result.judge_parse_failed is False
    assert result.satisfied_all_requirements == "NO"


async def test_verify_scores_zero_after_every_attempt_is_unparseable() -> None:
    server = create_fixture_server()
    post = judge_with(server, *["not json"] * server.config.judge_max_attempts)

    result = await server.verify(make_request())

    assert result.reward == 0.0
    assert result.judge_parse_failed is True
    assert result.judge_attempts == server.config.judge_max_attempts == post.await_count


async def test_verify_fails_a_truncated_or_empty_reply_without_the_judge() -> None:
    server = create_fixture_server()
    post = judge_with(server)

    truncated = await server.verify(make_request("Day 1", incomplete_reason="max_output_tokens"))
    filtered = await server.verify(make_request("", reasoning="declining", incomplete_reason="content_filter"))
    empty = await server.verify(make_request("", reasoning="thinking only"))
    unclosed = await server.verify(make_request("<think>still thinking"))

    assert [r.reward for r in (truncated, filtered, empty, unclosed)] == [0.0, 0.0, 0.0, 0.0]
    assert truncated.policy_incomplete_reason == "max_output_tokens" and truncated.extracted_answer == "Day 1"
    assert filtered.policy_incomplete_reason == "content_filter" and empty.policy_incomplete_reason is None
    assert filtered.extracted_answer == empty.extracted_answer == unclosed.extracted_answer == ""
    post.assert_not_awaited()


async def test_verify_sends_the_judge_the_answer_without_reasoning() -> None:
    server = create_fixture_server()
    post = judge_with(server, verdict("Yes", "Yes", satisfied="YES"))

    result = await server.verify(make_request("<think>private plan</think>Day 2: beach.", reasoning="hidden"))

    assert result.extracted_answer == "Day 2: beach."
    prompt = post.await_args.kwargs["json"].messages[0]["content"]
    assert "private plan" not in prompt and "hidden" not in prompt


async def test_verify_lets_a_judge_failure_propagate(monkeypatch) -> None:
    server = create_fixture_server()
    monkeypatch.setattr(app, "call_judge", AsyncMock(side_effect=JudgeError("judge down")))
    with pytest.raises(JudgeError):
        await server.verify(make_request())


def test_missing_judge_prompts_point_to_prepare(tmp_path) -> None:
    server = create_fixture_server()
    config = AdvancedIfResourcesServerConfig.model_validate(
        {**server.config.model_dump(), "judge_prompts_fpath": str(tmp_path / "missing.json")}
    )
    with pytest.raises(FileNotFoundError, match="gym eval prepare --benchmark advanced_if"):
        AdvancedIfResourcesServer(config=config, server_client=server.server_client)


def test_metrics_report_category_scores_and_their_mean() -> None:
    def rollout(category, reward, **flags):
        return {
            "category": category,
            "reward": reward,
            "extracted_answer": "" if flags.get("reason") == "content_filter" or flags.get("empty") else "answer",
            "policy_incomplete_reason": flags.get("reason"),
            "judge_parse_failed": flags.get("parse_failed", False),
        }

    tasks = [
        [rollout("complex_if", 1.0)],
        [rollout("complex_if", 1.0)],
        [rollout("carried_context", 0.0, parse_failed=True)],
        [rollout("carried_context", 1.0)],
        [rollout("system_steerability", 0.0, reason="max_output_tokens")],
        [rollout("system_steerability", 0.0, reason="content_filter")],
    ]
    server = create_fixture_server()
    metrics = server.compute_metrics(tasks)

    assert metrics["complex_if/pass@1[avg-of-1]/reward"] == 100.0
    assert metrics["carried_context/pass@1[avg-of-1]/reward"] == 50.0
    assert metrics["system_steerability/pass@1[avg-of-1]/reward"] == 0.0
    assert metrics["category_mean/pass@1[avg-of-1]/reward"] == 50.0
    assert metrics["pass@1[avg-of-1]/reward"] == pytest.approx(50.0)
    assert metrics["judge_parse_failed_rate"] == pytest.approx(1 / 6)
    assert metrics["policy_max_output_tokens_rate"] == pytest.approx(1 / 6)
    assert metrics["policy_content_filter_rate"] == pytest.approx(1 / 6)
    assert metrics["empty_answer_rate"] == pytest.approx(1 / 6)

    key = server.get_key_metrics(metrics)
    assert key["category_mean/pass@1[avg-of-1]/reward"] == 50.0
    assert key["system_steerability/pass@1[avg-of-1]/reward"] == 0.0
    assert {"pass@1[avg-of-1]/reward", "judge_parse_failed_rate", "policy_content_filter_rate"} <= set(key)
