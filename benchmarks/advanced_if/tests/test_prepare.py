# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.advanced_if import prepare


JUDGE_SOURCE = '''
import inspect

JUDGE_PROMPT: str = inspect.cleandoc(
    """
    Judge {response_text}
      against {rubrics_text}
    """
)

STEER_FEW_SHOT_EXAMPLES: str = """
Example {\\"SATISFIED_ALL_REQUIREMENTS\\": \\"Yes\\"}
"""

SYSTEM_STEER_JUDGE_PROMPT = inspect.cleandoc("""Steer {few_shot_examples}
    {system_prompt}""")
'''


def csv_row(benchmark_name: str, messages: list, rubrics: list) -> dict:
    return {
        "conversation_history": json.dumps(messages),
        "benchmark_name": benchmark_name,
        "prompt_metadata": json.dumps({"rubrics": json.dumps(rubrics)}),
    }


def test_build_row_decodes_the_conversation_and_double_encoded_rubrics() -> None:
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    row = prepare.build_row(7, csv_row("system_steerability_v2", messages, ["Is it brief?"]))
    assert row == {
        "task_id": 7,
        "category": "system_steerability",
        "benchmark_name": "system_steerability_v2",
        "rubrics": ["Is it brief?"],
        "responses_create_params": {"input": messages},
    }


@pytest.mark.parametrize(
    ("messages", "rubrics", "error"),
    [
        (
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            ["a"],
            "end with a user turn",
        ),
        ([{"role": "user", "content": "Hi"}], [], "rubric strings"),
    ],
)
def test_build_row_rejects_rows_the_judge_cannot_score(messages, rubrics, error) -> None:
    with pytest.raises(ValueError, match=error):
        prepare.build_row(0, csv_row("complex_if_single_turn_v5", messages, rubrics))


def test_extract_judge_prompts_reads_runtime_strings_without_running_the_file() -> None:
    prompts = prepare.extract_judge_prompts(JUDGE_SOURCE)
    assert prompts == {
        "if_judge_prompt": "Judge {response_text}\n  against {rubrics_text}",
        "system_steer_few_shot_examples": '\nExample {"SATISFIED_ALL_REQUIREMENTS": "Yes"}\n',
        "system_steer_judge_prompt": "Steer {few_shot_examples}\n{system_prompt}",
    }


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ('JUDGE_PROMPT = "a"\nSTEER_FEW_SHOT_EXAMPLES = "b"\n', "missing"),
        (JUDGE_SOURCE + "JUDGE_PROMPT = build()\n", "not a string literal"),
    ],
)
def test_extract_judge_prompts_rejects_unexpected_sources(source, error) -> None:
    with pytest.raises(ValueError, match=error):
        prepare.extract_judge_prompts(source)


def test_prepare_pins_both_sources_and_writes_rows_and_prompts(tmp_path, monkeypatch) -> None:
    rows = [
        csv_row("complex_if_single_turn_v5", [{"role": "user", "content": "Write a poem."}], ["Is it a poem?"]),
        csv_row(
            "carried_context_multi_turn_eval_v5",
            [
                {"role": "user", "content": "Draft an email."},
                {"role": "assistant", "content": "Dear team, ..."},
                {"role": "user", "content": "Shorter, and say café."},
            ],
            ["Is it shorter?", "Does it say café?"],
        ),
    ]
    source = JUDGE_SOURCE.encode()
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", len(rows))
    monkeypatch.setattr(prepare, "JUDGE_SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    output = tmp_path / "advanced_if_benchmark.jsonl"

    with (
        patch.object(prepare, "load_dataset", return_value=rows) as load_dataset,
        patch.object(prepare.requests, "get", return_value=MagicMock(content=source)) as get,
    ):
        assert prepare.prepare(output) == output

    load_dataset.assert_called_once_with("facebook/AdvancedIF", split="train", revision=prepare.HF_REVISION)
    get.assert_called_once_with(prepare.JUDGE_SOURCE_URL, timeout=60)
    assert prepare.JUDGE_SOURCE_COMMIT in prepare.JUDGE_SOURCE_URL
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["task_id"] for row in written] == [0, 1]
    assert [row["category"] for row in written] == ["complex_if", "carried_context"]
    assert written[1]["rubrics"] == ["Is it shorter?", "Does it say café?"]
    assert "café" in output.read_text(encoding="utf-8")
    prompts = json.loads((tmp_path / prepare.JUDGE_PROMPTS_FPATH.name).read_text(encoding="utf-8"))
    assert prompts["source"] == prepare.JUDGE_SOURCE_URL and prompts["license"] == "CC BY-NC 4.0"
    assert prompts["if_judge_prompt"].startswith("Judge {response_text}")


def test_prepare_refuses_a_changed_judge_source(tmp_path) -> None:
    with patch.object(prepare.requests, "get", return_value=MagicMock(content=b"tampered")):
        with pytest.raises(ValueError, match="expected"):
            prepare.prepare_judge_prompts(tmp_path / "prompts.json")


def test_prepare_checks_the_row_count(tmp_path) -> None:
    rows = [csv_row("complex_if_single_turn_v5", [{"role": "user", "content": "Hi"}], ["a"])]
    with patch.object(prepare, "load_dataset", return_value=rows):
        with pytest.raises(ValueError, match="expected 1645 rows"):
            prepare.prepare(tmp_path / "out.jsonl")
