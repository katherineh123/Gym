# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the AdvancedIF benchmark and its official judge prompts.

Data: https://huggingface.co/datasets/facebook/AdvancedIF (1,645 conversations, CC BY-NC 4.0).
Judge prompts: https://github.com/facebookresearch/AdvancedIF/blob/<JUDGE_SOURCE_COMMIT>/judge.py (CC BY-NC 4.0).

Both are downloaded at pinned revisions and written under `data/`, which is not committed. The judge prompts are read
from judge.py's syntax tree; the file is never imported or run.
"""

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import requests
from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "advanced_if_benchmark.jsonl"
JUDGE_PROMPTS_FPATH = DATA_DIR / "advanced_if_judge_prompts.json"

HF_DATASET = "facebook/AdvancedIF"
HF_REVISION = "e20cba9b94b59c027dfab00b29244e8bc42e4ab4"  # pragma: allowlist secret
EXPECTED_ROWS = 1645

JUDGE_SOURCE_COMMIT = "f9d30137c4139d4d9af260ae28108b5afae828c0"  # pragma: allowlist secret
JUDGE_SOURCE_URL = f"https://raw.githubusercontent.com/facebookresearch/AdvancedIF/{JUDGE_SOURCE_COMMIT}/judge.py"
JUDGE_SOURCE_SHA256 = "415164e9c3cb1e267321fa0561a8d61b81b3ac134a7d018764ae53a6e5a84955"  # pragma: allowlist secret

# The dataset's `benchmark_name` values and the paper's category names.
CATEGORY_BY_BENCHMARK_NAME = {
    "complex_if_single_turn_v5": "complex_if",
    "carried_context_multi_turn_eval_v5": "carried_context",
    "system_steerability_v2": "system_steerability",
}
# judge.py constant -> key in the prepared prompts file.
JUDGE_PROMPT_KEYS = {
    "JUDGE_PROMPT": "if_judge_prompt",
    "SYSTEM_STEER_JUDGE_PROMPT": "system_steer_judge_prompt",
    "STEER_FEW_SHOT_EXAMPLES": "system_steer_few_shot_examples",
}


def build_row(index: int, row: Dict[str, str]) -> Dict[str, Any]:
    """Turn one dataset row into a Gym task: the conversation is the policy input, the rubrics stay out of it."""
    messages: List[Dict[str, str]] = json.loads(row["conversation_history"])
    rubrics: List[str] = json.loads(json.loads(row["prompt_metadata"])["rubrics"])
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"row {index}: the conversation must end with a user turn")
    if not rubrics or not all(isinstance(rubric, str) for rubric in rubrics):
        raise ValueError(f"row {index}: expected a non-empty list of rubric strings")
    return {
        "task_id": index,
        "category": CATEGORY_BY_BENCHMARK_NAME[row["benchmark_name"]],
        "benchmark_name": row["benchmark_name"],
        "rubrics": rubrics,
        "responses_create_params": {"input": [{"role": m["role"], "content": m["content"]} for m in messages]},
    }


def extract_judge_prompts(source: str) -> Dict[str, str]:
    """Read the judge prompt constants from judge.py without running it, as the scorer sees them at runtime."""
    prompts: Dict[str, str] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in JUDGE_PROMPT_KEYS:
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                prompts[JUDGE_PROMPT_KEYS[target.id]] = value.value
            elif (
                isinstance(value, ast.Call)
                and ast.unparse(value.func) == "inspect.cleandoc"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
            ):
                prompts[JUDGE_PROMPT_KEYS[target.id]] = inspect.cleandoc(value.args[0].value)
            else:
                raise ValueError(f"{target.id} in judge.py is not a string literal")
    missing = set(JUDGE_PROMPT_KEYS.values()) - set(prompts)
    if missing:
        raise ValueError(f"judge.py is missing {sorted(missing)}")
    return prompts


def prepare_judge_prompts(output_fpath: Path = JUDGE_PROMPTS_FPATH) -> Path:
    response = requests.get(JUDGE_SOURCE_URL, timeout=60)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != JUDGE_SOURCE_SHA256:
        raise ValueError(f"{JUDGE_SOURCE_URL} has sha256 {digest}, expected {JUDGE_SOURCE_SHA256}")
    prompts = extract_judge_prompts(response.content.decode("utf-8"))
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text(
        json.dumps({"source": JUDGE_SOURCE_URL, "license": "CC BY-NC 4.0", **prompts}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_fpath


def prepare(output_fpath: Path = OUTPUT_FPATH) -> Path:
    dataset = load_dataset(HF_DATASET, split="train", revision=HF_REVISION)
    rows = [build_row(index, row) for index, row in enumerate(dataset)]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows from {HF_DATASET}@{HF_REVISION}, got {len(rows)}")
    prepare_judge_prompts(output_fpath.parent / JUDGE_PROMPTS_FPATH.name)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_fpath


if __name__ == "__main__":
    print(prepare())
