# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from responses_api_agents.opencode_if_agent.tests.test_builder import constraint, row


def test_generation_cli_emits_cartesian_variants_and_refuses_overwrite(tmp_path):
    source, specs, target = (tmp_path / name for name in ("tasks.jsonl", "specs.json", "variants.jsonl"))
    source.write_text(json.dumps(row()) + "\n")
    specs.write_text(json.dumps([{}, {"instructions": [constraint()]}]))
    cmd = [
        sys.executable,
        "-m",
        "nemo_gym.task_variants.generate",
        "--input",
        str(source),
        "--specs",
        str(specs),
        "--output",
        str(target),
        "--agent-name",
        "scale_if",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["instance_id"] == rows[1]["instance_id"]
    assert rows[0]["task_variant"]["variant_id"] != rows[1]["task_variant"]["variant_id"]
    assert rows[1]["agent_ref"]["name"] == "scale_if"
    assert "End with FINISH." in rows[1]["responses_create_params"]["input"][0]["content"]
    assert subprocess.run(cmd, capture_output=True).returncode != 0


def legacy_item():
    return {
        "instance_id": "x",
        "type": "fresh",
        "tool_names": {"BASH_TOOL_NAME": "shell"},
        "constraints": [
            {
                "id": "x#c1",
                "reference_instruction": "Reference, not displayed.",
                "surface": "system_prompt",
                "position": "mid",
                "verifier_parameter": {
                    "template": "turn_output",
                    "trigger": {"tool": "BASH_TOOL_NAME"},
                    "obligation": {"match": "length_bound", "value": {"n": 80, "unit": "chars", "dir": "max"}},
                    "no_answer": "ungradable",
                },
            }
        ],
        "row_metadata": {"system_prompt_template_text": "Before.\nKeep shell narration under 80 characters.\nAfter."},
        "materialization": {
            "surfaces": {
                "system_prompt": {
                    "checks": {
                        "faithfulness": {
                            "per_constraint": [
                                {"id": "c1", "sentences": ["Keep shell narration under 80 characters."]},
                            ]
                        }
                    }
                }
            }
        },
    }


def test_migration_uses_displayed_wording_preserves_placement_and_adds_taxonomy():
    from nemo_gym.task_variants.migrate_charlie import migrate_item

    old = legacy_item()
    before = deepcopy(old)
    result = migrate_item(old)
    item = result["instructions"][0]
    assert old == before
    assert item["instruction_text"] == "Keep shell narration under 80 characters."
    assert item["placement"] == {"surface": "system_prompt", "position": "mid"}
    assert item["taxonomy"] == ["IF-LENGTH"]
    assert "shell" in item["rubric"]
    assert "trigger" not in item and "obligation" not in item
    assert result["native_runnable"] is False


def test_migration_does_not_guess_missing_displayed_wording():
    from nemo_gym.task_variants.migrate_charlie import migrate_item

    old = legacy_item()
    old["materialization"] = {}
    with pytest.raises(ValueError, match="displayed"):
        migrate_item(old)


def test_continuations_keep_their_phase_instead_of_becoming_fresh():
    from nemo_gym.task_variants.migrate_charlie import migrate_item

    old = legacy_item()
    old["type"] = "interject"
    result = migrate_item(old)
    assert result["source_phase"] == "interject"
    assert "continuation" in result["native_blockers"]


def test_migration_prefers_literal_placement_over_checker_paraphrase():
    from nemo_gym.task_variants.migrate_charlie import migrate_item

    old = legacy_item()
    surface = old["materialization"]["surfaces"]["system_prompt"]
    surface["placements"] = {"c1": ["Keep shell narration under 80 characters."]}
    surface["checks"]["faithfulness"]["per_constraint"][0]["sentences"] = ["Use fewer than eighty characters."]
    assert migrate_item(old)["instructions"][0]["instruction_text"] == "Keep shell narration under 80 characters."
    old["row_metadata"]["system_prompt_template_text"] = "No constraint here."
    with pytest.raises(ValueError, match="displayed"):
        migrate_item(old)
