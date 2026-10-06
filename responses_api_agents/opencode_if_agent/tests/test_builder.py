# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import pytest


def build(row, spec):
    from nemo_gym.task_variants.builder import build_variant

    return build_variant(row, spec)


def row():
    return {
        "instance_id": "public-example-1",
        "problem_statement": "Fix the parser.",
        "image_url": "example/image:1",
        "workdir": "/app",
        "patch": "PRIVATE GOLDEN PATCH",
        "test_patch": "PRIVATE TESTS",
        "responses_create_params": {"input": [{"role": "user", "content": "Fix the parser. Do not edit tests."}]},
        "agent_ref": {"name": "old-agent", "type": "responses_api_agents"},
    }


def constraint(surface="user_prompt", **extra):
    return {
        "id": "finish",
        "taxonomy": ["IF-FORMAT"],
        "instruction_text": "End with FINISH.",
        "placement": {"surface": surface, "position": "end"},
        "rubric": "The final response must end with FINISH.",
        **extra,
    }


def test_baseline_keeps_native_grading_fields_and_does_not_mutate_source():
    original = row()
    before = deepcopy(original)
    result = build(original, {})
    assert original == before
    for key in ("instance_id", "patch", "test_patch", "image_url", "workdir"):
        assert result[key] == original[key]
    assert result["responses_create_params"] == original["responses_create_params"]
    assert result["task_variant"]["base_task_id"] == "public-example-1"
    assert result["task_variant"]["harness"] == "opencode"


def test_system_and_user_constraints_compose_without_exposing_rubrics():
    result = build(row(), {"instructions": [constraint("system_prompt"), constraint(id="format")]})
    messages = result["responses_create_params"]["input"]
    assert messages == [
        {"role": "system", "content": "End with FINISH."},
        {"role": "user", "content": "Fix the parser. Do not edit tests.\n\nEnd with FINISH."},
    ]
    assert "PRIVATE" not in str(messages)
    assert "The final response must" not in str(messages)


def test_aliases_resolve_only_explicit_placeholders_in_instruction_and_rubric():
    original = row()
    original["responses_create_params"]["input"][0]["content"] = "Do not rename bash in this issue."
    result = build(
        original,
        {
            "tool_names": {"bash": "shell"},
            "instructions": [
                constraint(instruction_text="Explain each ${tool:bash} call.", rubric="Check ${tool:bash}.")
            ],
        },
    )
    assert result["responses_create_params"]["input"][-1]["content"] == (
        "Do not rename bash in this issue.\n\nExplain each shell call."
    )
    assert result["task_variant"]["instructions"][0]["rubric"] == "Check shell."


def test_family_is_selected_before_injection_and_requires_explicit_normalized_context():
    result = build(
        row(),
        {
            "prompt_family": {"id": "minimal", "system": "Be concise.", "user": "Issue: ${issue}\n${rules}"},
            "task_rules": "Do not edit tests.",
            "instructions": [constraint("system_prompt")],
        },
    )
    assert result["responses_create_params"]["input"] == [
        {"role": "system", "content": "Be concise.\n\nEnd with FINISH."},
        {"role": "user", "content": "Issue: Fix the parser.\nDo not edit tests."},
    ]
    with pytest.raises(ValueError, match="task_rules"):
        build(row(), {"prompt_family": {"id": "minimal", "user": "${issue}"}})


def test_variant_identity_is_stable_and_distinct_from_base_task_identity():
    a = build(row(), {"seed": 4, "instructions": [constraint()]})
    b = build(row(), {"instructions": [constraint()], "seed": 4})
    c = build(row(), {"seed": 5, "instructions": [constraint()]})
    assert a["task_variant"]["variant_id"] == b["task_variant"]["variant_id"]
    assert a["task_variant"]["variant_id"] != c["task_variant"]["variant_id"]
    assert a["instance_id"] == c["instance_id"] == "public-example-1"


@pytest.mark.parametrize(
    "spec,match",
    [
        ({"harness": "openhands"}, "opencode"),
        ({"tool_names": {"bash": "read"}}, "collision"),
        ({"tool_names": {"bash": "a b"}}, "tool"),
        ({"instructions": [constraint(), constraint()]}, "unique"),
        ({"instructions": [constraint(taxonomy=["made-up"])]}, "taxonomy"),
        ({"instructions": [constraint(instruction_text="Use ${tool:not_a_tool}.")]}, "tool"),
    ],
)
def test_invalid_variants_fail_before_rollouts(spec, match):
    with pytest.raises(ValueError, match=match):
        build(row(), spec)


def test_existing_variant_or_conversation_cannot_be_silently_reinjected():
    with pytest.raises(ValueError, match="already"):
        build(build(row(), {}), {})
    source = row()
    source["responses_create_params"]["input"].append({"role": "assistant", "content": "old answer"})
    with pytest.raises(ValueError, match="fresh"):
        build(source, {})
