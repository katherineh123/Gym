# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Migrate historical IF metadata without changing its displayed prompts or pretending it is a native rollout.

Historical early/mid/late placements and continuation phases are preserved as evidence.
They are not automatically relabeled as start/end fresh OpenCode tasks. Porting tool
interfaces, supplying OCI images, and reconstructing continuations are separate work.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from nemo_gym.task_variants.builder import digest


def _displayed_text(item: dict[str, Any], constraint: dict[str, Any]) -> tuple[str, str]:
    short_id = constraint["id"].rsplit("#", 1)[-1]
    surface = item.get("materialization", {}).get("surfaces", {}).get(constraint["surface"], {})
    checks = surface.get("checks", {}).get("faithfulness", {}).get("per_constraint", [])
    sentences = next((entry.get("sentences", []) for entry in checks if entry["id"] == short_id), [])
    if sentences:
        return "\n".join(sentences), "materialization.faithfulness.sentences"
    # Some historical checks lack per-constraint sentences; preserve the exact saved
    # placement block (which may cover several constraints), never invent a paraphrase.
    block = surface.get("placements", {}).get(short_id)
    if isinstance(block, list) and block:
        return "\n".join(block), "materialization.placements.shared_block"
    raise ValueError(f"missing exact displayed wording for {constraint['id']}")


def _rubric(item: dict[str, Any], constraint: dict[str, Any]) -> str:
    parameter = constraint["verifier_parameter"]
    trigger = parameter["trigger"]
    if "tool" in trigger:
        name = item["tool_names"].get(trigger["tool"], trigger["tool"])
        scope = (
            "each assistant turn with any tool call" if name == "ANY_TOOL" else f"each assistant turn calling `{name}`"
        )
    else:
        scope = {
            "final": "the final assistant response",
            "first_turn": "the first assistant turn",
            "any_turn": "every assistant turn",
        }[trigger["position"]]
    reference = constraint["reference_instruction"]
    no_answer = parameter.get("no_answer", "fail")
    absence = (
        "If an affected turn has no assistant text, evidence is insufficient to grade this constraint."
        if no_answer == "ungradable"
        else "An affected turn with no assistant text fails this constraint."
    )
    return (
        f"On {scope}, evaluate this requirement: {reference} Judge the exact displayed instruction, "
        "not unrelated instructions in a shared placement block. "
        + absence
        + " If the applicability condition never occurs, return not_applicable. "
        "For continuation evidence, judge only generated turns after the recorded prefix, not the frozen prefix."
    )


def migrate_item(item: dict[str, Any]) -> dict[str, Any]:
    """Replace the matcher DSL with displayed instruction/placement/rubric/taxonomy records."""
    instructions = []
    for constraint in item["constraints"]:
        text, wording_source = _displayed_text(item, constraint)
        matcher = constraint["verifier_parameter"]["obligation"]["match"]
        if matcher not in {
            "length_bound",
            "language",
            "forbidden",
            "regex",
            "prefix",
            "fenced",
            "exact",
            "json_schema",
        }:
            raise ValueError(f"unreviewed historical matcher taxonomy: {matcher}")
        taxonomy = "IF-LENGTH" if matcher == "length_bound" else "IF-LANG" if matcher == "language" else "IF-FORMAT"
        instructions.append(
            {
                "id": constraint["id"],
                "instruction_text": text,
                "placement": {
                    "surface": {"problem_statement": "user_prompt"}.get(constraint["surface"], constraint["surface"]),
                    "position": constraint["position"],
                },
                "rubric": _rubric(item, constraint),
                "taxonomy": [taxonomy],
                "provenance": {"wording_source": wording_source, "source_constraint_sha256": digest(constraint)},
            }
        )
    blockers = ["legacy_tool_interface", "OCI_image_mapping_required", "legacy_prompt_positions"]
    if item["type"] != "fresh":
        blockers.append("continuation")
    return {
        "schema_version": "rubric-migration-1",
        "base_task_id": item["instance_id"],
        "source_phase": item["type"],
        "source_sha256": digest(item),
        "instructions": instructions,
        "legacy_rendered_prompts": deepcopy(item.get("row_metadata", {})),
        "legacy_tool_names": deepcopy(item["tool_names"]),
        "prefix": deepcopy(item.get("prefix")),
        "native_runnable": False,
        "native_blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    params = json.loads(args.params.read_text())
    migrated = [migrate_item(item) for item in params["items"]]
    by_id = {item["base_task_id"]: item for item in migrated}
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    if len(by_id) != len(migrated) or {row["instance_id"] for row in rows} != set(by_id) or len(rows) != len(migrated):
        raise ValueError("parameter and row task identities must match one-to-one")
    for row in rows:
        item = by_id[row["instance_id"]]
        row["if_migration"] = item
        metadata = row["responses_create_params"].setdefault("metadata", {})
        metadata.pop("sdg_item", None)
        metadata["if_instructions"] = json.dumps(item["instructions"], ensure_ascii=False)
        # Do not allow the old deterministic wrapper to run with missing DSL metadata.
        row.pop("agent_ref", None)
    report = {
        "params_source": str(args.params),
        "params_sha256": digest(params),
        "rows_source": str(args.rows),
        "rows": len(rows),
        "constraints": sum(len(item["instructions"]) for item in migrated),
        "native_runnable": 0,
        "note": "Metadata migration only. Native variants must be generated from OCI-backed base rows.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "params.rubric.json").write_text(
        json.dumps({"items": migrated, "provenance": report}, indent=2) + "\n"
    )
    (args.output_dir / "rows.rubric.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
