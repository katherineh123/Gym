# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a reproducible JSONL from immutable native task rows and explicit variant specs."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from nemo_gym.task_variants.builder import build_variant
from nemo_gym.task_variants.schema import VariantSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True, help="JSON list of fully resolved VariantSpec objects")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-name", required=True, help="Configured Gym agent instance, not a harness class name")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new snapshot path")
    specs = [VariantSpec.model_validate(value) for value in json.loads(args.specs.read_text())]
    if not specs:
        parser.error("at least one variant specification is required")
    seen: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False, encoding="utf-8") as stream:
            pending = Path(stream.name)
            with args.input.open() as source:
                for line in source:
                    if not line.strip():
                        continue
                    base = json.loads(line)
                    for spec in specs:
                        row = build_variant(base, spec)
                        variant_id = row["task_variant"]["variant_id"]
                        if variant_id in seen:
                            raise ValueError(f"duplicate generated variant: {variant_id}")
                        seen.add(variant_id)
                        row["agent_ref"] = {"type": "responses_api_agents", "name": args.agent_name}
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not seen:
            raise ValueError("input contains no tasks")
        # Atomic no-clobber publication: a failed validation never leaves a partial dataset.
        os.link(pending, args.output)
        print(json.dumps({"rows": len(seen), "output": str(args.output)}))
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
