# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose variants without modifying native task/grading metadata."""

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from nemo_gym.task_variants.schema import TOOLS, VariantSpec


def content_text(content: object) -> str:
    """Read a text-only input, rejecting unsupported multimodal/history inputs."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str)
        for part in content
    ):
        return "\n".join(part["text"] for part in content)
    raise ValueError("fresh-task variants require text-only messages")


def resolve_tool_text(text: str, names: dict[str, str]) -> str:
    """Substitute explicit logical-tool placeholders, never arbitrary issue/source words."""

    def replace(match: re.Match[str]) -> str:
        tool = match[1]
        if tool not in TOOLS:
            raise ValueError(f"unknown tool placeholder: {tool}")
        return names.get(tool, tool)

    return re.sub(r"\$\{tool:([^}]+)\}", replace, text)


def _render(template: str, context: dict[str, str], names: dict[str, str]) -> str:
    template = resolve_tool_text(template, names)

    # One pass: task text containing template-looking strings is data, not a second template.
    def replace(match: re.Match[str]) -> str:
        if match[1] not in context:
            raise ValueError(f"unknown prompt placeholder: {match[1]}")
        if match[1] == "workdir" and not context["workdir"]:
            raise ValueError("${workdir} requires an explicit native base-row workdir")
        return context[match[1]]

    return re.sub(r"\$\{([^}]+)\}", replace, template)


def digest(value: object) -> str:
    """Stable content identity independent of mapping insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def actor_input_digest(messages: list[dict[str, Any]]) -> str:
    """Hash model-visible text, invariant to API message/content-block defaults."""
    return digest([{"role": item["role"], "content": content_text(item["content"])} for item in messages])


def build_variant(row: dict[str, Any], spec: dict[str, Any] | VariantSpec) -> dict[str, Any]:
    """Return a fresh runnable row and namespaced variant receipt; leave source untouched.

    Golden patches and native grader fields remain in the row, never in actor inputs.
    No instruction is synthesized at runtime. The full source digest makes changed base
    images, test contracts or prompts produce a different variant identity.
    """
    if "task_variant" in row:
        raise ValueError("row already contains a task_variant; compose from the immutable base")
    spec = VariantSpec.model_validate(spec)
    base_id = row.get("instance_id") or row.get("task_id")
    if not isinstance(base_id, str) or not base_id:
        raise ValueError("base row needs a nonempty instance_id or task_id")
    messages = row.get("responses_create_params", {}).get("input")
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) or message.get("role") not in {"system", "user"} for message in messages
    ):
        raise ValueError("only fresh system/user task inputs are supported")
    users = [message for message in messages if message["role"] == "user"]
    systems = [message for message in messages if message["role"] == "system"]
    if len(users) != 1 or len(systems) > 1:
        raise ValueError("fresh tasks require exactly one user message and at most one system message")
    user = content_text(users[0]["content"])
    system = content_text(systems[0]["content"]) if systems else ""
    family = spec.prompt_family
    context = {
        "issue": row.get("problem_statement", ""),
        "rules": spec.task_rules or "",
        "workdir": row.get("workdir", ""),
    }
    if family.user is not None:
        if not context["issue"]:
            raise ValueError("replacing a user template requires problem_statement")
        user = _render(family.user, context, spec.tool_names)
    if family.system is not None:
        system = _render(family.system, context, spec.tool_names)
    instructions = []
    for instruction in spec.instructions:
        resolved = instruction.model_dump(mode="json")
        resolved["placement"] = instruction.placement.model_dump(exclude_none=True)
        resolved["instruction_text"] = resolve_tool_text(instruction.instruction_text, spec.tool_names)
        resolved["rubric"] = resolve_tool_text(instruction.rubric, spec.tool_names)
        instructions.append(resolved)
    texts = {"system_prompt": system, "user_prompt": user}
    for surface in texts:
        before = [
            i["instruction_text"] for i in instructions if i["placement"] == {"surface": surface, "position": "start"}
        ]
        after = [
            i["instruction_text"] for i in instructions if i["placement"] == {"surface": surface, "position": "end"}
        ]
        texts[surface] = "\n\n".join(part for part in [*before, texts[surface], *after] if part)
    result = deepcopy(row)
    rendered = []
    if texts["system_prompt"]:
        rendered.append({"role": "system", "content": texts["system_prompt"]})
    rendered.append({"role": "user", "content": texts["user_prompt"]})
    # Preserve the exact original representation for a baseline (including content blocks).
    if spec.instructions or family.user is not None or family.system is not None:
        result["responses_create_params"]["input"] = rendered
    receipt = {
        "schema_version": 1,
        "base_task_id": base_id,
        "base_sha256": digest(row),
        **spec.model_dump(mode="json"),
        "instructions": instructions,
        "actor_input_sha256": actor_input_digest(result["responses_create_params"]["input"]),
    }
    receipt["variant_id"] = "variant_" + digest(receipt)
    result["task_variant"] = receipt
    return result
