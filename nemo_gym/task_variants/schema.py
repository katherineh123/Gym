# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, serializable choices for fresh-task variants (no runtime sampling)."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TOOLS = frozenset({"bash", "read", "write", "edit", "glob", "grep", "apply_patch", "task", "todowrite"})
TAXONOMY = frozenset(
    {
        "IF-FORMAT",
        "IF-COVERAGE",
        "IF-CITE",
        "IF-PLAN",
        "IF-REASONACT",
        "IF-LANG",
        "IF-TOOLPREF",
        "IF-TOOLGROUND",
        "IF-SCHEMA",
        "IF-NEWINSTR",
        "IF-SCOPE",
        "IF-LENGTH",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Placement(StrictModel):
    """An initial instruction location; replay/tool-output placements are not accepted."""

    surface: Literal["system_prompt", "user_prompt"]
    position: Literal["start", "end"] = "end"


class Instruction(StrictModel):
    """Actor-visible wording plus private judge rubric and semantic taxonomy labels."""

    id: str = Field(min_length=1)
    instruction_text: str = Field(min_length=1)
    placement: Placement
    rubric: str = Field(min_length=1)
    taxonomy: list[str] = Field(min_length=1)
    provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("taxonomy")
    @classmethod
    def known_taxonomy(cls, labels: list[str]) -> list[str]:
        if len(labels) != len(set(labels)) or not set(labels) <= TAXONOMY:
            raise ValueError("taxonomy must contain unique catalog labels")
        return labels


class PromptFamily(StrictModel):
    """Frozen templates. None retains the existing prompt; templates are not executable Jinja."""

    id: str = "native"
    system: str | None = None
    user: str | None = None
    revision: str = "1"


class VariantSpec(StrictModel):
    """All resolved variation axes; only the actual OpenCode harness is currently supported."""

    harness: Literal["opencode"] = "opencode"
    seed: int = 0
    prompt_family: PromptFamily = Field(default_factory=PromptFamily)
    # Required when replacing a source's already-templated user prompt: normalization is explicit.
    task_rules: str | None = None
    tool_names: dict[str, str] = Field(default_factory=dict)
    instructions: list[Instruction] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_composition(self) -> "VariantSpec":
        if len({item.id for item in self.instructions}) != len(self.instructions):
            raise ValueError("instruction IDs must be unique within a variant")
        if not set(self.tool_names) <= TOOLS:
            raise ValueError("unsupported native OpenCode tool in tool_names")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) for name in self.tool_names.values()):
            raise ValueError("invalid tool alias")
        names = [self.tool_names.get(tool, tool) for tool in TOOLS]
        if len(names) != len(set(names)):
            raise ValueError("tool alias collision with another alias or native tool")
        if self.prompt_family.user is not None:
            if self.task_rules is None:
                raise ValueError("set task_rules explicitly before replacing an already-templated user prompt")
            if self.prompt_family.user.count("${issue}") != 1 or "${rules}" not in self.prompt_family.user:
                raise ValueError("user template must include ${issue} exactly once and ${rules}")
        return self
