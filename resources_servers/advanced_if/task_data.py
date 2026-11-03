# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the advanced_if server (AdvancedIF rubric-judged instruction following).

Mirrors ``AdvancedIfRunRequest`` (app.py): ``category`` and ``rubrics`` are required, and the conversation the
policy answers is the row's ``responses_create_params.input``, which ``verify()`` also renders for the judge.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[int] = Field(
        default=None,
        description="Row index in the pinned AdvancedIF CSV; passed through, not read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    category: Literal["complex_if", "carried_context", "system_steerability"] = Field(
        description=(
            "Paper category. system_steerability selects the system-steer judge prompt; every category is a "
            "metrics subset, and category_mean averages the three."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    benchmark_name: Optional[str] = Field(
        default=None,
        description="The dataset's own category label (for example complex_if_single_turn_v5); not read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    rubrics: List[str] = Field(
        description="Expert-written yes/no questions about the final assistant turn; all go to one judge call.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
