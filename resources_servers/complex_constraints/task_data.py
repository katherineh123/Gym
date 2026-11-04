# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the complex_constraints server (ComplexConstraints rubric-judged instruction following).

Mirrors ``ComplexConstraintsRunRequest`` (app.py): ``rubrics`` is required, and the prompt the policy answers is the
row's ``responses_create_params.input``. The judge never sees that prompt, only the reply and one criterion.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="The dataset's benchmark_id (CIF-001 to CIF-075); passed through, not read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    use_case: Optional[str] = Field(
        default=None,
        description="Work domain of the prompt, for example 'Logistics, Scheduling & Event Planning'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    instruction_type: Optional[str] = Field(
        default=None,
        description="Constraint type the prompt stresses (Conditional, Implicit, Multistep or Negative); a metrics subset.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    prompt_style: Optional[str] = Field(
        default=None,
        description="How the prompt is phrased (Direct, Context or Rambling/Stream-of-Consciousness prompting).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    rubrics: List[str] = Field(
        min_length=1,
        description="Expert-written criteria for the reply; each one is judged in its own call.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    data_source: Optional[str] = Field(
        default=None,
        description="Where prepare.py read the row: the pinned copy's URL or the official Hugging Face revision.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
