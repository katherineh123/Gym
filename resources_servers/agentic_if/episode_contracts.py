# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verification contracts shared by the agentic IF Environment and Resources Servers."""

from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import ResourcesVerifyRequest
from resources_servers.agentic_if.if_grading.contracts import IFResult, SessionEvidence


class AgenticIFVerificationInput(BaseModel):
    """What the Environment Server observed during one attempt."""

    model_config = ConfigDict(extra="forbid")

    evidence: SessionEvidence


class AgenticIFVerifyRequest(ResourcesVerifyRequest[AgenticIFVerificationInput]):
    """Grade one Environment-owned agentic IF episode."""


class AgenticIFVerification(BaseModel):
    """Instruction-following compliance; task correctness stays with the task's own verifier."""

    model_config = ConfigDict(extra="forbid")

    if_result: IFResult
