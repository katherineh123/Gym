# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution contracts between the Environment Server and the OpenCode driver."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


class BoundSandbox:
    """Execute as the task identity; staging transport remains explicitly separate.

    This enforces driver configuration, not an OS security boundary. Trusted
    harness bootstrap may use ``transport`` for root-owned staging operations;
    agent execution must use this bound interface. Lifecycle methods deliberately
    are not forwarded: the runner/resource integration owns them.
    """

    def __init__(
        self,
        transport: AsyncSandbox,
        *,
        user: str | int | None,
        workdir: str | None,
        timeout_s: float,
        execution_env: Mapping[str, str] | None = None,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("execution timeout must be finite and positive")
        self.transport = transport
        self.user = user
        self.workdir = workdir
        self.timeout_s = timeout_s
        self.execution_env = dict(execution_env or {})

    async def exec(
        self,
        *,
        command: str,
        timeout_s: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxExecResult:
        """Preserve explicit image-default identity and cap per-command timeouts."""
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("command timeout must be finite and positive")
        return await self.transport.exec(
            command=command,
            user=self.user,
            cwd=self.workdir if cwd is None else cwd,
            timeout_s=min(self.timeout_s, timeout_s) if timeout_s is not None else self.timeout_s,
            env=(self.execution_env | dict(env or {})) if self.execution_env or env is not None else None,
        )

    async def upload(self, source: Path, destination: str) -> None:
        """Stage an artifact through the provider; this is not an agent shell call."""
        await self.transport.upload(source, destination)

    async def download(self, source: str, destination: Path) -> None:
        """Collect an artifact without changing the actor identity."""
        await self.transport.download(source, destination)


@dataclass(frozen=True)
class ModelRoute:
    """Attempt-scoped model gateway; provider credentials never enter task rows."""

    base_url: str
    model: str
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class RunContext:
    """One prepared run, with detached settings and no server-global session maps."""

    run_id: str
    spec: RuntimeSpec
    params: NeMoGymResponseCreateParamsNonStreaming
    sandbox: BoundSandbox
    model: ModelRoute
    artifacts_dir: Path
    # The model transport appends attempt-local observations while the native
    # process runs. Drivers consume the same sequence after execution finishes.
    model_observations: Sequence[Mapping[str, JsonValue]] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        # Frozen dataclasses alone do not isolate mutable Pydantic subobjects.
        object.__setattr__(self, "spec", self.spec.model_copy(deep=True))
        object.__setattr__(self, "params", self.params.model_copy(deep=True))


@dataclass(frozen=True)
class RunOutcome:
    """Execution/evidence only; the task verifier and IF scorer are separate owners."""

    response: NeMoGymResponse
    evidence: SessionEvidence
    # "length": the last assistant turn stopped at its output-token limit.
    status: Literal["completed", "length", "timeout", "error"]
    artifacts: dict[str, str] = field(default_factory=dict)
