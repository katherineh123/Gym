# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment settings; per-task behavior lives only in the canonical task row."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FilePath, StrictInt

from nemo_gym.base_environment_server import BaseEnvironmentServerConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef


TaskBackend = Literal["swe", "tb21"]
PermissionAction = Literal["allow", "ask", "deny"]
OpenCodePermission = dict[str, PermissionAction | dict[str, PermissionAction]]


class ResourceSettings(BaseModel):
    """One task backend: the Resources Server that owns its sandboxes, and the default actor identity there."""

    model_config = ConfigDict(extra="forbid")
    server: ResourcesServerRef
    # Used only for rows that do not set ``agent_user`` themselves.
    agent_user: str | StrictInt | None = None


class HarnessSettings(BaseModel):
    """The pinned OpenCode binary and its native limits."""

    model_config = ConfigDict(extra="forbid")
    # None keeps OpenCode's native unlimited-step behavior.
    max_turns: int | None = Field(default=None, gt=0)
    context_limit: int = Field(default=131072, gt=0)
    # The opencode-linux-x64 npm package version; binary_sha256 pins its bin/opencode.
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # A host copy to upload into each task sandbox. Unset, each sandbox downloads the pinned package instead.
    binary_path: FilePath | None = None
    # OpenCode-native deployment policy, not a universal IF benchmark restriction.
    # Omit to preserve native defaults; patterns retain their configured order.
    permission: OpenCodePermission | None = None


class AgenticIFConfig(BaseEnvironmentServerConfig):
    """One server running the OpenCode harness on explicitly enabled task backends."""

    num_workers: Literal[1] = 1
    model_server: ModelServerRef
    # The paired agentic IF Resources Server: it validates each sealed task and grades its evidence.
    resources_server: ResourcesServerRef
    task_resources: dict[TaskBackend, ResourceSettings] = Field(min_length=1)
    opencode: HarnessSettings
    artifacts_dir: Path
    # Bounds the OpenCode run and each command and model call it makes.
    sandbox_timeout: float = Field(default=3600, gt=0, allow_inf_nan=False)
