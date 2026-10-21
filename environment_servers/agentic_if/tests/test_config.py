# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped environment-server block is a valid configuration of this server."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.agentic_if.config import AgenticIFConfig


CONFIG = Path(__file__).parents[3] / "resources_servers" / "agentic_if" / "configs" / "agentic_if.yaml"


def test_shipped_configuration_validates_and_bounds_each_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTIC_IF_OPENCODE_BINARY", raising=False)
    shipped = OmegaConf.load(CONFIG).agentic_if_environment.environment_servers.agentic_if
    values = OmegaConf.to_container(shipped, resolve=True)
    config = AgenticIFConfig.model_validate(values | {"host": "127.0.0.1", "port": 9001, "name": "agentic_if"})
    # Without a host copy, each task sandbox installs the pinned package itself.
    assert config.opencode.binary_path is None
    # The episode deadline leaves room for setup, verification, cleanup and grading after the harness budget.
    assert (config.default_episode_timeout_seconds or 0) > config.sandbox_timeout
