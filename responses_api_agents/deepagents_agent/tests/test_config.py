# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


def test_module_parses():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    src = app_path.read_text()
    compile(src, str(app_path), "exec")


def test_config_yaml_parses():
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "deepagents_agent.yaml"
    data = yaml.safe_load(cfg_path.read_text())
    assert "deepagents_agent" in data
    inner = data["deepagents_agent"]["responses_api_agents"]["deepagents_agent"]
    assert inner["entrypoint"] == "app.py"
    assert inner["max_turns"] == 30
    assert inner["backend"] == "state"
    assert inner["system_prompt"] is None
    assert inner["chat_template_kwargs"]["enable_thinking"] is True
    assert inner["chat_template_kwargs"]["truncate_history_thinking"] is False
