# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from responses_api_agents.opencode_if_agent.app import OpenCodeIFConfig


@pytest.mark.parametrize("dataset", ["scale_swe", "swe_rebench"])
def test_dataset_config_resolves_to_native_if_agent_and_original_grader(dataset, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "not-a-real-key")
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [f"responses_api_agents/opencode_if_agent/configs/{dataset}.yaml"],
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    agents = [key for key in config if OmegaConf.is_dict(config[key]) and "responses_api_agents" in config[key]]
    assert agents == ["opencode_if_agent"]
    fields = config[agents[0]].responses_api_agents.opencode_if_agent
    parsed = OpenCodeIFConfig.model_validate({**OmegaConf.to_container(fields, resolve=True), "name": agents[0]})
    assert parsed.num_workers == 1
    assert parsed.resources_server.name == f"{dataset}_resources_server"
    assert parsed.opencode_config["permission"]["bash"]["*git clone*"] == "deny"
    assert parsed.judge.model == "nvidia/zai-org/glm-5.3"
    assert "opensandbox" in config.sandbox
    assert config[f"{dataset}_resources_server"].resources_servers[dataset].apply_anti_cheating is True
    assert Path("responses_api_agents/opencode_if_agent/data/specs.json").exists()
