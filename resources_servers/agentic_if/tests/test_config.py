# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped configuration keeps the settings the benchmark's results depend on."""

from pathlib import Path

import yaml

from resources_servers.agentic_if.if_grading.configuration import load_config


GYM = Path(__file__).parents[3]
CONFIG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "agentic_if.yaml").read_text())


def test_the_judge_model_server_serves_the_model_judge_yaml_names():
    reference = CONFIG["agentic_if_resources"]["resources_servers"]["agentic_if"]["judge_model_server"]
    assert reference["type"] == "responses_api_models"
    server = CONFIG[reference["name"]]["responses_api_models"]["openai_model"]
    # The model server replaces the request's model with its own, so the two must name the same judge.
    assert server["openai_model"] == load_config()["model"]["name"]


def test_the_opencode_permission_policy_is_opencode_sandboxed_agents():
    harness = CONFIG["agentic_if_environment"]["environment_servers"]["agentic_if"]["opencode"]
    path = GYM / "responses_api_agents" / "opencode_sandboxed_agent" / "configs" / "opencode_sandboxed_agent.yaml"
    upstream = yaml.safe_load(path.read_text())["opencode_sandboxed_agent"]["responses_api_agents"]
    policy = upstream["opencode_sandboxed_agent"]["opencode_config"]["permission"]
    # Later matches win, so the order of the rules is part of the policy.
    assert list(harness["permission"].items()) == list(policy.items())
    assert list(harness["permission"]["bash"].items()) == list(policy["bash"].items())
