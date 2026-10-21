# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment permissions reach OpenCode without imposing benchmark policy globally."""

import json
from dataclasses import replace

import pytest

from environment_servers.agentic_if.app import AgenticIFConfig, AgenticIFEnvironment
from environment_servers.agentic_if.config import HarnessSettings
from environment_servers.agentic_if.tests.support import Provider, context, driver
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


POLICY = {"*": "allow", "bash": {"*": "allow", "*git show*": "deny"}, "skill": "deny"}


def settings(tmp_path, **changes):
    binary = tmp_path / "configured-opencode"
    binary.write_bytes(b"pin")
    return HarnessSettings(binary_path=binary, binary_sha256="0" * 64, version="1.17.11", **changes)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [POLICY, {"*": "deny"}])
async def test_permissions_reach_native_launch(tmp_path, policy):
    provider = Provider(child=False)
    executor = replace(driver(tmp_path), permission=settings(tmp_path, permission=policy).permission)
    result = await executor.execute(context(tmp_path, provider))
    assert result.status == "completed"
    launch = next(call for call in provider.calls if "opencode run " in call["command"])
    native = json.loads(launch["env"]["OPENCODE_CONFIG_CONTENT"])
    assert native["permission"] == policy


def test_absent_permission_keeps_native_default_and_configurations_are_isolated(tmp_path):
    ctx = context(tmp_path, Provider())
    assert "permission" not in driver(tmp_path)._configuration(ctx)
    executor = replace(driver(tmp_path), permission=POLICY)
    first = executor._configuration(ctx)
    first["permission"]["bash"]["*git show*"] = "allow"
    assert executor._configuration(ctx)["permission"]["bash"]["*git show*"] == "deny"
    assert POLICY["bash"]["*git show*"] == "deny"


def test_environment_forwards_permission_to_opencode(setup):
    existing, *_ = setup
    values = existing.config.model_dump()
    values["opencode"]["permission"] = POLICY
    configured = AgenticIFEnvironment(
        config=AgenticIFConfig.model_validate(values), server_client=existing.server_client
    )
    assert configured._driver.permission == POLICY


def test_step_limit_reaches_opencode_only_when_configured(tmp_path):
    # An omitted limit keeps OpenCode's native unlimited steps, as the shipped configuration measures.
    assert settings(tmp_path).max_turns is None
    ctx = context(tmp_path, Provider())
    assert "steps" not in replace(driver(tmp_path), max_turns=None)._configuration(ctx)["agent"]["build"]
    assert driver(tmp_path)._configuration(ctx)["agent"]["build"]["steps"] == 5


@pytest.mark.parametrize("policy", [{"bash": "alow"}, {"bash": {"*": "alow"}}, {"bash": False}])
def test_invalid_permission_actions_fail_at_configuration_load(tmp_path, policy):
    with pytest.raises(ValueError, match="permission"):
        settings(tmp_path, permission=policy)


@pytest.mark.parametrize("selected", [{}, {"bash": "Run", "task": "Delegate"}])
def test_explicit_selection_disables_native_tools_globally_without_relaxing_policy(tmp_path, selected):
    policy = {"read": "allow", "*": "allow", "bash": {"*": "allow", "*git show*": "deny"}}
    ctx = replace(
        context(tmp_path, Provider()),
        spec=RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": selected}),
    )
    native = replace(driver(tmp_path), permission=policy)._configuration(ctx)
    permission = native["permission"]
    for tool in ["read", "edit", "write", "glob", "grep", "todowrite", "webfetch", "websearch", "skill"]:
        assert permission[tool] == "deny"
        assert list(permission).index(tool) > list(permission).index("*")
    assert permission["bash"] == (policy["bash"] if selected else "deny")
    assert ("task" not in permission) if selected else permission["task"] == "deny"
    assert "external_directory" not in permission
    assert policy["read"] == "allow"
    assert "permission" not in native["agent"]["build"]  # global policy reaches subagents too


@pytest.mark.parametrize("selected", [{"edit": "Modify", "write": "Create"}, {"apply_patch": "Patch"}])
def test_selected_editing_tools_do_not_lose_the_shared_edit_permission(tmp_path, selected):
    ctx = replace(
        context(tmp_path, Provider()),
        spec=RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": selected}),
    )
    permission = driver(tmp_path)._configuration(ctx)["permission"]
    assert "edit" not in permission
    assert permission["bash"] == "deny"


@pytest.mark.parametrize("selected", [{"write": "Write"}, {"edit": "Edit"}])
def test_native_edit_and_write_must_be_selected_together(selected):
    with pytest.raises(ValueError, match="edit and write.*together"):
        RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": selected})
