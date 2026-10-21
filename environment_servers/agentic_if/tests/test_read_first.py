# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored read guards survive compilation and run inside the native hook."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from environment_servers.agentic_if.drivers import opencode_hooks
from environment_servers.agentic_if.drivers.opencode_hooks import compile_hooks
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, ToolConfiguration


GUARD = {
    "edit": {"unread": "Read {{path}} first.", "changed": "Read {{path}} again."},
    "write": {"unread": "Read {{path}} before overwriting."},
}


def test_guard_survives_hook_compilation():
    spec = RuntimeSpec(harness="opencode", task_backend="swe", tools={"read_first": GUARD})
    assert compile_hooks(spec)["read_first"] == GUARD
    assert spec.tools.model_dump(mode="json")["read_first"] == GUARD


@pytest.mark.parametrize(
    "guard, message",
    [
        ({}, "non-empty"),
        ({"read": {"unread": "No"}}, "'edit' or 'write'"),
        ({"edit": {}}, "non-empty"),
        ({"edit": {"unread": " "}}, "blank"),
        ({"edit": {"unread": "Use {{other}}"}}, "placeholder"),
        ({"edit": {"other": "No"}}, "'unread' or 'changed'"),
    ],
    ids=["empty", "unknown_tool", "empty_cases", "blank", "unknown_placeholder", "unknown_case"],
)
def test_invalid_guard_is_rejected_by_the_runtime_contract(guard, message):
    with pytest.raises(ValueError, match=message):
        ToolConfiguration(read_first=guard)


@pytest.mark.parametrize("selected", [{"edit": "edit", "write": "write"}, {"read": "read"}])
def test_guard_requires_its_native_tools(selected):
    with pytest.raises(ValueError, match="read_first"):
        RuntimeSpec(harness="opencode", task_backend="swe", tools={"read_first": GUARD, "tool_names": selected})


def test_absent_guard_does_not_change_existing_sealed_tool_identity():
    assert ToolConfiguration().model_dump(mode="json") == {
        "tool_names": None,
        "tool_descriptions": {},
        "tool_parameter_descriptions": {},
        "tool_interfaces": {},
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required for native hook")
def test_read_first_execution_and_resumed_session_state(tmp_path):
    plugin = tmp_path / "profile.mjs"
    plugin.write_text(Path(opencode_hooks.__file__).with_name("if-profile.js").read_text())
    config = tmp_path / "profile.json"
    config.write_text(json.dumps({"profile_sha256": "test", "read_first": GUARD}))
    env = os.environ | {
        "IF_PROFILE_PLUGIN": str(plugin),
        "IF_TEST_ROOT": str(tmp_path),
        "NEMO_GYM_OPENCODE_PROFILE": str(config),
        "NEMO_GYM_OPENCODE_PROFILE_APPLIED": str(tmp_path / "applied.jsonl"),
        "NEMO_GYM_OPENCODE_PLANTS": str(tmp_path / "plants.json"),
        "NEMO_GYM_OPENCODE_PLANTS_FIRED": str(tmp_path / "fired.jsonl"),
    }
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("read_first.test.mjs"))],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "read-first native behavior passed" in result.stdout
