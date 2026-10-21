# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both workspace capture paths must retain Git's racy-entry content check."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from environment_servers.agentic_if.drivers import opencode_profile as profile
from environment_servers.agentic_if.drivers.opencode_replay import WORKSPACE_DIFF


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for workspace capture")
@pytest.mark.parametrize("capture", ["plugin", "replay"])
def test_same_size_same_second_edit_is_captured_without_changing_real_index(tmp_path, capture):
    if capture == "plugin" and shutil.which("node") is None:
        pytest.skip("Node is required to execute the native plugin")
    repo = tmp_path / "repo"
    repo.mkdir()
    env = os.environ | {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], env=env, capture_output=True, text=True, check=True, timeout=30
        ).stdout

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    # ctime cannot be restored. Ignore it so the test deterministically exercises
    # Git's same-size, same-mtime content check instead of depending on timing.
    git("config", "core.trustctime", "false")
    source = repo / "a.py"
    past = int(tmp_path.stat().st_mtime) - 100
    source.write_text("x = 1\n")
    os.utime(source, (past, past))
    git("add", "a.py")
    git("commit", "-qm", "baseline")
    source.write_text("x = 2\n")
    os.utime(source, (past, past))
    index = repo / ".git" / "index"
    os.utime(index, (past, past))
    index_bytes, index_mtime = index.read_bytes(), index.stat().st_mtime_ns
    expected = git("diff", "--binary", "--full-index")
    assert "-x = 1\n+x = 2" in expected, "Git itself must detect the racy entry"

    if capture == "replay":
        command = ["sh", "-c", WORKSPACE_DIFF]
    else:
        plugin = tmp_path / "profile.mjs"
        plugin.write_text(Path(profile.__file__).with_name("if-profile.js").read_text())
        config = tmp_path / "profile.json"
        config.write_text(json.dumps({"tools": {}}))
        env |= {
            "IF_PROFILE_PLUGIN": plugin.as_uri(),
            "NEMO_GYM_OPENCODE_PROFILE": str(config),
            "NEMO_GYM_OPENCODE_PROFILE_APPLIED": str(tmp_path / "applied.jsonl"),
            "NEMO_GYM_OPENCODE_PLANTS": "",
        }
        command = [
            "node",
            "--input-type=module",
            "--eval",
            """
const {IfProfile} = await import(process.env.IF_PROFILE_PLUGIN);
const hooks = await IfProfile({directory: process.cwd()});
const receipt = {output: "ok", metadata: {}};
await hooks["tool.execute.after"]({tool: "bash", sessionID: "root", callID: "edit"}, receipt);
process.stdout.write(receipt.metadata.nemo_gym_profile.workspace_diff);
""",
        ]
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
    assert index.read_bytes() == index_bytes
    assert index.stat().st_mtime_ns == index_mtime
