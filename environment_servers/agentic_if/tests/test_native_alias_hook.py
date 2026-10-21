# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from environment_servers.agentic_if.drivers import opencode_profile as profile


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to exercise native hooks")
def test_hook_blocks_guessed_native_names_without_affecting_native_tools(tmp_path):
    config = tmp_path / "profile.json"
    config.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "id": "guard",
                "tools": {},
                "visible_tools": {"read": "ReadSource", "bash": "bash"},
                "check_aliases": True,
            }
        )
    )
    script = Path(__file__).with_name("native_alias.test.mjs")
    result = subprocess.run(
        ["node", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ
        | {
            "IF_TEST_ROOT": str(tmp_path),
            "IF_PROFILE_PLUGIN": Path(profile.__file__).with_name("if-profile.js").as_uri(),
            "NEMO_GYM_OPENCODE_PROFILE": str(config),
            "NEMO_GYM_OPENCODE_PROFILE_APPLIED": str(tmp_path / "applied.jsonl"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native alias guard passed" in result.stdout
