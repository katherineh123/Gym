# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool profiles: the profile hash (opencode_profile.py) and the plugin's hooks (if-profile.js, run under Node)."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from environment_servers.agentic_if.drivers.opencode_profile import profile_sha256


HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "if_profile" / "profile_fixture.json").read_text())


def test_hash_ignores_key_order_and_existing_hash() -> None:
    reordered = json.loads(json.dumps(FIXTURE, sort_keys=True))
    assert profile_sha256(reordered) == profile_sha256(FIXTURE)
    assert profile_sha256({**FIXTURE, "profile_sha256": "x"}) == profile_sha256(FIXTURE)
    changed = copy.deepcopy(FIXTURE)
    changed["tools"]["read"]["description"] += " "
    assert profile_sha256(changed) != profile_sha256(FIXTURE)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("script", ["plugin_hooks", "plants_hooks"])
def test_plugin_hooks(tmp_path: Path, script: str) -> None:
    plugin = tmp_path / "if-profile.mjs"
    plugin.write_text((HERE.parent / "drivers" / "if-profile.js").read_text())
    env = {
        "PATH": os.pathsep.join([str(Path(shutil.which("node")).parent), os.defpath]),
        "TMPDIR": str(tmp_path),
        "HOME": str(tmp_path),
        "IF_PROFILE_PLUGIN": str(plugin),
        "NEMO_GYM_OPENCODE_PROFILE": str(HERE / "if_profile" / "profile_fixture.json"),
        "NEMO_GYM_OPENCODE_PROFILE_APPLIED": str(tmp_path / "applied.jsonl"),
    }
    out = subprocess.run(
        ["node", str(HERE / "if_profile" / f"{script}.test.mjs")], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "all checks passed" in out.stdout
