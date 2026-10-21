# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collection failures stay incomplete and retain private, bounded diagnostics."""

import json
from types import SimpleNamespace

import pytest

from environment_servers.agentic_if.drivers.opencode_sessions import collect_sessions
from nemo_gym.sandbox.providers import SandboxExecResult


class FailedCollection:
    directory = "/private/runtime"

    def __init__(self, inventory=False):
        self.inventory = inventory

    async def exec(self, **kwargs):
        if "opencode db" in kwargs["command"] and not self.inventory:
            return SandboxExecResult(stdout='[{"id":"root","parent_id":null}]', stderr="", return_code=0)
        return SandboxExecResult(
            stdout="", stderr="private-error-detail:" + "x" * 10000, return_code=124, error_type="Timeout"
        )

    async def download(self, *args):
        raise AssertionError("A failed export is not a valid session artifact")


@pytest.mark.asyncio
async def test_failed_export_retains_inventory_and_diagnostic(tmp_path, caplog):
    context = SimpleNamespace(artifacts_dir=tmp_path, run_id="test")
    exports, sessions, artifacts = await collect_sessions(context, FailedCollection(), {"API_KEY": "not-recorded"})
    assert exports == artifacts == {}
    assert [s.id for s in sessions] == ["root"]
    receipt = json.loads((tmp_path / "native-export-root-failure.json").read_text())
    assert receipt["return_code"] == 124
    assert receipt["error_type"] == "Timeout"
    assert len(receipt["stderr"]) == 8192
    assert receipt["stderr_truncated"] is True
    assert "not-recorded" not in json.dumps(receipt)
    assert "return_code=124" in caplog.text
    assert "private-error-detail" not in caplog.text


@pytest.mark.asyncio
async def test_failed_inventory_retains_diagnostic_without_accepting_empty_sessions(tmp_path):
    context = SimpleNamespace(artifacts_dir=tmp_path, run_id="test")
    with pytest.raises(RuntimeError, match="return_code=124"):
        await collect_sessions(context, FailedCollection(inventory=True), {})
    receipt = json.loads((tmp_path / "native-inventory-failure.json").read_text())
    assert receipt["return_code"] == 124
