# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialized prompt facts must agree with the actual bound task directory."""

import os

import pytest

from environment_servers.agentic_if.runtime import BoundSandbox
from environment_servers.agentic_if.tests.support import LocalTransport, Transport, run_episode, task_row


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["physical", "symlink", "omitted"])
async def test_cwd_fact_accepts_equivalent_directories_without_changing_identity(tmp_path, spelling):
    from environment_servers.agentic_if.workspace import verify_materialized_cwd

    task = tmp_path / "task root"
    task.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(task)
    transport = LocalTransport()
    sandbox = BoundSandbox(transport, user=os.getuid(), workdir=str(task), timeout_s=60)
    expected = {"physical": str(task), "symlink": str(alias), "omitted": None}[spelling]
    await verify_materialized_cwd(sandbox, expected)
    assert sandbox.workdir == str(task)
    assert all(user == os.getuid() for _, user in transport.calls)
    assert len(transport.calls) == (0 if expected is None else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("matches", [False, True])
async def test_episode_checks_materialized_cwd_before_staging_or_driver(setup, monkeypatch, matches):
    from nemo_gym.sandbox.providers import SandboxExecResult
    from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, seal_runtime_row

    agent, _, events, transports, verifications = setup
    original = Transport.exec

    async def execute(self, command=None, **kwargs):
        if command.startswith("pwd -P &&"):
            events.append("cwd-check")
            assert kwargs["user"] is None
            return SandboxExecResult(stdout="/app\n" + ("/app\n" if matches else "/other\n"), stderr="", return_code=0)
        return await original(self, command, **kwargs)

    monkeypatch.setattr(Transport, "exec", execute)
    row = task_row(check=False)
    row.pop("agentic_if")
    row = seal_runtime_row(
        row, RuntimeSpec(harness="opencode", task_backend="swe", provenance={"materialized_cwd": "/app"})
    )
    result = await run_episode(agent, row, failed=not matches)
    assert transports[0].stops == 1
    if matches:
        assert events == ["prepare", "cwd-check", "execute", "verify", "cleanup"]
    else:
        assert events == ["prepare", "cwd-check", "cleanup"]
        assert result.evidence.status == "infrastructure_error"
        assert not verifications


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", ["different", "missing", "relative", "not-text", "newline"])
async def test_inaccurate_cwd_fails_before_actor_execution(tmp_path, expected):
    from environment_servers.agentic_if.workspace import verify_materialized_cwd

    task = tmp_path / "task"
    task.mkdir()
    other = tmp_path / "different"
    other.mkdir()
    value = {
        "different": str(other),
        "missing": str(tmp_path / "absent"),
        "relative": "task",
        "not-text": 1,
        "newline": "/task\n/other",
    }[expected]
    sandbox = BoundSandbox(LocalTransport(), user=os.getuid(), workdir=str(task), timeout_s=60)
    with pytest.raises((ValueError, RuntimeError), match="materialized_cwd"):
        await verify_materialized_cwd(sandbox, value)
    assert sandbox.workdir == str(task)
