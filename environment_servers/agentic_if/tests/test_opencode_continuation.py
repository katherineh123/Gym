# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay must restore real workspace state and preserve native continuation boundaries."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from environment_servers.agentic_if.bootstrap import BinarySandbox
from environment_servers.agentic_if.runtime import BoundSandbox
from environment_servers.agentic_if.tests.support import LocalTransport, Provider, context, driver, exported
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [None, ""])
async def test_empty_workspace_diff_accepts_native_empty_output(tmp_path, output):
    from environment_servers.agentic_if.drivers.opencode_replay import _workspace_diff

    class EmptyOutput(Provider):
        async def exec(self, **kwargs):
            return SandboxExecResult(stdout=output, stderr=None, return_code=0)

    assert await _workspace_diff(context(tmp_path, EmptyOutput())) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_ms", [None, 100])
async def test_restore_replays_shell_as_agent_and_restores_binary_diff(tmp_path, timeout_ms):
    from environment_servers.agentic_if.drivers.opencode_replay import restore_workspace

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (repo / "file").write_text("before")
    git("add", ".")
    git("commit", "-qm", "base")
    binary = bytes([0, 42, 255, 0])
    (repo / "new.bin").write_bytes(binary)
    git("add", "-N", ".")
    diff = git("diff", "--binary", "--full-index")
    git("reset", "-q")
    (repo / "new.bin").unlink()
    prefix = exported("root")
    prefix["info"]["directory"] = str(repo)
    arguments = {"command": "printf changed > file; printf drift"}
    if timeout_ms is not None:
        arguments.update(command=arguments["command"] + "; sleep 1; printf late > ../late", timeout=timeout_ms)
    prefix["messages"][1]["parts"] = [
        {
            "type": "tool",
            "tool": "bash",
            "callID": "first",
            "state": {
                "status": "completed",
                "input": arguments,
                "output": "original",
                "metadata": {},
                "time": {"end": 3},
            },
        }
    ]
    continuation = ContinuationSpec(
        mode="restore",
        prefix_export=prefix,
        workspace_diff=diff,
        source_task_sha256="a" * 64,
        source_tools_sha256="b" * 64,
    )
    transport = LocalTransport()
    ctx = replace(
        context(tmp_path, transport), sandbox=BoundSandbox(transport, user="agent", workdir=str(repo), timeout_s=60)
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    receipt = await restore_workspace(ctx, BinarySandbox(ctx.sandbox, str(staging)), continuation)
    assert (repo / "new.bin").read_bytes() == binary
    assert (repo / "file").read_text() == "before"
    assert receipt["workspace_matches_diff"] is True
    assert receipt["command_output_mismatch_indices"] == [0]
    assert not (tmp_path / "late").exists()
    assert all(
        user == "agent" for command, user in transport.calls if "git " in command or "printf changed" in command
    )


class ReplayProvider(Provider):
    def __init__(self):
        super().__init__(child=False)
        self.live = False
        self.prefix = exported("root", text="Earlier decision")
        self.prefix["info"]["directory"] = "/app/repo"
        self.observations[0]["assistant_message_id"] = "a-fresh"
        self.observations[0]["response"]["choices"][0]["message"]["content"] = "Fresh decision"

    async def exec(self, **kwargs):
        if "opencode run " in kwargs["command"]:
            self.live = True
        if "pwd" in kwargs["command"]:
            self.calls.append(kwargs)
            return SandboxExecResult(stdout="/app/repo\n", stderr="", return_code=0)
        return await super().exec(**kwargs)

    async def download(self, source, destination):
        if "export-root" in source:
            document = json.loads(json.dumps(self.prefix))
            if self.live:
                more = exported("root", text="Fresh decision")["messages"]
                for message in more:
                    message["info"]["id"] += "-fresh"
                    message["info"]["time"]["created"] += 10
                document["messages"].extend(more)
            Path(destination).write_text(json.dumps(document))
            return
        await super().download(source, destination)


def restore_spec(prefix_export):
    """A restore continuation of a recorded native export, with no workspace change."""
    continuation = {
        "mode": "restore",
        "prefix_export": prefix_export,
        "workspace_diff": "",
        "source_task_sha256": "a" * 64,
        "source_tools_sha256": "b" * 64,
    }
    return RuntimeSpec(harness="opencode", task_backend="swe", continuation=continuation)


@pytest.mark.asyncio
async def test_prefix_then_real_continuation_uses_same_native_session(tmp_path):
    provider = ReplayProvider()
    ctx = context(tmp_path, provider)
    result = await driver(tmp_path).execute(replace(ctx, spec=restore_spec(provider.prefix)))
    assert result.status == "completed"
    assert result.evidence.continuation_start == 2
    assert result.evidence.steps[1].text == "Earlier decision"
    assert result.evidence.steps[-1].text == "Fresh decision"
    assert next(item for item in result.response.output if item.type == "message").content[0].text == "Fresh decision"
    assert "Earlier decision" not in result.response.model_dump_json()
    (launch,) = [call for call in provider.calls if "opencode run " in call["command"]]
    assert "--session root" in launch["command"]
    assert "--model nemo_gym/dummy_model" in launch["command"]


@pytest.mark.asyncio
async def test_restored_history_must_match_the_recorded_tool_outputs(tmp_path):
    provider = ReplayProvider()
    provider.prefix["messages"][1]["parts"].append(
        {
            "type": "tool",
            "tool": "read",
            "callID": "read-one",
            "state": {"status": "completed", "input": {"filePath": "/app/file"}, "output": "changed"},
        }
    )
    source = json.loads(json.dumps(provider.prefix))
    source["messages"][1]["parts"][-1]["state"]["output"] = "recorded"
    result = await driver(tmp_path).execute(replace(context(tmp_path, provider), spec=restore_spec(source)))
    assert result.evidence.facts["replay"]["tool_outputs_match"] is False
    assert result.status == "error"
    assert result.evidence.status == "incomplete"
    assert "restored_prefix_history_diverged" in result.evidence.issues
