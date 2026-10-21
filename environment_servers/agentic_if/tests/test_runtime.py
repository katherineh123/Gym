# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The common execution boundary preserves resource-owned identity and isolation."""

import pytest

from environment_servers.agentic_if.runtime import BoundSandbox, ModelRoute, RunContext
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


class SandboxTransport:
    """External provider double; execution bindings are the observable wire contract."""

    def __init__(self):
        self.calls = []

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        return SandboxExecResult(stdout="result", stderr="", return_code=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["agent", 1000, None, "root"])
async def test_commands_keep_bound_identity_and_workdir(user):
    transport = SandboxTransport()
    sandbox = BoundSandbox(transport, user=user, workdir="/app/project", timeout_s=60)
    result = await sandbox.exec(command="pwd", env={"X": "literal"})
    assert result.stdout == "result"
    assert transport.calls == [
        {"command": "pwd", "user": user, "cwd": "/app/project", "timeout_s": 60, "env": {"X": "literal"}}
    ]


@pytest.mark.asyncio
async def test_timeout_is_bounded_without_replacing_an_explicit_cwd():
    transport = SandboxTransport()
    sandbox = BoundSandbox(transport, user="agent", workdir="/app", timeout_s=60)
    await sandbox.exec(command="pwd", cwd="/app/src", timeout_s=10)
    await sandbox.exec(command="pwd", timeout_s=999)
    assert [(call["timeout_s"], call["cwd"]) for call in transport.calls] == [(10, "/app/src"), (60, "/app")]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_execution_budget_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        BoundSandbox(SandboxTransport(), user=None, workdir="/app", timeout_s=timeout)


def make_context(tmp_path, *, spec=None, params=None):
    return RunContext(
        run_id="run-1",
        spec=spec or RuntimeSpec(harness="opencode", task_backend="swe"),
        params=params or NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Fix it"}]),
        sandbox=BoundSandbox(SandboxTransport(), user="agent", workdir="/app", timeout_s=60),
        model=ModelRoute(base_url="http://gym/attempt/run-1/v1", model="policy", api_key="attempt-token"),
        artifacts_dir=tmp_path,
    )


def test_context_detaches_mutable_inputs_from_other_runs(tmp_path):
    spec = RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": {"bash": "Run"}})
    params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Literal {braces}"}])
    first = make_context(tmp_path, spec=spec, params=params)
    second = make_context(tmp_path, spec=spec, params=params)
    spec.tools.tool_names["bash"] = "Changed"
    first.spec.tools.tool_names["bash"] = "FirstOnly"
    params.input[0].content = "Replaced"
    assert second.spec.tools.tool_names == {"bash": "Run"}
    assert first.params.input[0].content == "Literal {braces}"
    assert second.params.input[0].content == "Literal {braces}"


def test_model_route_does_not_expose_its_token_in_repr():
    route = ModelRoute(base_url="http://gym/attempt/x/v1", model="policy", api_key="private-attempt-token")
    assert "private-attempt-token" not in repr(route)
    assert route.api_key == "private-attempt-token"
