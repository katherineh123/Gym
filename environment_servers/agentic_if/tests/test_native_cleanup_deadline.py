# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native Environment Server, not a shield loop, owns cleanup deadlines."""

import asyncio

import pytest

from environment_servers.agentic_if.tests.support import episode, task_row


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["seed", "verify"])
async def test_episode_deadline_is_not_suppressed_by_hung_resource_operation(setup, phase):
    agent, controls, _, _, _ = setup
    agent.config.default_episode_timeout_seconds = 0.02
    agent.config.cleanup_timeout_seconds = 0.03
    release = asyncio.Event()
    setattr(controls, phase + "_release", release)
    run = asyncio.create_task(agent.run_request(episode(task_row(check=False))))
    await getattr(controls, phase + "_entered").wait()
    await asyncio.sleep(0.15)
    finished_within_budget = run.done()
    release.set()  # Let the old broken implementation unwind instead of hanging the test.
    result = await run
    assert finished_within_budget, "native episode + cleanup deadlines were bypassed"
    assert result.failure is not None
    assert agent._attempts == {}
