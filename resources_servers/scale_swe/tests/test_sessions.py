# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment Server sessions: typed seed, the /verify cookie, and close."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.scale_swe.app import ScaleSWEResourcesServer, ScaleSWEResourcesServerConfig


ROW = {"instance_id": "task", "workdir": "/repo", "image_url": "image"}
EPISODE = EpisodeId(rollout_id="rollout")


def fake_sandbox(**overrides) -> SimpleNamespace:
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="base\n", stderr="")),
        serialize=AsyncMock(return_value={"sandbox_id": "sb"}),
        stop=AsyncMock(),
        _handle=SimpleNamespace(sandbox_id="sb"),
    )
    for name, value in overrides.items():
        setattr(sandbox, name, value)
    return sandbox


@pytest.fixture
def server(monkeypatch) -> ScaleSWEResourcesServer:
    server = ScaleSWEResourcesServer(
        config=ScaleSWEResourcesServerConfig(
            host="",
            port=0,
            name="",
            entrypoint="",
            sandbox_provider="sandbox",
            sandbox_config={},
            apply_anti_cheating=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=lambda body: fake_sandbox()))
    return server


def seed(session_id: str = "attempt", **task_data) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EPISODE,
        task_id=TaskId(taskset="scale_swe", task_id="task"),
        task_data=ROW | task_data,
    )


def close(episode_id: EpisodeId = EPISODE) -> ResourcesCloseSessionRequest:
    return ResourcesCloseSessionRequest(resources_session_id="attempt", episode_id=episode_id)


def test_environment_server_session_contract(server):
    check_resources_session_contract(server.setup_webserver(), seed(), keeps_state=True)


@pytest.mark.asyncio
async def test_typed_seed_returns_sandbox_access_and_binds_the_verify_cookie(server):
    request = SimpleNamespace(session={SESSION_ID_KEY: "cookie"})
    response = await server.seed_session(request, seed())

    assert request.session[SESSION_ID_KEY] == "attempt"
    assert server._session_id_to_base_commit["attempt"] == "base"
    assert response.sandbox_access.connection.provider_config_ref == "sandbox"
    assert response.sandbox_access.connection.descriptor == {"sandbox_id": "sb"}
    assert response.sandbox_access.workdir == "/repo"
    # A retried seed returns the same session rather than starting a second sandbox.
    assert await server.seed_session(request, seed()) == response
    server._create_sandbox.assert_awaited_once()
    with pytest.raises(HTTPException, match="different request"):
        await server.seed_session(request, seed(instance_id="other"))


@pytest.mark.asyncio
async def test_failed_seed_stops_its_sandbox(server):
    sandbox = fake_sandbox(serialize=AsyncMock(side_effect=OSError("provider unavailable")))
    server._create_sandbox.side_effect = None
    server._create_sandbox.return_value = sandbox

    with pytest.raises(OSError):
        await server.seed_session(SimpleNamespace(session={}), seed())
    sandbox.stop.assert_awaited_once()
    assert "attempt" not in server._session_id_to_sandbox


@pytest.mark.asyncio
async def test_close_stops_the_sandbox_and_a_failed_stop_stays_retryable(server):
    await server.seed_session(SimpleNamespace(session={}), seed())
    sandbox = server._session_id_to_sandbox["attempt"]
    sandbox.stop.side_effect = [OSError("temporary"), None]

    with pytest.raises(OSError):
        await server.close_resources_session(close())
    assert server._session_id_to_sandbox["attempt"] is sandbox
    await server.close_resources_session(close())
    await server.close_resources_session(close())
    assert sandbox.stop.await_count == 2
    assert "attempt" not in server._session_id_to_sandbox
    with pytest.raises(HTTPException, match="already closed"):
        await server.seed_session(SimpleNamespace(session={}), seed())


@pytest.mark.asyncio
async def test_close_for_another_episode_is_refused(server):
    await server.seed_session(SimpleNamespace(session={}), seed())

    with pytest.raises(HTTPException, match="episode_id"):
        await server.close_resources_session(close(EpisodeId(rollout_id="other")))
    server._session_id_to_sandbox["attempt"].stop.assert_not_awaited()
