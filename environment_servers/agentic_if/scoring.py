# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One episode's session with the paired agentic IF Resources Server."""

from uuid import uuid4

from pydantic import JsonValue

from environment_servers.agentic_if.resources import ResourcesSessionClient
from nemo_gym._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER
from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import ServerClient
from resources_servers.agentic_if.episode_contracts import (
    AgenticIFVerification,
    AgenticIFVerificationInput,
    AgenticIFVerifyRequest,
)
from resources_servers.agentic_if.if_grading.contracts import IFResult, SessionEvidence


class ScoringSession(ResourcesSessionClient):
    """Seed validates the sealed task and keeps its checks; verify grades the recorded evidence."""

    def __init__(
        self,
        *,
        client: ServerClient,
        server_name: str,
        episode_id: EpisodeId,
        task_id: TaskId,
        row: dict[str, JsonValue],
    ) -> None:
        self.client = client
        self.server_name = server_name
        self.episode_id = episode_id
        self.task_id = task_id
        self.row = row
        self.resources_session_id = f"agentic-if-{uuid4().hex}"
        self.cookies = {}

    async def seed(self) -> None:
        """Have the Resources Server validate the sealed task and keep its checks in a new session."""
        request = ResourcesSeedSessionRequest(
            resources_session_id=self.resources_session_id,
            episode_id=self.episode_id,
            task_id=self.task_id,
            task_data=self.row,
        )
        reply = ResourcesSeedSessionResponse.model_validate(
            await self._post("/seed_session", request.model_dump(mode="json"))
        )
        if reply.resources_session_id != self.resources_session_id:
            raise ValueError("IF scoring seed returned a different resources_session_id")
        if not self.cookies:
            raise ValueError("IF scoring seed did not establish a session cookie")

    async def verify(self, evidence: SessionEvidence) -> IFResult:
        """Grade the recorded session evidence against the seeded checks."""
        request = AgenticIFVerifyRequest(
            episode_id=self.episode_id,
            task_id=self.task_id,
            verification_input=AgenticIFVerificationInput(evidence=evidence),
        )
        reply = await self._post(
            "/verify",
            request.model_dump(mode="json"),
            # The registered cleanup closes the session, so the scorer keeps it until then.
            headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"},
        )
        return AgenticIFVerification.model_validate(reply).if_result

    async def close(self) -> None:
        """Release the scoring session."""
        request = ResourcesCloseSessionRequest(
            resources_session_id=self.resources_session_id, episode_id=self.episode_id
        )
        await self._post("/close_session", request.model_dump(mode="json"))
