# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate sealed agentic IF rows and grade the evidence their Environment Server records."""

import asyncio
import logging
from functools import partial
from time import monotonic
from typing import Any, ClassVar

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.agentic_if.episode_contracts import AgenticIFVerification, AgenticIFVerifyRequest
from resources_servers.agentic_if.if_grading.contracts import IFResult, ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.engine import grade
from resources_servers.agentic_if.if_grading.judge import Judge
from resources_servers.agentic_if.task_variants.materialization import validate_task_identity
from resources_servers.agentic_if.task_variants.runtime import validate_runtime


LOG = logging.getLogger(__name__)


class AgenticIFResourcesServerConfig(BaseResourcesServerConfig):
    """Grading settings; each sealed row carries its own checks."""

    # Verification grades the evidence of a seeded episode, which a rollout record does not carry.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    # The model server that answers judge questions. It must serve the model `if_grading/data/judge.yaml` names: the
    # judge's prompts and model settings are sent as that file sets them.
    judge_model_server: ModelServerRef
    max_concurrent_grades: int = Field(default=8, gt=0)
    # A session spans the whole harness run before verification, which can take several hours.
    session_ttl_seconds: float = Field(default=6 * 3600, gt=0, allow_inf_nan=False)


class SeededAgenticIFEpisode(BaseModel):
    """The checks one Environment-owned episode is graded against."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    episode_id: EpisodeId
    task_id: TaskId
    checks: list[ResolvedCheck]
    expires_at: float


class AgenticIFResourcesServer(SimpleResourcesServer):
    """Seed validates a sealed task and keeps its checks; verify grades the episode's evidence against them."""

    config: AgenticIFResourcesServerConfig
    _grading_slots: asyncio.Semaphore = PrivateAttr()
    _grade_jobs: set[asyncio.Task[IFResult]] = PrivateAttr(default_factory=set)
    _seeded: dict[str, SeededAgenticIFEpisode] = PrivateAttr(default_factory=dict)
    _closed: dict[str, float] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._grading_slots = asyncio.Semaphore(self.config.max_concurrent_grades)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        """Validate the sealed task's identity and receipts, and keep its checks for the episode's verification."""
        try:
            validate_task_identity({"task_id": body.task_id.model_dump(mode="json"), "task_input": body.task_data})
            checks = validate_runtime(body.task_data).checks
        except ValueError as error:
            detail = error.errors(include_input=False) if isinstance(error, ValidationError) else str(error)
            raise HTTPException(status_code=422, detail=detail) from error
        self._expire_sessions()
        if body.resources_session_id in self._closed:
            raise HTTPException(status_code=409, detail="Resources session is already closed")
        session_id = request.session[SESSION_ID_KEY]
        existing = self._seeded.get(session_id)
        if existing is not None:
            # The task identity is a content digest, so a matching identity means the same checks.
            if (existing.resources_session_id, existing.episode_id, existing.task_id) != (
                body.resources_session_id,
                body.episode_id,
                body.task_id,
            ):
                raise HTTPException(status_code=409, detail="Resources session is already seeded for another task")
            return ResourcesSeedSessionResponse(resources_session_id=existing.resources_session_id)
        self._seeded[session_id] = SeededAgenticIFEpisode(
            resources_session_id=body.resources_session_id,
            episode_id=body.episode_id,
            task_id=body.task_id,
            checks=checks,
            expires_at=monotonic() + self.config.session_ttl_seconds,
        )
        return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)

    async def verify(self, request: Request, body: AgenticIFVerifyRequest) -> AgenticIFVerification:
        """Grade the seeded episode's checks against the evidence the Environment Server recorded."""
        seeded = self._seeded_episode(request)
        if body.episode_id != seeded.episode_id or body.task_id != seeded.task_id:
            raise HTTPException(status_code=409, detail="Verification identity does not match the seeded episode")
        grading = asyncio.create_task(self._grade(seeded.checks, body.verification_input.evidence))
        self._grade_jobs.add(grading)
        grading.add_done_callback(self._graded)
        # Cancelling the caller does not cancel grading: the job owns its slot until the judge finishes.
        return AgenticIFVerification(if_result=await asyncio.shield(grading))

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Forget the episode's checks; a closed session ID cannot be seeded again until it expires."""
        self._expire_sessions()
        session_id = request.session[SESSION_ID_KEY]
        seeded = self._seeded.get(session_id)
        if seeded is not None:
            if body.resources_session_id != seeded.resources_session_id or body.episode_id != seeded.episode_id:
                raise HTTPException(status_code=409, detail="Resources session does not match the active episode")
            del self._seeded[session_id]
        self._closed.setdefault(body.resources_session_id, monotonic() + self.config.session_ttl_seconds)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Instruction-following rates over scored rollouts, beside the task reward Gym already reports.

        A rollout whose applicable checks all passed counts as an IF pass. Rollouts with no applicable check, or
        with a check that could not be graded, are counted separately and excluded from the rates.
        """
        rollouts = [rollout for task in tasks for rollout in task]
        graded, not_applicable, errored = [], 0, 0
        for rollout in rollouts:
            if_result = rollout.get("if_result") or {}
            statuses = [check.get("status") for check in if_result.get("checks") or []]
            if if_result.get("if_all") is not None:
                graded.append(rollout)
            elif "error" in statuses:
                errored += 1
            else:
                not_applicable += 1
        metrics: dict[str, Any] = {
            "if/graded": len(graded),
            "if/not_applicable": not_applicable,
            "if/error": errored,
        }
        if graded:
            metrics["mean/if_all"] = sum(rollout["if_result"]["if_all"] for rollout in graded) / len(graded)
            metrics["mean/if_frac"] = sum(rollout["if_result"]["if_frac"] for rollout in graded) / len(graded)
            metrics["mean/task_and_if"] = sum(
                1.0 for rollout in graded if rollout["if_result"]["if_all"] == 1 and rollout.get("reward", 0) >= 1
            ) / len(graded)
        return metrics

    async def _grade(self, checks: list[ResolvedCheck], evidence: SessionEvidence) -> IFResult:
        # The job, rather than its HTTP waiter, owns this slot. Cancelling a waiter cannot free capacity while
        # the judge is still busy.
        judge = Judge(partial(self._ask_judge, asyncio.get_running_loop()))
        async with self._grading_slots:
            return await asyncio.to_thread(grade, checks, evidence, judge=judge)

    def _ask_judge(self, loop: asyncio.AbstractEventLoop, body: dict[str, JsonValue]) -> NeMoGymResponse:
        # Grading runs on a worker thread; the request runs on the server's event loop, where Gym's HTTP client lives.
        request = call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=body,
            response_model=NeMoGymResponse,
        )
        return asyncio.run_coroutine_threadsafe(request, loop).result()

    def _graded(self, task: asyncio.Task[IFResult]) -> None:
        self._grade_jobs.discard(task)
        if not task.cancelled() and task.exception() is not None:
            LOG.error("IF grading job failed", exc_info=task.exception())

    def _seeded_episode(self, request: Request) -> SeededAgenticIFEpisode:
        self._expire_sessions()
        seeded = self._seeded.get(request.session[SESSION_ID_KEY])
        if seeded is None:
            raise HTTPException(status_code=404, detail="No active agentic IF task. Call /seed_session first.")
        return seeded

    def _expire_sessions(self) -> None:
        now = monotonic()
        for session_id, seeded in tuple(self._seeded.items()):
            if seeded.expires_at <= now:
                del self._seeded[session_id]
        for resources_session_id, expires_at in tuple(self._closed.items()):
            if expires_at <= now:
                del self._closed[resources_session_id]


if __name__ == "__main__":
    AgenticIFResourcesServer.run_webserver()
