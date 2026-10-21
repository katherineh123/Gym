# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sole IF run lifecycle: resources, native execution, cleanup, then grading."""

import asyncio
import json
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ConfigDict, JsonValue, PrivateAttr, model_validator

from environment_servers.agentic_if import resources
from environment_servers.agentic_if.config import AgenticIFConfig
from environment_servers.agentic_if.drivers.opencode import OpenCodeDriver
from environment_servers.agentic_if.model_gateway import ModelGateway
from environment_servers.agentic_if.runtime import ModelRoute, RunContext, RunOutcome
from environment_servers.agentic_if.scoring import ScoringSession
from environment_servers.agentic_if.workspace import snapshot_workspace, stage_workspace_files, verify_materialized_cwd
from nemo_gym.base_environment_server import BaseEnvironmentServer, CleanupContext, HandledEpisodeError
from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.global_config import TOKEN_ID_CAPTURE_BLOCK
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.sandbox.agent_tools import sandbox_server_url
from nemo_gym.server_utils import (
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
    rollout_path_prefix,
)
from resources_servers.agentic_if.if_grading.contracts import IFResult, SessionEvidence
from resources_servers.agentic_if.task_variants.materialization import (
    AgenticIFTaskInput,
    sealed_task_input,
    validate_task_identity,
)
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, validate_runtime


LOG = logging.getLogger(__name__)


class AgenticIFResult(BaseVerifyResponse):
    """Native correctness and IF compliance remain distinct in one run result."""

    model_config = ConfigDict(extra="allow")
    agentic_if: RuntimeSpec
    if_result: IFResult
    execution_status: Literal["completed", "length", "timeout", "error"]
    evidence: SessionEvidence
    run_id: str
    artifacts: dict[str, JsonValue]


class AgenticIFRequest(BaseEpisodeRequest[AgenticIFTaskInput]):
    """One native Gym episode; the collector supplies attempt identity."""

    @model_validator(mode="before")
    @classmethod
    def validate_content_identity(cls, value: object) -> object:
        """Reject a task whose ID is not the digest of its content, and seal the row's task input."""
        if isinstance(value, dict) and isinstance(value.get("task"), dict):
            validate_task_identity(value["task"])
            # Collation keeps a flat row's id inside task_input; nothing downstream sees that copy.
            value = {**value, "task": {**value["task"], "task_input": sealed_task_input(value["task"])}}
        return value


class AgenticIFFailure(EpisodeFailure):
    """Retain the partial IF result using Gym's native diagnostic convention."""

    partial_response: AgenticIFResult | None = None


class AgenticIFResponse(BaseEpisodeResponse[AgenticIFResult]):
    """Native episode identity with either the completed IF result or failure."""

    failure: AgenticIFFailure | None = None


@dataclass(frozen=True)
class _Attempt:
    context: RunContext
    gateway: ModelGateway
    rollout_id: str | None


def _save(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    # Native JavaScript slicing may leave a lone surrogate in a raw receipt.
    # JSON escapes preserve it losslessly even when the response view is invalid.
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class AgenticIFEnvironment(BaseEnvironmentServer[AgenticIFRequest, AgenticIFResponse]):
    """Run one sealed task in OpenCode on its task backend, then have the paired Resources Server grade it."""

    config: AgenticIFConfig
    request_model = AgenticIFRequest
    response_model = AgenticIFResponse
    _driver: OpenCodeDriver = PrivateAttr()
    _attempts: dict[str, _Attempt] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        """Build the OpenCode driver from the deployment settings."""
        super().model_post_init(__context)
        settings = self.config.opencode
        self._driver = OpenCodeDriver(
            version=settings.version,
            binary_sha256=settings.binary_sha256,
            context_limit=settings.context_limit,
            max_turns=settings.max_turns,
            binary_path=settings.binary_path,
            permission=settings.permission,
        )

    def setup_webserver(self) -> FastAPI:
        """Add the attempt-scoped model and tool-call routes that the OpenCode process calls back."""
        app = super().setup_webserver()
        app.post("/attempt/{attempt_id}/v1/chat/completions", response_model=None)(self.model_chat)
        app.post("/attempt/{attempt_id}/tool-call")(self.tool_call)
        return app

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Delegate to the paired Resources Server: it owns IF scoring, so it owns the metrics over its results."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    def _attempt(self, request: Request, attempt_id: str) -> _Attempt:
        attempt = self._attempts.get(attempt_id)
        if attempt is None or not secrets.compare_digest(
            request.headers.get("authorization", ""), "Bearer " + attempt.context.model.api_key
        ):
            raise HTTPException(404, "Unknown or completed attempt")
        return attempt

    async def model_chat(self, request: Request, attempt_id: str) -> JSONResponse | StreamingResponse:
        """Forward one of the attempt's native Chat Completions calls to the policy model."""
        attempt = self._attempt(request, attempt_id)
        # The native process makes a new HTTP request; it cannot inherit the
        # collector's ContextVar from /run. Recover only its bound attempt ID.
        with rollout_context(attempt.rollout_id):
            return await attempt.gateway.forward(await request.json(), headers=dict(request.headers))

    async def tool_call(self, request: Request, attempt_id: str) -> dict[str, JsonValue]:
        """Authenticate a native hook's lookup against the existing attempt."""
        attempt = self._attempt(request, attempt_id)
        body = await request.json()
        fields = {"session_id", "call_id", "native_tool"}
        if (
            not isinstance(body, dict)
            or set(body) != fields
            or any(not isinstance(value, str) or not value or len(value) > 512 for value in body.values())
        ):
            raise HTTPException(422, "Invalid tool call identity")
        try:
            return attempt.gateway.tool_call(**body)
        except ValueError as error:
            raise HTTPException(409, "Tool call provenance unavailable") from error

    async def run(self, request: AgenticIFRequest, cleanup: CleanupContext) -> AgenticIFResponse:
        """Seed scoring, run the task in its sandbox, release the sandbox, then grade the recorded session."""
        body = request.task.task_input
        row = body.model_dump(mode="json", exclude_unset=True)
        spec = validate_runtime(row)
        if spec.task_backend not in self.config.task_resources:
            raise ValueError("requested task backend is not enabled")
        scoring = ScoringSession(
            client=self.server_client,
            server_name=self.config.resources_server.name,
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            row=row,
        )
        # Registered before seeding, so a lost seed reply is still closed; cleanup runs it last.
        cleanup.register_cleanup("agentic-if-scoring-session", scoring.close)
        try:
            await scoring.seed()
        except Exception as error:
            raise HandledEpisodeError(
                AgenticIFFailure(
                    failure_reason=f"IF scoring rejected the task: {error}"[:2000],
                    failure_kind="agentic_if:invalid_task",
                    terminal=True,
                    stage="seed",
                )
            ) from error
        base_url = sandbox_server_url(self.config.name, require_reachable=True)
        run_id = uuid4().hex
        directory = self.config.artifacts_dir / run_id
        response = NeMoGymResponse(
            id="not-started-" + run_id,
            created_at=int(time()),
            model=self.config.model_server.name,
            object="response",
            output=[],
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        outcome = RunOutcome(
            response=response,
            status="error",
            evidence=SessionEvidence(
                status="infrastructure_error",
                steps=[],
                issues=["native execution did not produce evidence"],
            ),
        )
        resource_config = self.config.task_resources[spec.task_backend]
        session = resources.SandboxTaskSession(
            client=self.server_client,
            server_name=resource_config.server.name,
            row=row,
            client_session_id=run_id,
            timeout_s=self.config.sandbox_timeout,
            default_agent_user=resource_config.agent_user,
            episode_id=request.episode_id,
            task_id=request.task.task_id,
        )
        verified, gateway, context = None, None, None
        errors: list[str] = []
        workspace_initial_diff: str | None = None
        workspace_snapshot_observed = False

        async def close_attempt() -> None:
            self._attempts.pop(run_id, None)
            await session.close()

        closer = cleanup.register_cleanup("agentic-if-task-session", close_attempt)
        try:
            await session.prepare()
            await verify_materialized_cwd(session.sandbox, spec.provenance.get("materialized_cwd"))
            await stage_workspace_files(session.sandbox, spec.workspace_files)
            workspace_initial_diff = await snapshot_workspace(session.sandbox)
            workspace_snapshot_observed = True
            gateway = ModelGateway(
                spec=spec,
                params=body.responses_create_params,
                client=self.server_client,
                model_server=self.config.model_server.name,
                run_id=run_id,
                timeout_s=session.sandbox.timeout_s,
                api_path=self._model_path(request),
            )
            context = RunContext(
                run_id=run_id,
                spec=spec,
                params=body.responses_create_params,
                sandbox=session.sandbox,
                model=ModelRoute(
                    base_url=base_url.rstrip("/") + f"/attempt/{run_id}/v1",
                    model=self.config.model_server.name,
                    api_key=secrets.token_urlsafe(32),
                ),
                artifacts_dir=directory,
                model_observations=gateway.observations,
            )
            self._attempts[run_id] = _Attempt(context, gateway, request.episode_id.capture_key)
            outcome = await self._driver.execute(context)
            verified = await session.finish(outcome.response)
        # Cancellation is not caught: it unwinds through BaseEnvironmentServer's bounded CleanupContext.
        except Exception as error:
            LOG.exception("IF attempt %s failed during preparation, execution, or verification", run_id)
            errors.append(type(error).__name__)
        try:
            await closer.close()
        except Exception as error:
            errors.append(type(error).__name__)
        evidence = outcome.evidence.model_copy(deep=True)
        if context is not None and context.sandbox.workdir is not None:
            evidence.facts.setdefault("workdir", context.sandbox.workdir)
        if workspace_snapshot_observed:
            evidence.facts["workspace_initial_diff"] = workspace_initial_diff
        errors += gateway.issues if gateway else []
        if errors:
            evidence.status = "infrastructure_error"
            evidence.issues = list(dict.fromkeys(evidence.issues + errors))
        await asyncio.to_thread(
            _save,
            directory / "evidence.json",
            {
                "row": row,
                "evidence": evidence.model_dump(mode="json"),
                "model_observations": gateway.observations if gateway else [],
            },
        )
        try:
            if_result = await scoring.verify(evidence)
        except Exception as error:
            raise HandledEpisodeError(
                AgenticIFFailure(
                    failure_reason=f"IF scoring failed: {error}"[:2000],
                    failure_kind="agentic_if:infrastructure_error",
                    terminal=True,
                    stage="verification",
                )
            ) from error
        result = dict(
            verified
            or {
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "agentic_if:infrastructure_error",
                "failure_reason": "Task verification was not completed",
            }
        )
        result.setdefault("responses_create_params", body.responses_create_params.model_dump(mode="json"))
        result.setdefault("response", outcome.response.model_dump(mode="json"))
        execution_status = "error" if errors else outcome.status
        result.update(
            agentic_if=spec.model_dump(mode="json"),
            if_result=if_result.model_dump(mode="json"),
            execution_status=execution_status,
            evidence=evidence.model_dump(mode="json"),
            run_id=run_id,
            artifacts=outcome.artifacts | {"if_evidence": str(directory / "evidence.json")},
        )
        output = AgenticIFResult.model_validate(result)
        await asyncio.to_thread(_save, directory / "result.json", output.model_dump(mode="json"))
        if (
            verified is None
            or result.get("evaluation_completed") is False
            or execution_status in {"error", "timeout"}
            or evidence.status != "complete"
        ):
            detail = result.get("failure_reason") or "; ".join(evidence.issues) or f"Episode {execution_status}"
            return AgenticIFResponse(
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                failure=AgenticIFFailure(
                    failure_reason=str(detail)[:2000],
                    failure_kind=result.get("failure_kind") or "agentic_if:episode_failed",
                    # Execution and diagnostics are already settled. Do not
                    # silently rerun a task; a caller can explicitly retry it.
                    terminal=True,
                    partial_response=output,
                ),
            )
        return AgenticIFResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=output)

    def _model_path(self, request: AgenticIFRequest) -> str:
        # Upstream Environment Servers capture training tokens when the run enables capture for all agents or for
        # the agent they call. This server calls no agent, so only ``all_agents`` applies.
        settings = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        capture = bool(settings.get("enabled", False)) and bool(settings.get("all_agents", False))
        return rollout_path_prefix(request.episode_id.capture_key, token_capture=capture) + "/v1"


if __name__ == "__main__":
    AgenticIFEnvironment.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgenticIFEnvironment.run_webserver()
