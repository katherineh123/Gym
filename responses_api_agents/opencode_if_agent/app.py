# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose real OpenCode with a reversible per-run model boundary and an independent IF judge."""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from aiohttp import ClientTimeout
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field, field_validator

from nemo_gym.sandbox.agent_tools import sandbox_server_url
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.server_utils import request as http_request
from nemo_gym.task_variants.builder import content_text, digest
from nemo_gym.task_variants.judge import JudgeConfig, RubricJudge
from nemo_gym.task_variants.schema import VariantSpec
from responses_api_agents.opencode_if_agent.boundary import model_request, native_response, rewrite_sse
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
    OpenCodeSandboxedAgentVerifyResponse,
)


def validate_variant(row: dict[str, Any]) -> dict[str, Any]:
    """Reject stale/edited generated rows rather than grading a different prompt or rubric."""
    variant = row.get("task_variant")
    if not isinstance(variant, dict) or variant.get("schema_version") != 1:
        raise ValueError("a generated schema_version=1 task_variant is required")
    if variant.get("variant_id") != "variant_" + digest({k: v for k, v in variant.items() if k != "variant_id"}):
        raise ValueError("variant ID does not match its specification")
    VariantSpec.model_validate({key: variant[key] for key in VariantSpec.model_fields})
    actor = [{"role": item["role"], "content": item["content"]} for item in row["responses_create_params"]["input"]]
    if digest(actor) != variant["actor_input_sha256"]:
        raise ValueError("actor input changed after variant generation")
    return variant


class OpenCodeIFConfig(OpenCodeSandboxedAgentConfig):
    """Use one worker per agent instance; concurrent runs have independent in-memory routes."""

    num_workers: int = 1
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    @field_validator("num_workers")
    @classmethod
    def single_worker(cls, value: int) -> int:
        if value != 1:
            raise ValueError("OpenCode IF requires one worker per server instance for run-scoped routing")
        return value


class OpenCodeIFRunRequest(OpenCodeSandboxedAgentRunRequest):
    task_variant: dict[str, Any]


@dataclass
class VariantRun:
    attempt_id: str
    variant: dict[str, Any]
    system_text: str
    upstream_url: str = ""
    upstream_key: str = field(default="", repr=False)
    request_count: int = 0
    first_request: dict[str, Any] | None = None
    request_sha256: list[str] = field(default_factory=list)
    error: str | None = None


class OpenCodeIFAgent(OpenCodeSandboxedAgent):
    """Reuse native execution/resource verification; vary only the actor/model handoff."""

    config: OpenCodeIFConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._variant_runs: dict[str, VariantRun] = {}
        self._judge = RubricJudge(self.config.judge)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/if/{attempt_id}/v1/chat/completions", response_model=None)(self.variant_chat_completions)
        return app

    async def _create_opencode_config(self, request: Request) -> dict[str, Any]:
        config = await super()._create_opencode_config(request)
        state: VariantRun = request.state._ng_if_run
        options = config["provider"]["nemo_gym"]["options"]
        state.upstream_url = options["baseURL"]
        state.upstream_key = options.get("apiKey", "dummy_key")
        options["baseURL"] = (
            sandbox_server_url(self.config.name, require_reachable=True) + f"/if/{state.attempt_id}/v1"
        )
        # Native OpenCode's own agent prompt option replaces its selected base system
        # prompt while retaining native environment/reminder handling. A plain supplied
        # system overlay otherwise appends at the real model boundary.
        if state.variant["prompt_family"]["system"] is not None:
            config.setdefault("agent", {}).setdefault("build", {})["prompt"] = state.system_text
            state.system_text = ""
        return config

    async def variant_chat_completions(self, request: Request, attempt_id: str) -> JSONResponse | StreamingResponse:
        state = self._variant_runs.get(attempt_id)
        if state is None or not state.upstream_url:
            raise HTTPException(404, "Unknown or completed variant attempt")
        try:
            outgoing = model_request(
                await request.json(), tool_names=state.variant["tool_names"], system_text=state.system_text
            )
        except (ValueError, KeyError, TypeError) as exc:
            state.error = type(exc).__name__
            raise HTTPException(422, "Invalid variant model request or unavailable tool alias") from exc
        state.request_count += 1
        state.request_sha256.append(digest(outgoing))
        if state.first_request is None:
            state.first_request = outgoing
        response = None
        try:
            response = await http_request(
                "POST",
                state.upstream_url.rstrip("/") + "/chat/completions",
                json=outgoing,
                headers={"Authorization": f"Bearer {state.upstream_key}"},
                cookies=request.cookies,
                timeout=ClientTimeout(total=self.config.sandbox_timeout),
                _max_num_tries=1,
            )
            response.raise_for_status()
            if not outgoing.get("stream"):
                result = native_response(await response.json(), state.variant["tool_names"])
                response.release()
                return JSONResponse(result)
        except Exception as exc:
            state.error = type(exc).__name__
            if response is not None:
                response.release()
            raise HTTPException(502, "Variant model upstream failed") from exc

        async def stream():
            try:
                async for part in rewrite_sse(response.content.iter_any(), state.variant["tool_names"]):
                    yield part
            except BaseException as exc:
                state.error = type(exc).__name__
                raise
            finally:
                response.release()

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def run(self, request: Request, body: OpenCodeIFRunRequest) -> OpenCodeSandboxedAgentVerifyResponse:
        variant = validate_variant(body.model_dump(mode="json"))
        if body.responses_create_params.instructions:
            raise ValueError("put system text in input messages, not the unsupported instructions parameter")
        systems = [item for item in body.responses_create_params.input if item.role == "system"]
        state = VariantRun(
            attempt_id=uuid4().hex,
            variant=variant,
            system_text="\n\n".join(content_text(item.content) for item in systems),
        )
        self._variant_runs[state.attempt_id] = state
        request.state._ng_if_run = state
        try:
            result = await super().run(request, body)
            visible_output = []
            for item in result.response.output:
                value = item.model_dump(mode="json")
                if value.get("type") == "reasoning":
                    continue
                if value.get("type") == "function_call":
                    value["name"] = variant["tool_names"].get(value["name"], value["name"])
                visible_output.append(value)
            evidence = {
                "complete": state.request_count > 0 and not state.error and not result.opencode_failed,
                "initial_messages": (state.first_request or {}).get("messages", []),
                "tools": (state.first_request or {}).get("tools", []),
                "messages": visible_output,
                "model_patch": getattr(result, "model_patch", None),
            }
            result.if_result = await self._judge.grade(variant["instructions"], evidence)
            result.variant_receipt = {
                "variant_id": variant["variant_id"],
                "base_task_id": variant["base_task_id"],
                "attempt_id": state.attempt_id,
                "harness": "opencode",
                "opencode_version": self.config.opencode_version,
                "model_boundary": {
                    "first_request": state.first_request,
                    "request_count": state.request_count,
                    "request_sha256": state.request_sha256,
                    "error": state.error,
                },
            }
            return result
        finally:
            self._variant_runs.pop(state.attempt_id, None)
            del request.state._ng_if_run


if __name__ == "__main__":
    OpenCodeIFAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeIFAgent.run_webserver()
