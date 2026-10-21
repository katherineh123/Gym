# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attempt-scoped model transport for the native OpenCode harness."""

from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import replace

from aiohttp import ClientResponseError, ClientTimeout
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import JsonValue

from environment_servers.agentic_if.drivers.opencode_observations import prefix_model_calls
from environment_servers.agentic_if.model_boundary import ModelBoundary
from environment_servers.agentic_if.model_streaming import rewrite_stream
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient, raise_for_status
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


class ModelGateway:
    """Present selected tools and collect evidence, without executing or grading tasks."""

    def __init__(
        self,
        *,
        spec: RuntimeSpec,
        params: NeMoGymResponseCreateParamsNonStreaming,
        client: ServerClient,
        model_server: str,
        run_id: str,
        timeout_s: float,
        api_path: str = "/v1",
    ) -> None:
        self.boundary = ModelBoundary(spec, params)
        if spec.tools.tool_names is not None:
            # Missing frozen provenance cannot recover through native HTTP retries.
            # Validate it before launching the harness, using the same receipt reader.
            prefix_model_calls(spec.continuation)
        self.client, self.model_server, self.run_id = client, model_server, run_id
        self.timeout_s = timeout_s
        self.api_path = api_path.rstrip("/")
        self.root_session: str | None = None
        self._validated_prompt_sessions: set[str] = set()
        self.observations: list[dict[str, JsonValue]] = []
        self.issues: list[str] = []
        self._tool_calls: dict[tuple[str, str], dict[str, JsonValue] | None] = {}

    def tool_call(self, *, session_id: str, call_id: str, native_tool: str) -> dict[str, JsonValue]:
        """Return dispatch provenance, never credentials or hidden tool inventory."""
        key = (session_id, call_id)
        if key not in self._tool_calls:
            raise ValueError("unobserved tool call")
        recorded = self._tool_calls[key]
        if recorded is None:
            raise ValueError("ambiguous tool call")
        # Only OpenCode's native lowercase recovery is accepted. The model's
        # original spelling must independently have been an exposed alias.
        return {
            "allowed": bool(
                recorded["allowed"] and native_tool in {recorded["native_name"], recorded["native_name"].lower()}
            ),
            "model_name": recorded["model_name"],
            "available_names": recorded["available_names"],
        }

    def _record_dispatch(
        self,
        session_id: str,
        available_names: set[str],
        call_id: str,
        model_name: str,
        native_name: str,
        allowed: bool,
    ) -> None:
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("missing tool call identity")
        key = (session_id, call_id)
        if key in self._tool_calls:
            self._tool_calls[key] = None
            raise ValueError("duplicate tool call identity")
        self._tool_calls[key] = {
            "model_name": model_name,
            "native_name": native_name,
            "allowed": allowed,
            "available_names": sorted(available_names),
        }

    def _restore_rejected_calls(
        self, outgoing: dict[str, JsonValue], session_id: str, *, native_payload: dict[str, JsonValue]
    ) -> None:
        """Present the observed call after native recovery, without changing recovery.

        OpenCode's invalid-tool recovery still runs normally. Its repaired input
        is not what the model generated, though, and can expose private tool IDs.
        Never infer original arguments from that input or from another session.
        """
        if self.boundary.spec.tools.tool_names is None:
            # Omitted selection preserves native recovery/history unchanged.
            return
        # Recovery is an execution fact, not a display name: a selected tool
        # may legitimately be advertised as "invalid".
        invalid_calls = {
            call.get("id")
            for message in native_payload.get("messages", [])
            for call in message.get("tool_calls") or []
            if call.get("function", {}).get("name") == "invalid"
        }
        rejections = {}
        historical = prefix_model_calls(self.boundary.spec.continuation) if session_id == self.root_session else {}
        for message in outgoing.get("messages", []):
            for call in message.get("tool_calls") or []:
                invalid = call.get("id") in invalid_calls
                matches = []
                for observation in self.observations:
                    if observation.get("session_id") != session_id:
                        continue
                    for choice in observation.get("response", {}).get("choices", []):
                        for original in choice.get("message", {}).get("tool_calls") or []:
                            if original.get("id") == call.get("id"):
                                matches.append(original.get("function"))
                if not matches:
                    receipt = historical.get(call.get("id"))
                    if receipt is None:
                        if not invalid:
                            continue
                        raise ValueError(
                            "unobserved rejected model call in native history; prefix requires captured call evidence"
                        )
                    matches = [receipt["function"]]
                    available_names = receipt["available_names"]
                elif invalid:
                    dispatch = self._tool_calls.get((session_id, call["id"]))
                    if dispatch is None:
                        raise ValueError("missing rejected-call dispatch evidence")
                    available_names = dispatch["available_names"]
                if len(matches) != 1 or not isinstance(matches[0], dict):
                    raise ValueError("ambiguous rejected model call in native history")
                call["function"] = deepcopy(matches[0])
                if invalid:
                    available = ", ".join(available_names) or "none"
                    rejections[call["id"]] = (
                        f"{matches[0]['name']} is unavailable or its arguments are invalid. "
                        f"Available tools: {available}."
                    )
        for message in outgoing.get("messages", []):
            if message.get("role") == "tool" and message.get("tool_call_id") in rejections:
                message["content"] = rejections[message["tool_call_id"]]

    async def forward(
        self, payload: dict[str, JsonValue], *, headers: dict[str, str]
    ) -> JSONResponse | StreamingResponse:
        """Forward one native model call; mapping state belongs to this request."""
        response = None
        observation = None
        try:
            request_kind = headers.get("x-opencode-request-kind")
            if request_kind not in {None, "actor", "compaction"}:
                raise ValueError("unknown native OpenCode request kind")
            compaction = request_kind == "compaction"
            if compaction and (not headers.get("x-opencode-assistant-message-id") or payload.get("tools")):
                raise ValueError("native compaction requires an assistant identity and no tools")
            actor = not compaction and bool(headers.get("x-opencode-assistant-message-id") or payload.get("tools"))
            session_id = headers.get("x-session-id", self.run_id)
            if actor and self.root_session is None:
                self.root_session = session_id
            prepared = (
                self.boundary.prepare(
                    payload,
                    child_session=session_id != self.root_session,
                    # Validate the initial task handoff, not the harness's
                    # subsequent context management. Native compaction may
                    # replace that original user turn with a summary.
                    require_user_prompt=session_id not in self._validated_prompt_sessions,
                )
                if actor
                else None
            )
            if prepared is not None:
                self._validated_prompt_sessions.add(session_id)
            if prepared is not None and prepared.restricted:
                available_names = prepared.allowed_names
                prepared = replace(
                    prepared, observe_dispatch=lambda *args: self._record_dispatch(session_id, available_names, *args)
                )
            outgoing = deepcopy(prepared.payload if prepared else payload)
            if actor:
                self._restore_rejected_calls(outgoing, session_id, native_payload=payload)
            limit = outgoing.pop("max_completion_tokens", None)
            if limit is not None:
                outgoing["max_tokens"] = min(outgoing.get("max_tokens") or limit, limit)
            if actor:
                observation = {
                    "source": "model",
                    "session_id": session_id,
                    "assistant_message_id": headers.get("x-opencode-assistant-message-id"),
                    "request": deepcopy(outgoing),
                    "execution_tools": dict(prepared.execution_tools),
                }
                self.observations.append(observation)
            elif compaction:
                # Keep internal requests for audit, without applying task
                # prompts or tool presentation.
                observation = {
                    "source": "compaction",
                    "session_id": session_id,
                    "assistant_message_id": headers["x-opencode-assistant-message-id"],
                    "request": deepcopy(outgoing),
                    "execution_tools": {},
                }
                self.observations.append(observation)

            def observe(raw: dict[str, JsonValue]) -> None:
                if observation is not None:
                    observation["response"] = deepcopy(raw)
                    if prepared is not None:
                        observation["execution_tools"] = dict(prepared.execution_tools)

            response = await self.client.post(
                server_name=self.model_server,
                url_path=self.api_path + "/chat/completions",
                json=outgoing,
                headers={
                    name: value
                    for name, value in headers.items()
                    if name in {"x-session-id", "x-opencode-assistant-message-id"}
                },
                timeout=ClientTimeout(total=self.timeout_s),
                _max_num_tries=1,
            )
            try:
                await raise_for_status(response)
            except ClientResponseError:
                if observation is not None:
                    # A rejected HTTP request produced no actor response. Keep
                    # it for audit; the native session/message join determines
                    # whether a later native retry actually recovered it.
                    observation["transport_error"] = {"phase": "http_rejected", "status": response.status}
                raise
            if not outgoing.get("stream"):
                raw = await response.json()
                result = prepared.native_reply(raw) if prepared else raw
                observe(raw)
                response.release()
                return JSONResponse(result)
        except BaseException as error:
            if observation is None or "transport_error" not in observation:
                self.issues.append(type(error).__name__)
            if response is not None:
                response.release()
            raise

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for part in rewrite_stream(response.content.iter_any(), prepared=prepared, observe=observe):
                    yield part
            except BaseException as error:
                self.issues.append(type(error).__name__)
                raise
            finally:
                response.release()

        return StreamingResponse(stream(), media_type="text/event-stream")
