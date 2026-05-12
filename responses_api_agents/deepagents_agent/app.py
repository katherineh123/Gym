# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import sys
from asyncio import Semaphore
from time import time
from typing import Any, Optional, Sequence
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


class _SafeStderrHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = sys.__stderr__
            if stream is None:
                return
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            pass


if not LOG.handlers:
    LOG.addHandler(_SafeStderrHandler(level=logging.WARNING))


DEFAULT_CHAT_TEMPLATE_KWARGS = {
    "enable_thinking": True,
    "truncate_history_thinking": False,
}
_DEEPAGENTS_PROFILE_REGISTERED = False


def _ensure_deepagents_harness_profile() -> None:
    global _DEEPAGENTS_PROFILE_REGISTERED

    if _DEEPAGENTS_PROFILE_REGISTERED:
        return

    from deepagents import HarnessProfile, register_harness_profile

    register_harness_profile(
        "gym",
        HarnessProfile(excluded_middleware=frozenset({"SummarizationMiddleware"})),
    )
    _DEEPAGENTS_PROFILE_REGISTERED = True


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(getattr(part, "text", "") or str(part))
        return "".join(parts)
    return str(content)


def _json_dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, ensure_ascii=False)


def _extract_token_details(message: Any) -> dict[str, Any]:
    metadata = getattr(message, "response_metadata", None) or {}
    return {
        "prompt_token_ids": metadata.get("prompt_token_ids") or [],
        "generation_token_ids": metadata.get("generation_token_ids") or [],
        "generation_log_probs": metadata.get("generation_log_probs") or [],
    }


def _tool_calls_from_ai_message(message: Any) -> list[dict[str, Any]]:
    raw_tool_calls = (getattr(message, "additional_kwargs", None) or {}).get("tool_calls")
    if raw_tool_calls:
        return list(raw_tool_calls)

    tool_calls = []
    for tc in getattr(message, "tool_calls", []) or []:
        tool_calls.append(
            {
                "id": tc.get("id") or f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": _json_dumps(tc.get("args")),
                },
            }
        )
    return tool_calls


def _messages_to_output_items(messages: Sequence[Any]) -> list[Any]:
    output_items: list[Any] = []
    for message in messages:
        msg_type = getattr(message, "type", None)
        if msg_type == "ai":
            token_details = _extract_token_details(message)
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=getattr(message, "id", None) or f"msg_{uuid4().hex}",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=_content_to_text(getattr(message, "content", "")),
                            annotations=[],
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                    **token_details,
                )
            )
            for tc in _tool_calls_from_ai_message(message):
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn:
                    continue
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=fn.get("arguments", ""),
                        call_id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        type="function_call",
                        id=tc.get("id"),
                        status="completed",
                    )
                )
        elif msg_type == "tool":
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=getattr(message, "tool_call_id", "") or "",
                    output=_content_to_text(getattr(message, "content", "")),
                    status="completed",
                )
            )
    return output_items


def _chat_message_to_langchain(message: dict[str, Any]) -> Any:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    role = message.get("role")
    content = message.get("content") or ""
    if role == "user":
        return HumanMessage(content=content)
    if role in {"system", "developer"}:
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(content=content, tool_call_id=message.get("tool_call_id") or "")
    if role == "assistant":
        raw_tool_calls = message.get("tool_calls") or []
        parsed_tool_calls = []
        invalid_tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    args = {"value": args}
                parsed_tool_calls.append(
                    {
                        "name": fn.get("name", ""),
                        "args": args,
                        "id": tc.get("id"),
                        "type": "tool_call",
                    }
                )
            except json.JSONDecodeError:
                invalid_tool_calls.append(
                    {
                        "name": fn.get("name", ""),
                        "args": raw_args,
                        "id": tc.get("id"),
                        "error": "Invalid JSON arguments",
                        "type": "invalid_tool_call",
                    }
                )

        response_metadata = {
            key: message[key]
            for key in ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
            if key in message
        }
        return AIMessage(
            content=content,
            additional_kwargs={"tool_calls": raw_tool_calls} if raw_tool_calls else {},
            tool_calls=parsed_tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata=response_metadata,
        )
    raise NotImplementedError(f"Unsupported chat message role: {role!r}")


def _responses_input_to_langchain_messages(body: NeMoGymResponseCreateParamsNonStreaming) -> list[Any]:
    from responses_api_models.vllm_model.app import VLLMConverter

    normalized_body = body.model_copy(deep=True)
    if isinstance(normalized_body.input, str):
        normalized_body.input = [NeMoGymEasyInputMessage(role="user", content=normalized_body.input)]
    normalized_body.tools = []

    converter = VLLMConverter(return_token_id_information=True)
    chat_params = converter.responses_to_chat_completion_create_params(normalized_body)
    return [_chat_message_to_langchain(dict(message)) for message in chat_params.messages]


def _split_system_prompt(
    messages: Sequence[Any],
    configured_system_prompt: Optional[str],
) -> tuple[Optional[str], list[Any]]:
    messages = list(messages)
    if not messages:
        return configured_system_prompt, messages

    first = messages[0]
    if getattr(first, "type", None) != "system":
        return configured_system_prompt, messages

    if configured_system_prompt is not None:
        return configured_system_prompt, messages[1:]
    return _content_to_text(getattr(first, "content", "")), messages[1:]


def _langchain_message_to_chat_dict(message: Any) -> dict[str, Any]:
    msg_type = getattr(message, "type", None)
    if msg_type == "system":
        return {"role": "system", "content": getattr(message, "content", "")}
    if msg_type == "human":
        return {"role": "user", "content": getattr(message, "content", "")}
    if msg_type == "tool":
        return {
            "role": "tool",
            "content": _content_to_text(getattr(message, "content", "")),
            "tool_call_id": getattr(message, "tool_call_id", "") or "",
        }
    if msg_type == "ai":
        chat_message = {
            "role": "assistant",
            "content": _content_to_text(getattr(message, "content", "")) or None,
        }
        tool_calls = _tool_calls_from_ai_message(message)
        if tool_calls:
            chat_message["tool_calls"] = tool_calls
        chat_message.update(_extract_token_details(message))
        return chat_message

    role = getattr(message, "role", None)
    if role:
        return {"role": role, "content": _content_to_text(getattr(message, "content", ""))}
    raise NotImplementedError(f"Unsupported LangChain message type: {msg_type!r}")


def _convert_tools_to_openai(tools: Sequence[Any]) -> list[dict[str, Any]]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    converted = []
    for tool in tools:
        tool_dict = convert_to_openai_tool(tool)
        if tool_dict.get("type") == "function":
            fn = dict(tool_dict["function"])
            fn.pop("strict", None)
            converted.append({"type": "function", "function": fn})
    return converted


def _merge_metadata_chat_template_kwargs(
    metadata: Optional[dict[str, Any]],
    chat_template_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    merged_metadata = dict(metadata or {})
    if not chat_template_kwargs:
        return merged_metadata

    existing = merged_metadata.get("chat_template_kwargs", "{}")
    if isinstance(existing, str):
        existing_kwargs = json.loads(existing or "{}")
    elif isinstance(existing, dict):
        existing_kwargs = existing
    else:
        existing_kwargs = {}

    ctk = dict(chat_template_kwargs)
    ctk.update(existing_kwargs)
    merged_metadata["chat_template_kwargs"] = json.dumps(ctk)
    return merged_metadata


def _build_gym_chat_model(
    *,
    server_client: Any,
    model_server_name: str,
    model_name: str,
    request_params: dict[str, Any],
    cookies: Any,
    chat_template_kwargs: Optional[dict[str, Any]],
) -> Any:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel, LanguageModelInput
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool

    class GymChatModel(BaseChatModel):
        server_client: Any
        model_server_name: str
        model_name: str
        request_params: dict[str, Any] = Field(default_factory=dict)
        chat_template_kwargs: Optional[dict[str, Any]] = None
        shared_state: dict[str, Any] = Field(default_factory=dict)
        bound_tools: Sequence[dict[str, Any] | type | Any | BaseTool] = Field(default_factory=list)
        bound_tool_choice: Any = None

        model_config = ConfigDict(arbitrary_types_allowed=True)

        @property
        def _llm_type(self) -> str:
            return "nemo-gym-chat"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {"model_name": self.model_name, "model_server_name": self.model_server_name}

        def _get_ls_params(self, **kwargs: Any) -> dict[str, Any]:
            return {"ls_provider": "gym", "ls_model_name": self.model_name, "ls_model_type": "chat"}

        def bind_tools(
            self,
            tools: Sequence[dict[str, Any] | type | Any | BaseTool],
            *,
            tool_choice: str | None = None,
            **kwargs: Any,
        ) -> Runnable[LanguageModelInput, AIMessage]:
            return self.model_copy(update={"bound_tools": list(tools), "bound_tool_choice": tool_choice})

        def _chat_body(self, messages: Sequence[Any], stop: Optional[list[str]] = None) -> dict[str, Any]:
            body: dict[str, Any] = {
                "messages": [_langchain_message_to_chat_dict(message) for message in messages],
            }
            request_params = dict(self.request_params)

            if request_params.get("temperature") is not None:
                body["temperature"] = request_params["temperature"]
            if request_params.get("top_p") is not None:
                body["top_p"] = request_params["top_p"]
            if request_params.get("max_output_tokens") is not None:
                body["max_tokens"] = request_params["max_output_tokens"]
            if request_params.get("parallel_tool_calls") is not None:
                body["parallel_tool_calls"] = request_params["parallel_tool_calls"]
            if request_params.get("user") is not None:
                body["user"] = request_params["user"]
            if request_params.get("service_tier") is not None:
                body["service_tier"] = request_params["service_tier"]
            if stop:
                body["stop"] = stop

            body["metadata"] = _merge_metadata_chat_template_kwargs(
                request_params.get("metadata"), self.chat_template_kwargs
            )

            if self.bound_tools:
                body["tools"] = _convert_tools_to_openai(self.bound_tools)
                body["tool_choice"] = self.bound_tool_choice or "auto"

            return body

        def _accumulate_usage(self, completion: NeMoGymChatCompletion) -> None:
            usage = getattr(completion, "usage", None)
            if usage is None:
                return
            totals = self.shared_state.setdefault(
                "usage",
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            totals["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            totals["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

        async def _agenerate(
            self,
            messages: Sequence[Any],
            stop: Optional[list[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = await self.server_client.post(
                server_name=self.model_server_name,
                url_path="/v1/chat/completions",
                json=self._chat_body(messages, stop=stop),
                cookies=self.shared_state.get("cookies"),
            )
            await raise_for_status(response)
            self.shared_state["cookies"] = response.cookies
            completion = NeMoGymChatCompletion.model_validate(await get_response_json(response))
            self._accumulate_usage(completion)

            choice = completion.choices[0]
            message_dict = choice.message.model_dump()
            ai_message = _chat_message_to_langchain(message_dict)
            return ChatResult(
                generations=[ChatGeneration(message=ai_message)],
                llm_output={"model": completion.model, "usage": completion.usage.model_dump() if completion.usage else None},
            )

        def _generate(
            self,
            messages: Sequence[Any],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))

    return GymChatModel(
        server_client=server_client,
        model_server_name=model_server_name,
        model_name=model_name,
        request_params=request_params,
        chat_template_kwargs=chat_template_kwargs,
        shared_state={"cookies": cookies, "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
    )


def _build_backend(config: "DeepAgentsAgentConfig") -> Any:
    if config.backend == "state":
        from deepagents.backends import StateBackend

        return StateBackend()
    if config.backend == "local_shell":
        from deepagents.backends import LocalShellBackend

        env = None
        if not config.backend_inherit_env:
            env = {"PATH": os.environ.get("PATH", "")}
        return LocalShellBackend(
            root_dir=config.backend_root_dir,
            virtual_mode=config.backend_virtual_mode,
            timeout=config.execute_timeout,
            env=env,
            inherit_env=config.backend_inherit_env,
        )
    raise ValueError(f"Unsupported Deep Agents backend: {config.backend!r}")


class DeepAgentsAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 32
    max_turns: int = 30
    recursion_limit: Optional[int] = None
    temperature: float = 1.0
    system_prompt: Optional[str] = None
    backend: str = "state"
    backend_root_dir: Optional[str] = None
    backend_virtual_mode: bool = False
    backend_inherit_env: bool = False
    execute_timeout: int = 60
    skills: Optional[list[str]] = None
    memory: Optional[list[str]] = None
    chat_template_kwargs: Optional[dict[str, Any]] = Field(default_factory=lambda: dict(DEFAULT_CHAT_TEMPLATE_KWARGS))


class DeepAgentsRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class DeepAgentsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class DeepAgentsAgent(SimpleResponsesAPIAgent):
    config: DeepAgentsAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _resolve_model_name(self) -> str:
        return str(self.config.model_server.name)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        from deepagents import create_deep_agent

        _ensure_deepagents_harness_profile()

        body = body.model_copy(deep=True)
        initial_messages = _responses_input_to_langchain_messages(body)
        system_prompt, initial_messages = _split_system_prompt(initial_messages, self.config.system_prompt)

        request_params = body.model_dump(exclude_unset=True)
        if request_params.get("temperature") is None:
            request_params["temperature"] = self.config.temperature
        model = _build_gym_chat_model(
            server_client=self.server_client,
            model_server_name=self.config.model_server.name,
            model_name=self._resolve_model_name(),
            request_params=request_params,
            cookies=request.cookies,
            chat_template_kwargs=self.config.chat_template_kwargs,
        )

        agent = create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            backend=_build_backend(self.config),
            skills=self.config.skills,
            memory=self.config.memory,
            name=self.config.name or "deepagents_agent",
        )

        recursion_limit = self.config.recursion_limit or max(25, self.config.max_turns * 4)
        final_state = await agent.ainvoke(
            {"messages": initial_messages},
            config={"recursion_limit": recursion_limit},
        )
        final_messages = final_state.get("messages", [])
        new_messages = final_messages[len(initial_messages) :]
        output_items = _messages_to_output_items(new_messages)

        if not any(getattr(item, "type", None) == "message" for item in output_items):
            LOG.warning("Deep Agents ended without an assistant message. Padding empty assistant message.")
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=[0],
                    generation_token_ids=[0],
                    generation_log_probs=[0.0],
                )
            )

        shared_state = getattr(model, "shared_state", {})
        cookies = shared_state.get("cookies")
        if cookies:
            for k, v in cookies.items():
                response.set_cookie(k, getattr(v, "value", v))

        usage_totals = shared_state.get("usage") or {}
        usage = NeMoGymResponseUsage(
            input_tokens=usage_totals.get("input_tokens", 0),
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=usage_totals.get("output_tokens", 0),
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=usage_totals.get("total_tokens", 0),
        )

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self._resolve_model_name(),
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

    async def run(self, request: Request, body: DeepAgentsRunRequest) -> DeepAgentsVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return DeepAgentsVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    DeepAgentsAgent.run_webserver()
