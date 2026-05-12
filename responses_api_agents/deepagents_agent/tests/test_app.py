# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.deepagents_agent.app import (
    DeepAgentsAgent,
    DeepAgentsAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
    _build_gym_chat_model,
    _merge_metadata_chat_template_kwargs,
    _messages_to_output_items,
    _split_system_prompt,
)


def _config(**kwargs) -> DeepAgentsAgentConfig:
    defaults = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "",
        "resources_server": ResourcesServerRef(type="resources_servers", name=""),
        "model_server": ModelServerRef(type="responses_api_models", name=""),
    }
    return DeepAgentsAgentConfig(**(defaults | kwargs))


class TestSanity:
    def test_construct(self) -> None:
        DeepAgentsAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_concurrency_semaphore_initialized(self) -> None:
        agent = DeepAgentsAgent(config=_config(concurrency=4), server_client=MagicMock(spec=ServerClient))
        assert agent.sem._value == 4


class TestMessagesToOutputItems:
    def test_ai_message_with_tokens_and_tool_call(self) -> None:
        messages = pytest.importorskip("langchain_core.messages")
        ai = messages.AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"/tmp/a"}'},
                    }
                ]
            },
            response_metadata={
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [-0.1, -0.2],
            },
        )

        out = _messages_to_output_items([ai])
        assert len(out) == 2
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert out[0].prompt_token_ids == [1, 2]
        assert out[0].generation_token_ids == [3, 4]
        assert isinstance(out[1], NeMoGymResponseFunctionToolCall)
        assert out[1].name == "read_file"
        assert out[1].arguments == '{"path":"/tmp/a"}'

    def test_tool_message(self) -> None:
        messages = pytest.importorskip("langchain_core.messages")
        tool = messages.ToolMessage(content="file contents", tool_call_id="call_1")

        out = _messages_to_output_items([tool])
        assert len(out) == 1
        assert isinstance(out[0], NeMoGymFunctionCallOutput)
        assert out[0].call_id == "call_1"
        assert out[0].output == "file contents"


class TestSystemPrompt:
    def test_input_system_message_is_promoted(self) -> None:
        messages = pytest.importorskip("langchain_core.messages")
        system = messages.SystemMessage(content="Use the policy.")
        human = messages.HumanMessage(content="hello")

        system_prompt, history = _split_system_prompt([system, human], configured_system_prompt=None)

        assert system_prompt == "Use the policy."
        assert history == [human]

    def test_config_system_prompt_overrides_input_system_message(self) -> None:
        messages = pytest.importorskip("langchain_core.messages")
        system = messages.SystemMessage(content="Use the policy.")
        human = messages.HumanMessage(content="hello")

        system_prompt, history = _split_system_prompt([system, human], configured_system_prompt="Configured.")

        assert system_prompt == "Configured."
        assert history == [human]


class TestGymChatModel:
    def test_chat_body_uses_deepagents_tools_only(self) -> None:
        messages = pytest.importorskip("langchain_core.messages")

        def read_file(path: str) -> str:
            """Read a file."""
            return path

        model = _build_gym_chat_model(
            server_client=MagicMock(),
            model_server_name="policy_model",
            model_name="policy_model",
            request_params={
                "temperature": 0.5,
                "tools": [{"type": "function", "name": "dataset_tool"}],
                "parallel_tool_calls": False,
            },
            cookies={},
            chat_template_kwargs={"enable_thinking": True},
        ).bind_tools([read_file])

        body = model._chat_body([messages.HumanMessage(content="hello")])

        assert body["temperature"] == 0.5
        assert body["parallel_tool_calls"] is False
        assert [tool["function"]["name"] for tool in body["tools"]] == ["read_file"]

    def test_chat_template_kwargs_are_serialized_into_metadata(self) -> None:
        merged = _merge_metadata_chat_template_kwargs(
            {"chat_template_kwargs": '{"enable_thinking": false}'},
            {"enable_thinking": True, "truncate_history_thinking": False},
        )

        assert merged["chat_template_kwargs"] == '{"enable_thinking": false, "truncate_history_thinking": false}'
