# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free AdvancedIF checks using canned judge replies and Gym-authored test prompt templates."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymChatCompletion


SERVER_DIR = Path(__file__).resolve().parent


def create_fixture_server():
    """Build the server from its YAML with the test prompt templates and no live judge."""
    from resources_servers.advanced_if.app import AdvancedIfResourcesServer, AdvancedIfResourcesServerConfig

    raw = yaml.safe_load((SERVER_DIR / "configs" / "advanced_if.yaml").read_text(encoding="utf-8"))
    server_config = raw["advanced_if_resources_server"]["resources_servers"]["advanced_if"]
    config = AdvancedIfResourcesServerConfig.model_validate(
        {
            **server_config,
            "host": "127.0.0.1",
            "port": 0,
            "name": "advanced_if_resources_server",
            "judge_prompts_fpath": str(SERVER_DIR / "tests" / "judge_prompts.json"),
        }
    )
    return AdvancedIfResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def judge_reply(content):
    """An HTTP reply that carries a Chat Completions judge answer with `content`."""
    completion = NeMoGymChatCompletion.model_validate(
        {
            "id": "fixture_judge",
            "created": 0,
            "model": "fixture",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }
    )
    reply = MagicMock(ok=True)
    reply.read = AsyncMock(return_value=completion.model_dump_json())
    return reply


async def invoke_fixture(server, request):
    """Answer every judge call with the case's canned `fixture_judge_text`."""
    server.server_client.post = AsyncMock(return_value=judge_reply(getattr(request, "fixture_judge_text", None)))
    return await server.verify(request)
