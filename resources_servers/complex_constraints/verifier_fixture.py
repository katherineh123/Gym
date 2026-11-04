# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free ComplexConstraints checks using canned judge replies and a Gym-authored test prompt template."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymChatCompletion


SERVER_DIR = Path(__file__).resolve().parent


def create_fixture_server():
    """Build the server from its YAML with the test prompt template and no live judge."""
    from resources_servers.complex_constraints.app import (
        ComplexConstraintsResourcesServer,
        ComplexConstraintsResourcesServerConfig,
    )

    raw = yaml.safe_load((SERVER_DIR / "configs" / "complex_constraints.yaml").read_text(encoding="utf-8"))
    server_config = raw["complex_constraints_resources_server"]["resources_servers"]["complex_constraints"]
    config = ComplexConstraintsResourcesServerConfig.model_validate(
        {
            **server_config,
            "host": "127.0.0.1",
            "port": 0,
            "name": "complex_constraints_resources_server",
            "judge_prompt_fpath": str(SERVER_DIR / "tests" / "judge_prompt.json"),
        }
    )
    return ComplexConstraintsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


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


def scores_by_criterion(rubrics, scores):
    """A judge stand-in that gives each criterion its score, whatever order the per-criterion calls arrive in."""

    async def post(**kwargs):
        prompt = kwargs["json"].messages[0]["content"]
        # The longest match, so a criterion that contains another one still gets its own score.
        _, score = max(((c, s) for c, s in zip(rubrics, scores) if c in prompt), key=lambda pair: len(pair[0]))
        return judge_reply(json.dumps({"rationale": "fixture", "score": score}))

    return post


async def invoke_fixture(server, request):
    """Answer each criterion's judge call with the case's canned `fixture_judge_scores` entry."""
    scores = getattr(request, "fixture_judge_scores", None) or []
    server.server_client.post = AsyncMock(side_effect=scores_by_criterion(request.rubrics, scores))
    return await server.verify(request)
