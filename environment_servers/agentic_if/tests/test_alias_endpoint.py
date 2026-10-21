# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from environment_servers.agentic_if.app import AgenticIFEnvironment


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,code", [("valid", None), ("wrong_token", 404), ("expired", 404), ("unobserved", 409), ("extra_field", 422)]
)
async def test_alias_lookup_is_scoped_to_the_authenticated_active_attempt(case, code):
    called = []

    def lookup(**kwargs):
        called.append(kwargs)
        if case == "unobserved":
            raise ValueError("unobserved")
        return {"allowed": False, "model_name": "read", "available_names": ["ReadSource"]}

    body = {"session_id": "root", "call_id": "one", "native_tool": "read"}
    if case == "extra_field":
        body["allow"] = True

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", b"Bearer wrong" if case == "wrong_token" else b"Bearer attempt-key")],
        },
        receive,
    )
    attempt = SimpleNamespace(
        context=SimpleNamespace(model=SimpleNamespace(api_key="attempt-key")),
        gateway=SimpleNamespace(tool_call=lookup),
    )
    owner = SimpleNamespace(_attempts={} if case == "expired" else {"attempt": attempt})
    owner._attempt = lambda request, identity: AgenticIFEnvironment._attempt(owner, request, identity)
    if code is not None:
        with pytest.raises(HTTPException) as error:
            await AgenticIFEnvironment.tool_call(owner, request, "attempt")
        assert error.value.status_code == code
    else:
        assert await AgenticIFEnvironment.tool_call(owner, request, "attempt") == {
            "allowed": False,
            "model_name": "read",
            "available_names": ["ReadSource"],
        }
    assert bool(called) == (case in {"valid", "unobserved"})
