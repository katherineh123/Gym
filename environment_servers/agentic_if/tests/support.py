# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared builders and doubles for these tests.

Native OpenCode exports and model observations, the model gateway and OpenCode driver doubles, sandbox transports,
and sealed task rows for the environment lifecycle.
"""

import asyncio
import hashlib
import json
import os
import signal
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from environment_servers.agentic_if import app as module
from environment_servers.agentic_if.drivers.opencode import OpenCodeDriver
from environment_servers.agentic_if.model_gateway import ModelGateway
from environment_servers.agentic_if.runtime import BoundSandbox, ModelRoute, RunContext
from nemo_gym.base_resources_server import ResourcesSeedSessionResponse
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, seal_runtime_row


def message(identity, role, *parts, session="root", created=1):
    return {
        "info": {"id": identity, "role": role, "sessionID": session, "time": {"created": created}},
        "parts": list(parts),
    }


def text(value):
    return {"type": "text", "text": value}


def call(identity, tool="read", *, status="completed", end=3, **state):
    return {
        "type": "tool",
        "id": "part-" + identity,
        "callID": identity,
        "tool": tool,
        "state": {
            "status": status,
            "input": {"filePath": "/app/a.py"},
            "output": "contents",
            "time": {"start": 2, "end": end},
            **state,
        },
    }


def export(*messages, session="root", parent=None):
    return {"info": {"id": session, "parentID": parent}, "messages": list(messages)}


def observation(session="root", identity="a", *, tool="RunCommand", arguments='{ "command": "pwd" }'):
    return {
        "session_id": session,
        "assistant_message_id": identity,
        "execution_tools": {tool: "bash"},
        "request": {"messages": [{"role": "system", "content": "Explain each command."}]},
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Inspect",
                        "tool_calls": [{"id": "c1", "function": {"name": tool, "arguments": arguments}}],
                    }
                }
            ]
        },
    }


def native(session="root", parent=None):
    return export(
        message("u", "user", text("Fix it"), session=session),
        message(
            "a",
            "assistant",
            text("Inspect"),
            call("c1", "bash", input={"command": "pwd"}, output="/app", end=4),
            session=session,
            created=2,
        ),
        session=session,
        parent=parent,
    )


def tool_observation(session="root", value="/app"):
    """The next actor request: it carries the tool result, and its reply ends the session."""
    return {
        "source": "model",
        "session_id": session,
        "assistant_message_id": "z",
        "execution_tools": {},
        "request": {"messages": [{"role": "tool", "tool_call_id": "c1", "content": value}]},
        "response": {"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
    }


def params(system=None):
    messages = [{"role": "user", "content": "Task"}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    return NeMoGymResponseCreateParamsNonStreaming(input=messages)


def definition(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Native",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


class ModelServer:
    def __init__(self, result, *, stream=False):
        self.result, self.stream = result, stream
        self.requests = []
        self.released = 0

    async def post(self, **request):
        self.requests.append(deepcopy(request))

        async def json_body():
            return deepcopy(self.result)

        async def chunks():
            wire = "".join("data: " + json.dumps(event) + "\r\n\r\n" for event in self.result)
            if request["url_path"].endswith("completions"):
                wire += "data: [DONE]\r\n\r\n"
            data = wire.encode()
            for start in range(0, len(data), 7):
                yield data[start : start + 7]

        def release():
            self.released += 1

        return SimpleNamespace(ok=True, json=json_body, release=release, content=SimpleNamespace(iter_any=chunks))


def gateway(server, *, tools=None):
    return ModelGateway(
        spec=RuntimeSpec(harness="opencode", task_backend="swe", tools=tools or {}),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Fix it"}]),
        client=server,
        model_server="actor",
        run_id="attempt-a",
        timeout_s=30,
    )


REASONING = "private-reasoning-sentinel"  # Distinct from the /private prefix of macOS temporary paths.


def exported(session, parent=None, text="Done"):
    return {
        "info": {"id": session, "parentID": parent},
        "messages": [
            {
                "info": {"id": "u", "role": "user", "sessionID": session, "time": {"created": 1}},
                "parts": [{"type": "text", "text": "Fix"}],
            },
            {
                "info": {
                    "id": "a",
                    "role": "assistant",
                    "sessionID": session,
                    "finish": "stop",
                    "time": {"created": 2, "completed": 3},
                },
                "parts": [{"type": "reasoning", "text": REASONING}, {"type": "text", "text": text}],
            },
        ],
    }


class Provider:
    def __init__(self, *, child=True, missing=False):
        self.calls, self.files = [], {}
        self.child, self.missing = child, missing
        self.observations = [
            {
                "session_id": name,
                "assistant_message_id": "a",
                "request": {"messages": []},
                "execution_tools": {},
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "Done", "reasoning_content": REASONING}}]
                },
            }
            for name in (["root", "child"] if child else ["root"])
        ]

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_bytes()

    async def download(self, source, destination):
        if source.endswith("hook-applied.jsonl"):
            config = json.loads(next(value for name, value in self.files.items() if name.endswith("hook-config.json")))
            Path(destination).write_text(
                json.dumps({"kind": "loaded", "profile_sha256": config["profile_sha256"]}) + "\n"
            )
            return
        # Driver downloads each explicitly exported native session separately.
        session = "child" if "child" in source else "root"
        if session == "child" and self.missing:
            raise FileNotFoundError(source)
        Path(destination).write_text(json.dumps(exported(session, "root" if session == "child" else None)))

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        command = kwargs["command"]
        stdout = ""
        if "--version" in command:
            stdout = "1.17.11"
        elif "opencode db " in command:
            sessions = [{"id": "root", "parent_id": None}]
            if self.child:
                sessions.append({"id": "child", "parent_id": "root"})
            stdout = json.dumps(sessions)
        return SandboxExecResult(stdout=stdout, stderr="", return_code=0)


def context(tmp_path, provider):
    return RunContext(
        run_id="oc-test",
        spec=RuntimeSpec(harness="opencode", task_backend="swe"),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Fix 'quoted' {literal}"}]),
        sandbox=BoundSandbox(provider, user="agent", workdir="/app/repo", timeout_s=60),
        model=ModelRoute(base_url="http://gym/attempt/oc/v1", model="policy", api_key="attempt-token"),
        artifacts_dir=tmp_path / "artifacts",
        model_observations=getattr(provider, "observations", []),
    )


def driver(tmp_path):
    path = tmp_path / "opencode"
    path.write_bytes(b"pinned native executable")
    return OpenCodeDriver(
        version="1.17.11",
        binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        context_limit=32000,
        max_turns=5,
        binary_path=path,
    )


class LocalTransport:
    def __init__(self):
        self.calls = []

    async def exec(self, *, command, user=None, cwd=None, timeout_s=None, env=None):
        self.calls.append((command, user))
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            cwd=cwd,
            env=os.environ | (env or {}),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communication = asyncio.create_task(proc.communicate())
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout_s)
            code = proc.returncode
        except TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = await communication
            code = 124
        return SandboxExecResult(stdout=stdout.decode(), stderr=stderr.decode(), return_code=code)

    async def upload(self, source, destination):
        Path(destination).write_bytes(Path(source).read_bytes())


class Transport:
    def __init__(self, *, uid="1001", gid="1001"):
        self.calls = []
        self.uid, self.gid = uid, gid
        self.stops = self.disconnects = 0

    async def exec(self, command=None, **kwargs):
        self.calls.append({"command": command, **kwargs})
        output = ""
        if command == "id -u":
            output = "0\n"
        elif command == "id -u && id -g":
            output = f"{self.uid}\n{self.gid}\n"
        elif "getent passwd" in command:
            output = "/home/agent\n/bin/bash\n"
        return SandboxExecResult(stdout=output, stderr="", return_code=0)

    async def stop(self):
        self.stops += 1

    async def disconnect(self):
        self.disconnects += 1


def seeded(workdir="/app/project", *, session_id="attempt-a", sandbox_id="sandbox-a"):
    """The typed seed reply of a resources server that owns the sandbox."""
    return ResourcesSeedSessionResponse(
        resources_session_id=session_id,
        sandbox_access=SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"sandbox_id": sandbox_id}),
            workdir=workdir,
        ),
    ).model_dump(mode="json")


def response():
    return NeMoGymResponse(
        id="actual-response",
        created_at=0,
        model="actor",
        object="response",
        output=[],
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
    )


def task_row(*, check=True, text="Fix it"):
    return seal_runtime_row(
        {"instance_id": "task-a", "responses_create_params": {"input": [{"role": "user", "content": text}]}},
        RuntimeSpec(
            harness="opencode",
            task_backend="swe",
            checks=[
                {
                    "id": "literal-rubric",
                    "kind": "judge",
                    "rubric": "Finish with DONE.",
                }
            ]
            if check
            else [],
        ),
    )


def episode(row, *, rollout_id=None, attempt=0):
    return module.AgenticIFRequest.model_validate(
        {
            "episode_id": {"rollout_id": rollout_id or uuid4().hex, "attempt": attempt},
            "task": {
                "task_id": {"taskset": "agentic_if:tests", "task_id": digest(row)},
                "task_input": row,
            },
        }
    )


async def run_episode(environment, row, *, failed=False):
    response = await environment.run_request(episode(row))
    if failed:
        assert response.result is None
        assert response.failure is not None
        return response.failure.partial_response
    assert response.failure is None, response.failure
    return response.result
