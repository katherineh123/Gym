# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared native OpenCode execution, independent of the Gym server lifecycle."""

import asyncio
import json
import logging
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from time import time
from uuid import uuid4

from pydantic import JsonValue

from environment_servers.agentic_if.bootstrap import prepare_binary
from environment_servers.agentic_if.config import OpenCodePermission
from environment_servers.agentic_if.drivers.opencode_context import actor_export
from environment_servers.agentic_if.drivers.opencode_evidence import opencode_evidence
from environment_servers.agentic_if.drivers.opencode_export import export_output
from environment_servers.agentic_if.drivers.opencode_hooks import stage_hooks
from environment_servers.agentic_if.drivers.opencode_observations import bind_observations, project_output
from environment_servers.agentic_if.drivers.opencode_replay import prepare_continuation
from environment_servers.agentic_if.drivers.opencode_sessions import collect_sessions
from environment_servers.agentic_if.runtime import RunContext, RunOutcome
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.agentic_if.task_variants.builder import content_text
from resources_servers.agentic_if.task_variants.catalog import NATIVE_TOOLS
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces


LOG = logging.getLogger(__name__)

# Additional built-ins in the pinned OpenCode registry, beyond our selectable
# catalog. An explicit selection must exclude these too; it is not permission to
# enable every other built-in. Unknown future tools fail the gateway inventory
# check rather than silently becoming usable.
OPENCODE_EXTRA_TOOLS = frozenset({"webfetch", "websearch", "skill", "question", "lsp", "plan_enter", "plan_exit"})


@dataclass(frozen=True, kw_only=True)
class OpenCodeDriver:
    """Run the native executable and export every session in its private data store.

    ``binary_path`` names a host copy to upload; None installs ``version`` from npm inside the task sandbox. Either
    way the binary must match ``binary_sha256``.
    """

    version: str
    binary_sha256: str
    context_limit: int
    max_turns: int | None
    binary_path: Path | None = None
    permission: OpenCodePermission | None = None

    async def execute(self, context: RunContext) -> RunOutcome:
        """Stage the pinned binary, run OpenCode once on the task prompt, and collect its sessions as evidence."""
        sandbox = await prepare_binary(
            context.sandbox, version=self.version, sha256=self.binary_sha256, binary_path=self.binary_path
        )
        env = {"XDG_DATA_HOME": sandbox.directory + "/data", "XDG_STATE_HOME": sandbox.directory + "/state"}
        version = await sandbox.exec(command="opencode --version", env=env, timeout_s=30)
        if version.return_code != 0 or version.error_type or version.stdout.strip() != self.version:
            raise ValueError("staged OpenCode version does not match its configured pin")
        config = self._configuration(context)
        hooks = await stage_hooks(context, sandbox)
        config["plugin"] = hooks.plugins
        messages = context.params.model_dump(mode="json")["input"]
        instruction = "\n\n".join(content_text(item["content"]) for item in messages if item.get("role") == "user")
        remote_prompt = sandbox.directory + "/prompt.txt"
        with tempfile.TemporaryDirectory(prefix="if-opencode-prompt-") as temporary:
            path = Path(temporary) / "prompt.txt"
            path.write_text(instruction)
            await sandbox.upload(path, remote_prompt)
        mode = await context.sandbox.transport.exec(
            command=f"chmod 0644 {quote(remote_prompt)}", user="root", timeout_s=30
        )
        if mode.return_code != 0 or mode.error_type:
            raise RuntimeError("cannot stage native OpenCode prompt")
        status, result, continuation = "error", None, None
        try:
            async with asyncio.timeout(context.sandbox.timeout_s):
                session_args = ""
                if context.spec.continuation:
                    continuation = await prepare_continuation(context, sandbox, env)
                    session_args = f"--session {quote(continuation.session_id)} --model {quote(config['model'])} "
                result = await sandbox.exec(
                    command=f"opencode run --title 'NeMo Gym IF' {session_args}--thinking < {quote(remote_prompt)}",
                    env=env
                    | hooks.environment()
                    | ({"NEMO_GYM_IF_PREFIX_SESSION": continuation.session_id} if continuation is not None else {})
                    | {
                        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                        "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "1000000000",
                    },
                )
            if result.return_code == 0 and not result.error_type:
                status = "completed"
        except TimeoutError:
            status = "timeout"
        context.artifacts_dir.mkdir(parents=True, exist_ok=True)
        exports, sessions, artifacts = {}, [], {}
        try:
            exports, sessions, artifacts = await collect_sessions(context, sandbox, env)
        except Exception:
            LOG.exception("OpenCode session collection failed in run %s", context.run_id)
        applied, receipt_error = None, None
        try:
            exports, applied = await hooks.enrich_exports(context, exports, artifacts)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            receipt_error = f"hook_execution_receipts:{error}"
        roots = [session.id for session in sessions if session.parent_id is None]
        root = exports.get(roots[0]) if len(roots) == 1 else None
        # The root is owned by this one non-interactive `opencode run` invocation:
        # it receives exactly the staged prompt, with no other live user ingress.
        # Delegated/frozen sessions do not gain this guarantee by resemblance.
        controlled = frozenset(
            (roots[0], message["info"]["id"])
            for message in (root or {}).get("messages", [])
            if continuation is None or message["info"]["id"] not in continuation.prefix_ids
        )
        evidence = opencode_evidence(
            root,
            child_exports=[value for key, value in exports.items() if key not in roots],
            expected_sessions=sessions,
            tool_names=context.spec.tools.tool_names,
            interfaces=installed_interfaces(context.spec.tools.tool_interfaces),
            artifacts=artifacts,
            controlled_messages=controlled,
        )
        evidence = bind_observations(
            evidence,
            context.model_observations,
            prefix_messages=frozenset((continuation.session_id, identity) for identity in continuation.prefix_ids)
            if continuation is not None
            else frozenset(),
        )
        evidence = await hooks.collect(context, evidence, applied=applied)
        if receipt_error is not None:
            evidence.status = "incomplete"
            evidence.issues.append(receipt_error)
        if continuation is not None:
            evidence = continuation.attach(evidence)
        artifacts = evidence.artifacts
        projected = {}
        for session_id, document in exports.items():
            document = actor_export(
                document,
                controlled_ids=frozenset(identity for session, identity in controlled if session == session_id),
            )
            if continuation is not None and session_id == continuation.session_id:
                document = document | {
                    "messages": [
                        item for item in document["messages"] if item["info"]["id"] not in continuation.prefix_ids
                    ]
                }
            projected[session_id], issues = project_output(
                export_output(document),
                context.model_observations,
                session_id=session_id,
                compaction_messages=frozenset(
                    (record["session_id"], record["message_id"])
                    for record in evidence.facts.get("observed_native_compactions", [])
                ),
            )
            if issues:
                if evidence.status == "complete":
                    evidence.status = "incomplete"
                evidence.issues.extend(f"training_capture:{session_id}:{issue}" for issue in issues)
            observed_outputs = {
                item.call_id: item.output
                for item in projected[session_id]
                if getattr(item, "type", None) == "function_call_output"
            }
            for step in evidence.steps:
                if step.session_id == session_id and step.role == "tool" and step.call_id in observed_outputs:
                    step.output = observed_outputs[step.call_id]
        if status == "completed":
            assistants = [
                item.get("info", {})
                for item in (root or {}).get("messages", [])
                if item.get("info", {}).get("role") == "assistant"
            ]
            if not assistants or assistants[-1].get("error") or evidence.status != "complete":
                status = "error"
            elif assistants[-1].get("finish") == "length":
                status = "length"
        output = projected.get(roots[0], []) if len(roots) == 1 else []
        # The export's first user is the initial task, already present in the row.
        if output and getattr(output[0], "role", None) == "user":
            output = output[1:]
        response = NeMoGymResponse(
            id="resp_" + uuid4().hex,
            created_at=int(time()),
            object="response",
            model=context.model.model,
            output=output,
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        return RunOutcome(response=response, evidence=evidence, status=status, artifacts=artifacts)

    def _configuration(self, context: RunContext) -> dict[str, JsonValue]:
        permission = deepcopy(self.permission) if self.permission is not None else {}
        selected = context.spec.tools.tool_names
        if selected is not None:
            disabled = (NATIVE_TOOLS | OPENCODE_EXTRA_TOOLS) - selected.keys()
            # OpenCode checks write/apply_patch under edit. Keep that permission
            # if any selected editing implementation needs it. Its model-native
            # registry decides edit/write versus apply_patch; the gateway checks
            # that the resulting inventory actually matches the requested one.
            if {"edit", "write", "apply_patch"} & selected.keys():
                disabled -= {"edit"}
            for tool in sorted(disabled):
                # Last matching native rule wins, including wildcard rules.
                # Append explicit denies without relaxing any selected-tool or
                # non-tool deployment restriction (external_directory, etc.).
                permission.pop(tool, None)
                permission[tool] = "deny"
        return {
            "$schema": "https://opencode.ai/config.json",
            **({"permission": permission} if permission or self.permission is not None else {}),
            "model": "nemo_gym/dummy_model",
            "provider": {
                "nemo_gym": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": context.model.base_url,
                        "apiKey": context.model.api_key,
                        "chunkTimeout": int(context.sandbox.timeout_s * 1000),
                        "timeout": False,
                    },
                    "models": {
                        "dummy_model": {
                            "temperature": True,
                            "limit": {
                                "context": self.context_limit,
                                "input": self.context_limit,
                                "output": self.context_limit,
                            },
                        }
                    },
                }
            },
            "agent": {
                "build": {
                    **({"steps": self.max_turns} if self.max_turns is not None else {}),
                    **{
                        name: value
                        for name in ("temperature", "top_p")
                        if (value := getattr(context.params, name, None)) is not None
                    },
                }
            },
        }
