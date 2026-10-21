# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile the canonical contract into the one installed native hook pipeline."""

import asyncio
import json
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from pydantic import JsonValue

from environment_servers.agentic_if.bootstrap import BinarySandbox
from environment_servers.agentic_if.drivers.opencode_observations import preserve_rejected_calls
from environment_servers.agentic_if.drivers.opencode_profile import profile_sha256
from environment_servers.agentic_if.runtime import RunContext
from resources_servers.agentic_if.if_grading.contracts import EventReceipt, SessionEvidence
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec, ToolEvent
from responses_api_agents.opencode_agent.runtime import OBSERVABILITY_PATCH


def prefix_receipts(continuation: ContinuationSpec | None) -> list[dict[str, JsonValue]]:
    """Retain frozen error metadata that stock OpenCode strips on import.

    Continuations currently contain one root only. Consumers bind these calls
    to the actual resumed root session, never to child or fresh live calls.
    """
    if continuation is None:
        return []
    return [
        {"call_id": part["callID"], "native_tool": part["tool"], "metadata": deepcopy(stamp)}
        for message in continuation.prefix_export["messages"]
        for part in message.get("parts", [])
        if part.get("type") == "tool" and part.get("state", {}).get("status") == "error"
        if (stamp := (part["state"].get("metadata") or {}).get("nemo_gym_profile"))
    ]


def merge_hook_receipts(exports: dict[str, dict], records: list[dict[str, JsonValue]]) -> dict[str, dict]:
    """Enrich a copy of native exports using exact invocation identities.

    The native status/result remains authoritative. Sidecar metadata restores
    facts that stock OpenCode discards for failed calls; it never changes an
    error into a success. Raw exports are retained separately for auditing.
    """
    result = deepcopy(exports)
    parts = {}
    for session, document in result.items():
        for message in document.get("messages", []):
            for part in message.get("parts", []):
                if part.get("type") != "tool":
                    continue
                identity = (session, part.get("callID"))
                if identity in parts:
                    raise ValueError("ambiguous exported invocation")
                parts[identity] = part
    seen = set()
    for record in records:
        if record.get("kind") not in {"tool_receipt", "alias_rejected"}:
            continue
        identity = (record.get("session_id"), record.get("call_id"))
        part = parts.get(identity)
        if identity in seen or part is None or part.get("tool") != record.get("native_tool"):
            raise ValueError("unbound or duplicate hook execution receipt")
        seen.add(identity)
        state = part["state"]
        if record["kind"] == "alias_rejected":
            if state.get("status") != "error":
                raise ValueError("alias rejection disagrees with native execution")
            stamp = {"dispatch_rejected": True, "model_name": record["model_name"], "error": record["error"]}
        else:
            stamp = record["metadata"]
        state.setdefault("metadata", {})["nemo_gym_profile"] = deepcopy(stamp)
        if "configured_args" in stamp:
            state["input"] = deepcopy(stamp["configured_args"])
    presented = {}
    for record in records:
        if record.get("kind") != "tool_presentation":
            continue
        identity = (record.get("session_id"), record.get("call_id"))
        part = parts.get(identity)
        state = part.get("state", {}) if part else {}
        if (
            (identity in presented and presented[identity] != record.get("model_error"))
            or part is None
            or part.get("tool") != record.get("native_tool")
            or state.get("status") != "error"
            or not isinstance(record.get("native_error"), str)
            or record["native_error"] != state.get("error")
            or not isinstance(record.get("model_error"), str)
        ):
            raise ValueError("unbound or duplicate hook presentation receipt")
        stamp = state.setdefault("metadata", {}).setdefault("nemo_gym_profile", {})
        if "model_error" in stamp and stamp["model_error"] != record["model_error"]:
            raise ValueError("hook presentation disagrees with execution receipt")
        stamp["model_error"] = record["model_error"]
        # Replaying history may present the same error in another native
        # process. Identical receipts are idempotent; conflicting views fail.
        presented[identity] = record["model_error"]
    return result


def compile_hooks(spec: RuntimeSpec) -> dict[str, JsonValue]:
    """Resolve trusted implementations; prompt/description text stays literal.

    This is internal plugin configuration, not a second accepted dataset shape.
    Native execution sees native tool IDs. The common model boundary owns aliases.
    ``stage_hooks`` adds the continuation receipts and the profile hash.
    """
    definitions = installed_interfaces(spec.tools.tool_interfaces)
    for definition in definitions.values():
        # Execution schemas/maps belong inside the native harness. Presentation
        # is applied once at the model boundary, including catalog descriptions.
        definition.pop("description", None)
    config = {
        "profile_version": 1,
        "id": "agentic-if",
        "tools": definitions,
    }
    if spec.tools.read_first is not None:
        config["read_first"] = spec.tools.read_first.model_dump(mode="json")
    if spec.tools.tool_names is not None:
        # Diagnostic labels only: the gateway still owns tool presentation and
        # the native harness still owns dispatch, validation and recovery.
        config["visible_tools"] = dict(spec.tools.tool_names)
        config["check_aliases"] = any(native != alias for native, alias in config["visible_tools"].items())
    config["literal_errors"] = [event.text for event in spec.events]
    return config


def attach_events(
    evidence: SessionEvidence, events: list[ToolEvent], records: list[dict[str, JsonValue]]
) -> SessionEvidence:
    """Bind actual plant firings to captured tool results; never infer a firing."""
    result = evidence.model_copy(deep=True)
    issues = []
    if not any(row.get("kind") == "loaded" and row.get("plants") == len(events) for row in records):
        issues.append("missing_event_load_receipt")
    declared = {event.id: event for event in events}
    seen = {event.id for event in result.events}
    for record in records:
        if record.get("kind") == "error":
            issues.append("event_load_error")
        if record.get("kind") != "fired":
            continue
        identity = record.get("event_id")
        event = declared.get(identity) if isinstance(identity, str) else None
        if event is None or identity in seen:
            issues.append("unknown_or_duplicate_event")
            continue
        seen.add(identity)
        matches = [
            index
            for index, step in enumerate(result.steps)
            if step.role == "tool"
            and step.session_id == record.get("session_id")
            and step.call_id == record.get("call_id")
            and step.tool == event.match.tool
        ]
        if len(matches) != 1 or record.get("action") != event.action or record.get("tool") != event.match.tool:
            issues.append(f"uncorrelated_event:{identity}")
            continue
        result.events.append(
            EventReceipt(
                id=identity,
                step_index=matches[0],
                call_id=record["call_id"],
                invocation_id=result.steps[matches[0]].invocation_id,
                details=record,
            )
        )
        result.steps[matches[0]].planted = True
    if issues:
        result.status = "incomplete"
        result.issues.extend(issues)
    return result


@dataclass(frozen=True)
class StagedHooks:
    """Attempt-private plugin artifacts for the live OpenCode run."""

    directory: str
    configuration_sha256: str
    alias_gate: tuple[str, str] | None = None

    @property
    def plugins(self) -> list[str]:
        """The ``plugin`` entries of the OpenCode configuration."""
        return ["file://" + self.directory + "/" + name for name in ("if-profile.mjs", "if-request-kind.mjs")]

    def environment(self) -> dict[str, str]:
        """The variables that point the plugins at their staged configuration and receipt files."""
        return {
            "NEMO_GYM_OPENCODE_PROFILE": self.directory + "/hook-config.json",
            "NEMO_GYM_OPENCODE_PROFILE_APPLIED": self.directory + "/hook-applied.jsonl",
            "NEMO_GYM_OPENCODE_PLANTS": self.directory + "/hook-events.json",
            "NEMO_GYM_OPENCODE_PLANTS_FIRED": self.directory + "/hook-events-fired.jsonl",
            **(
                {"NEMO_GYM_IF_TOOL_CALL_URL": self.alias_gate[0], "NEMO_GYM_IF_TOOL_CALL_TOKEN": self.alias_gate[1]}
                if self.alias_gate is not None
                else {}
            ),
        }

    async def collect(
        self, context: RunContext, evidence: SessionEvidence, *, applied: list[dict[str, JsonValue]] | None = None
    ) -> SessionEvidence:
        """Missing/malformed receipts make evidence incomplete, never a silent no-op."""
        result = evidence.model_copy(deep=True)
        try:
            if applied is None:
                applied = await self.records(context)
            result.artifacts["hook_applied"] = str(context.artifacts_dir / "hook-applied.jsonl")
            if not any(
                row.get("kind") == "loaded" and row.get("profile_sha256") == self.configuration_sha256
                for row in applied
            ):
                raise ValueError("missing live hook load receipt")
            if context.spec.events:
                records = await self._records(context, "hook-events-fired.jsonl")
                result = attach_events(result, context.spec.events, records)
                result.artifacts["hook_events"] = str(context.artifacts_dir / "hook-events-fired.jsonl")
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            result.status = "incomplete"
            result.issues.append(f"hook_receipt_error:{error}")
        return result

    async def records(self, context: RunContext) -> list[dict[str, JsonValue]]:
        """Download the plugin's tool receipts of this attempt."""
        return await self._records(context, "hook-applied.jsonl")

    async def enrich_exports(
        self, context: RunContext, exports: dict[str, dict], artifacts: dict[str, str]
    ) -> tuple[dict[str, dict], list[dict[str, JsonValue]]]:
        """Keep immutable raw exports beside replayable receipt-enriched copies."""
        applied = await self.records(context)
        continuation = context.spec.continuation
        if continuation is not None:
            roots = [key for key, value in exports.items() if value.get("info", {}).get("parentID") is None]
            if len(roots) != 1:
                raise ValueError("restored error receipts require exactly one root session")
            applied += [
                record
                | {
                    "kind": "tool_receipt",
                    "session_id": roots[0],
                    "restored_from": continuation.prefix_export["info"]["id"],
                }
                for record in prefix_receipts(continuation)
            ]
        enriched = merge_hook_receipts(exports, applied)
        if context.spec.tools.tool_names is not None:
            preserve_rejected_calls(enriched, context.model_observations, continuation)
        for session_id, document in enriched.items():
            path = context.artifacts_dir / f"export-{session_id}-with-if-receipts.json"
            await asyncio.to_thread(path.write_text, json.dumps(document))
            artifacts[f"native_export:{session_id}"] = artifacts[session_id]
            artifacts[session_id] = str(path)
        return enriched, applied

    async def _records(self, context: RunContext, filename: str) -> list[dict[str, JsonValue]]:
        path = context.artifacts_dir / filename
        await context.sandbox.download(self.directory + "/" + filename, path)
        content = await asyncio.to_thread(path.read_text)
        result = [json.loads(line) for line in content.splitlines() if line.strip()]
        if not all(isinstance(row, dict) for row in result):
            raise ValueError("hook receipt must be a JSON object")
        return result


async def stage_hooks(context: RunContext, sandbox: BinarySandbox) -> StagedHooks:
    """Upload one compiled pipeline outside the task repository."""
    config = compile_hooks(context.spec)
    source = Path(__file__).parent
    continuation = context.spec.continuation
    if continuation is not None:
        config["prefix_receipts"] = prefix_receipts(continuation)
        # The plugin does not read the mode; it stays in the hashed configuration that tool receipts record.
        config["prefix_mode"] = continuation.mode
    config["profile_sha256"] = profile_sha256(config)
    files = {
        "hook-config.json": json.dumps(config),
        "hook-events.json": json.dumps(
            [event.model_dump(mode="json", exclude_none=True) for event in context.spec.events]
        ),
        "if-profile.mjs": (source / "if-profile.js").read_text(),
        # Loaded through if-request-kind.mjs, which imports it by this name; not a plugin entry of its own.
        "assistant-message-header.mjs": OBSERVABILITY_PATCH.read_text(),
        "if-request-kind.mjs": (source / "if-request-kind.js").read_text(),
    }
    with tempfile.TemporaryDirectory(prefix="if-native-hooks-") as temporary:
        for filename, content in files.items():
            path = Path(temporary) / filename
            path.write_text(content)
            await context.sandbox.upload(path, sandbox.directory + "/" + filename)
    paths = " ".join(quote(sandbox.directory + "/" + filename) for filename in files)
    mode = await context.sandbox.transport.exec(command=f"chmod 0644 {paths}", user="root", timeout_s=30)
    if mode.return_code != 0 or mode.error_type:
        raise RuntimeError("cannot stage native hooks")
    alias_gate = (
        (context.model.base_url.removesuffix("/v1") + "/tool-call", context.model.api_key)
        if config.get("check_aliases")
        else None
    )
    return StagedHooks(sandbox.directory, config["profile_sha256"], alias_gate)
