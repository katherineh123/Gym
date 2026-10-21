# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native prefix restoration; no separate rollout or grading lifecycle."""

import json
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from shlex import quote

from pydantic import JsonValue

from environment_servers.agentic_if.bootstrap import BinarySandbox
from environment_servers.agentic_if.drivers import opencode_prefix as replay
from environment_servers.agentic_if.drivers.opencode_sessions import collect_sessions
from environment_servers.agentic_if.runtime import RunContext
from environment_servers.agentic_if.workspace import WORKSPACE_DIFF
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec


def workspace_patches_equal(left: str, right: str) -> bool:
    """Compare recorded Git patches, allowing only compatible object-ID abbreviations.

    Git's default abbreviation length depends on repository configuration. It is
    not a file-content difference. Keep every other line (including modes, binary
    bodies, and no-newline markers) exact; even index IDs must agree by prefix.
    """
    first, second = left.splitlines(), right.splitlines()
    if len(first) != len(second):
        return False
    index = re.compile(r"index ([0-9a-f]{4,64})\.\.([0-9a-f]{4,64})( [0-7]{6})?")
    for a, b in zip(first, second, strict=True):
        if a == b:
            continue
        x, y = index.fullmatch(a), index.fullmatch(b)
        if x is None or y is None or x[3] != y[3]:
            return False
        if any(not (x[k].startswith(y[k]) or y[k].startswith(x[k])) for k in (1, 2)):
            return False
    return True


async def _upload(sandbox: BinarySandbox, *, filename: str, text: str) -> str:
    remote = sandbox.directory + "/" + filename
    with tempfile.TemporaryDirectory(prefix="if-replay-") as temporary:
        path = Path(temporary) / filename
        path.write_text(text)
        await sandbox.upload(path, remote)
    mode = await sandbox.transport.exec(command=f"chmod 0644 {quote(remote)}", user="root", timeout_s=30)
    if mode.return_code != 0 or mode.error_type:
        raise RuntimeError("cannot stage native prefix")
    return remote


async def _workspace_diff(context: RunContext, *, exclude: tuple[str, ...] = ()) -> str:
    command = WORKSPACE_DIFF
    if exclude:
        paths = " ".join(quote(":(top,literal,exclude)" + path) for path in exclude)
        command = command.replace(
            "git diff --binary --full-index",
            f"git diff --binary --full-index -- . {paths}",
        )
    current = await context.sandbox.exec(command=command, timeout_s=600)
    if current.return_code != 0 or current.error_type:
        raise RuntimeError("cannot record replay workspace diff")
    return current.stdout or ""


async def restore_workspace(
    context: RunContext, sandbox: BinarySandbox, continuation: ContinuationSpec
) -> dict[str, JsonValue]:
    """Restore recorded repository state after replaying shell environment effects."""
    source = continuation.prefix_export
    workdir = context.sandbox.workdir or str(source["info"].get("directory") or ".")
    mismatches = []
    commands = replay.recorded_commands(source, default_workdir=workdir)
    remote = None
    touched = set()
    if continuation.workspace_diff.strip():
        remote = await _upload(sandbox, filename="prefix.patch", text=continuation.workspace_diff)
        stats = await context.sandbox.exec(command=f"git apply --numstat -z {quote(remote)}", timeout_s=30)
        if stats.return_code or stats.error_type:
            raise RuntimeError("cannot inspect recorded prefix workspace diff")
        touched = {entry.split("\t", 2)[2] for entry in (stats.stdout or "").split("\0") if entry}
    initial_paths = []
    for path in context.spec.workspace_files:
        if path.startswith("/"):
            prefix = workdir.rstrip("/") + "/"
            if not path.startswith(prefix):
                # Repository cleanup cannot touch a declared file outside it.
                continue
            path = path[len(prefix) :]
        initial_paths.append(path)
    untouched = tuple(path for path in initial_paths if path not in touched)
    for index, recorded in enumerate(commands):
        result = await context.sandbox.exec(
            command=f'"${{SHELL:-/bin/sh}}" -c {quote(recorded.command)}',
            cwd=recorded.workdir,
            timeout_s=recorded.timeout_s,
        )
        if result.error_type:
            raise RuntimeError("prefix shell replay failed at the sandbox transport")
        observed = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.return_code != 0 or observed != recorded.output.strip():
            mismatches.append(index)
    # This is a fresh, resource-owned task sandbox, never the host checkout.
    checkout = " ".join(quote(":(top,literal,exclude)" + path) for path in untouched)
    clean = " ".join("-e " + quote("/" + path) for path in untouched)
    reset = await context.sandbox.exec(
        command=f"git checkout -- . {checkout} && git clean -fdq {clean}", timeout_s=600
    )
    if reset.return_code != 0 or reset.error_type:
        raise RuntimeError("cannot reset the sandbox repository before restoring the prefix")
    if continuation.workspace_diff.strip():
        applied = await context.sandbox.exec(
            command=f"git apply --binary --whitespace=nowarn {quote(remote)}", timeout_s=600
        )
        if applied.return_code != 0 or applied.error_type:
            raise RuntimeError("recorded prefix workspace diff does not apply")
    current = await _workspace_diff(context, exclude=untouched)
    return {
        "mode": "restore",
        "replayed_commands": len(commands),
        "command_output_mismatch_indices": mismatches,
        "workspace_matches_diff": workspace_patches_equal(current, continuation.workspace_diff),
    }


@dataclass(frozen=True)
class ContinuedSession:
    """Actual prefix session and message IDs, before the fresh user continuation."""

    session_id: str
    prefix_ids: frozenset[str]
    receipt: dict[str, JsonValue]
    artifacts: dict[str, str]

    def attach(self, evidence: SessionEvidence) -> SessionEvidence:
        """Return a copy of ``evidence`` with this replay's receipt, artifacts and continuation start."""
        result = evidence.model_copy(deep=True)
        starts = [
            index
            for index, step in enumerate(result.steps)
            if step.session_id == self.session_id and step.role == "user" and step.id not in self.prefix_ids
        ]
        result.facts["replay"] = self.receipt
        result.artifacts.update(self.artifacts)
        if starts:
            result.continuation_start = starts[0]
        else:
            result.status = "incomplete"
            result.issues.append("missing_continuation_user_message")
        if not self.receipt["workspace_matches_diff"]:
            result.status = "incomplete"
            result.issues.append("replay_workspace_diverged")
        if not self.receipt["tool_outputs_match"]:
            result.status = "incomplete"
            result.issues.append("restored_prefix_history_diverged")
        return result


async def prepare_continuation(context: RunContext, sandbox: BinarySandbox, env: dict[str, str]) -> ContinuedSession:
    """Restore a canonical native prefix, retaining real replay evidence."""
    continuation = context.spec.continuation
    if continuation is None:
        raise ValueError("native prefix is required")
    recorded = replay.recorded_workspace_diff(continuation.prefix_export)
    if recorded is not None and not workspace_patches_equal(recorded, continuation.workspace_diff):
        raise ValueError("continuation workspace diff disagrees with the recorded prefix")
    receipt = await restore_workspace(context, sandbox, continuation)
    path = await _upload(sandbox, filename="prefix-export.json", text=json.dumps(continuation.prefix_export))
    imported = await sandbox.exec(command=f"opencode import {quote(path)}", env=env, timeout_s=600)
    if imported.return_code != 0 or imported.error_type:
        raise RuntimeError("native prefix import failed")
    prefix_context = replace(context, artifacts_dir=context.artifacts_dir / "prefix")
    prefix_context.artifacts_dir.mkdir(parents=True, exist_ok=True)
    exports, sessions, artifacts = await collect_sessions(prefix_context, sandbox, env)
    roots = [session.id for session in sessions if session.parent_id is None]
    if len(roots) != 1 or set(exports) != {roots[0]}:
        raise RuntimeError("native prefix did not produce exactly its root session")
    export = exports[roots[0]]
    receipt["prefix_messages"] = len(export["messages"])
    # Workspace divergence and real tool-result divergence are distinct facts.
    expected = [state.get("output", state.get("error")) for _, state in replay.tool_parts(continuation.prefix_export)]
    observed = [state.get("output", state.get("error")) for _, state in replay.tool_parts(export)]
    receipt["tool_outputs_match"] = expected == observed
    return ContinuedSession(
        roots[0], frozenset(replay.prefix_message_ids(export)), receipt, {"prefix_export": artifacts[roots[0]]}
    )
