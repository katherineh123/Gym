# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authoritative native session inventory shared by prefix replay and live capture."""

import asyncio
import json
import logging
import re
from shlex import quote

from environment_servers.agentic_if.bootstrap import BinarySandbox
from environment_servers.agentic_if.runtime import RunContext
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.if_grading.contracts import SessionInfo


LOG = logging.getLogger(__name__)


async def _collection_error(context: RunContext, name: str, result: SandboxExecResult) -> RuntimeError:
    # Keep process diagnostics beside private native exports, never in shared
    # service logs. Do not record the command environment or accept partial JSON.
    stdout, stderr = result.stdout or "", result.stderr or ""
    receipt = {
        "return_code": result.return_code,
        "error_type": result.error_type,
        "stdout": stdout[:8192],
        "stderr": stderr[:8192],
        "stdout_truncated": len(stdout) > 8192,
        "stderr_truncated": len(stderr) > 8192,
    }
    path = context.artifacts_dir / f"native-{name}-failure.json"
    await asyncio.to_thread(path.write_text, json.dumps(receipt) + "\n")
    return RuntimeError(
        f"native OpenCode {name} failed (return_code={result.return_code}, error_type={result.error_type})"
    )


async def collect_sessions(
    context: RunContext, sandbox: BinarySandbox, env: dict[str, str]
) -> tuple[dict[str, dict], list[SessionInfo], dict[str, str]]:
    """Export every session of the native data store: (exports by ID, session inventory, saved export paths).

    The inventory is authoritative: a session whose export fails is logged and left out of the exports, so its
    evidence is marked missing rather than empty.
    """
    # Query through the pinned executable: no sqlite3/Python package needs
    # to exist in a vendor image, and parent relationships come from storage.
    inventory = await sandbox.exec(
        command="opencode db 'SELECT id, parent_id FROM session ORDER BY time_created, id' --format json",
        env=env,
        timeout_s=60,
    )
    if inventory.return_code != 0 or inventory.error_type:
        raise await _collection_error(context, "inventory", inventory)
    rows = json.loads(inventory.stdout)
    if not isinstance(rows, list) or not rows:
        raise ValueError("native OpenCode session inventory is empty")
    sessions = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"])
        ):
            raise ValueError("invalid native OpenCode session ID")
        sessions.append(SessionInfo(id=row["id"], parent_id=row.get("parent_id")))
    if (
        len({session.id for session in sessions}) != len(sessions)
        or len([s for s in sessions if s.parent_id is None]) != 1
    ):
        raise ValueError("native OpenCode session inventory does not identify one root")
    exports, artifacts = {}, {}
    for session in sessions:
        target = sandbox.directory + f"/export-{session.id}.json"
        path = context.artifacts_dir / f"export-{session.id}.json"
        try:
            exported = await sandbox.exec(
                command=f"opencode export {quote(session.id)} > {quote(target)}", env=env, timeout_s=60
            )
            if exported.return_code != 0 or exported.error_type:
                raise await _collection_error(context, f"export-{session.id}", exported)
            await sandbox.download(target, path)
            value = await asyncio.to_thread(path.read_text)
            exports[session.id] = json.loads(value)
            artifacts[session.id] = str(path)
        except Exception:
            # Retain the authoritative inventory so evidence marks the
            # missing session, even if every other export succeeded.
            LOG.exception("Cannot collect OpenCode session %s in run %s", session.id, context.run_id)
    return exports, sessions, artifacts
