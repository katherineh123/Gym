# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from types import SimpleNamespace

import pytest


def export(session, tool="bash"):
    return {
        "info": {"id": session},
        "messages": [
            {
                "info": {"id": "message", "sessionID": session, "role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": tool,
                        "callID": "same",
                        "state": {
                            "status": "error",
                            "input": {
                                "command": "exit 3",
                                "__nemo_gym_profile": {"configured_args": {"cmd": "exit 3"}},
                            },
                            "error": "failure",
                        },
                    }
                ],
            }
        ],
    }


def receipt(session, code):
    return {
        "kind": "tool_receipt",
        "session_id": session,
        "call_id": "same",
        "native_tool": "bash",
        "metadata": {"configured_args": {"cmd": "exit 3"}, "workspace_diff": f"diff {code}"},
    }


def test_failure_receipts_join_exact_sessions_without_overwriting_native_status_or_output():
    from environment_servers.agentic_if.drivers.opencode_hooks import merge_hook_receipts

    originals = {"root": export("root"), "child": export("child")}
    saved = deepcopy(originals)
    result = merge_hook_receipts(originals, [receipt("child", 4), receipt("root", 3)])
    for session, code in [("root", 3), ("child", 4)]:
        state = result[session]["messages"][0]["parts"][0]["state"]
        assert state["metadata"]["nemo_gym_profile"]["workspace_diff"] == f"diff {code}"
        assert state["input"] == {"cmd": "exit 3"}
        assert state["status"] == "error" and state["error"] == "failure"
    assert originals == saved


def presentation(session="root"):
    return {
        "kind": "tool_presentation",
        "session_id": session,
        "call_id": "same",
        "native_tool": "bash",
        "native_error": "failure",
        "model_error": "RunCommand failed: failure",
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_presentation_receipt_preserves_execution_facts_and_raw_error(reverse):
    from environment_servers.agentic_if.drivers.opencode_hooks import merge_hook_receipts
    from environment_servers.agentic_if.drivers.opencode_profile import model_visible_tool_output

    original = {"root": export("root")}
    before = deepcopy(original)
    records = [receipt("root", 3), presentation()]
    result = merge_hook_receipts(original, records[::-1] if reverse else records)
    state = result["root"]["messages"][0]["parts"][0]["state"]
    assert state["status"] == "error" and state["error"] == "failure"
    assert state["metadata"]["nemo_gym_profile"]["workspace_diff"] == "diff 3"
    assert model_visible_tool_output(state) == "RunCommand failed: failure"
    assert original == before


def test_identical_replayed_presentation_is_idempotent():
    from environment_servers.agentic_if.drivers.opencode_hooks import merge_hook_receipts

    records = [receipt("root", 3), presentation()]
    assert merge_hook_receipts({"root": export("root")}, records) == merge_hook_receipts(
        {"root": export("root")}, records + [presentation()]
    )


@pytest.mark.parametrize("defect", ["native_error", "session", "tool", "duplicate", "output", "status"])
def test_unbound_presentation_receipt_cannot_replace_native_error(defect):
    from environment_servers.agentic_if.drivers.opencode_hooks import merge_hook_receipts

    original = {"root": export("root")}
    records = [receipt("root", 3), presentation()]
    if defect == "native_error":
        records[-1]["native_error"] = "different failure"
    elif defect == "session":
        records[-1]["session_id"] = "child"
    elif defect == "tool":
        records[-1]["native_tool"] = "read"
    elif defect == "duplicate":
        records.append(presentation() | {"model_error": "Conflicting presentation"})
    elif defect == "output":
        records[-1]["model_error"] = None
    else:
        original["root"]["messages"][0]["parts"][0]["state"]["status"] = "completed"
    with pytest.raises(ValueError, match="presentation"):
        merge_hook_receipts(original, records)


def test_alias_rejection_receipt_is_explicit_execution_evidence():
    from environment_servers.agentic_if.drivers.opencode_hooks import merge_hook_receipts

    native = export("root", "read")
    native["messages"][0]["parts"][0]["state"]["input"] = {"filePath": "/tmp/file"}
    result = merge_hook_receipts(
        {"root": native},
        [
            {
                "kind": "alias_rejected",
                "session_id": "root",
                "call_id": "same",
                "native_tool": "read",
                "model_name": "read",
                "error": "read unavailable",
            }
        ],
    )
    stamp = result["root"]["messages"][0]["parts"][0]["state"]["metadata"]["nemo_gym_profile"]
    assert stamp["dispatch_rejected"] is True


@pytest.mark.parametrize("case", ["duplicate", "wrong_tool", "missing_session"])
def test_incomplete_or_mismatched_receipts_do_not_make_a_replayable_export(case):
    from environment_servers.agentic_if.drivers.opencode_hooks import merge_hook_receipts

    records = [receipt("root", 3)]
    if case == "duplicate":
        records.append(receipt("root", 4))
    if case == "wrong_tool":
        records[0]["native_tool"] = "write"
    if case == "missing_session":
        records[0]["session_id"] = "child"
    with pytest.raises(ValueError, match="unbound or duplicate hook execution receipt"):
        merge_hook_receipts({"root": export("root")}, records)


@pytest.mark.asyncio
async def test_prefix_receipts_bind_to_the_resumed_root(tmp_path, monkeypatch):
    from environment_servers.agentic_if.drivers.opencode_hooks import StagedHooks, merge_hook_receipts

    frozen = merge_hook_receipts({"old-root": export("old-root")}, [receipt("old-root", 3)])["old-root"]
    saved = deepcopy(frozen)
    continuation = SimpleNamespace(mode="restore", prefix_export=frozen)
    context = SimpleNamespace(
        spec=SimpleNamespace(continuation=continuation, tools=SimpleNamespace(tool_names=None)), artifacts_dir=tmp_path
    )
    native = {"new-root": export("new-root"), "child": export("child")}
    native["child"]["info"]["parentID"] = "new-root"
    records = [receipt("child", 4)]

    async def captured_records(self, context):
        return deepcopy(records)

    monkeypatch.setattr(StagedHooks, "records", captured_records)
    artifacts = {"new-root": "raw-root.json", "child": "raw-child.json"}
    enriched, applied = await StagedHooks("/private/run", "hash").enrich_exports(context, native, artifacts)
    for session, code in [("new-root", 3), ("child", 4)]:
        stamp = enriched[session]["messages"][0]["parts"][0]["state"]["metadata"]["nemo_gym_profile"]
        assert stamp["workspace_diff"] == f"diff {code}"
    (historical,) = [row for row in applied if "restored_from" in row]
    assert historical["session_id"] == "new-root"
    assert historical["restored_from"] == "old-root"
    assert artifacts["native_export:new-root"] == "raw-root.json"
    assert "metadata" not in native["new-root"]["messages"][0]["parts"][0]["state"]
    assert frozen == saved
