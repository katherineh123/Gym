# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived releases, the neutral history label and the rotated seed schedules."""

import json
from pathlib import Path

import pytest

from benchmarks.agentic_if_realworld import prepare
from resources_servers.agentic_if_realworld.benchmark import core
from resources_servers.agentic_if_realworld.benchmark.core import (
    CONVERSATION_MARKER,
    HISTORY_MARKER,
    PHASES,
    TOOLS_MARKER,
    Dataset,
    captured_context,
    derived_from,
    digest,
    phases,
    read,
    write,
)
from resources_servers.agentic_if_realworld.benchmark.synthetic import create


FROZEN_MANIFEST_SHA256 = read(core.ROOT / "source_manifest.json")["release_manifest_sha256"]
ACCOUNTING = {
    "views": {"original_failure_focused": 1},
    "atomic_checks": 1,
    "nonempty_registries": 0,
    "empty_registries": 1,
}


def derive(directory: Path, parent: str = FROZEN_MANIFEST_SHA256, accounting: dict = ACCOUNTING) -> Path:
    """Rewrite the one-task synthetic fixture's manifest as a derived release of the frozen manifest."""
    create(directory)
    manifest = read(directory / "release_manifest.json")
    del manifest["synthetic"]
    manifest["derived_from"] = {"release_manifest_sha256": parent, "version": "cleaned-subset-test"}
    manifest["accounting"] = accounting
    write(directory / "release_manifest.json", manifest)
    return directory


def test_derived_release_declares_its_own_size_and_records_its_parent(tmp_path):
    dataset = Dataset(derive(tmp_path / "data"))
    assert set(dataset.inputs) == {"synthetic"}
    assert dataset.report["derived_from"]["release_manifest_sha256"] == FROZEN_MANIFEST_SHA256
    assert dataset.manifest_sha256 == digest(tmp_path / "data" / "release_manifest.json")
    assert derived_from(tmp_path / "data") == dataset.derived_from


def test_derived_release_must_name_the_frozen_manifest(tmp_path):
    with pytest.raises(ValueError, match="must name the frozen release manifest"):
        Dataset(derive(tmp_path / "data", parent="0" * 64))


def test_derived_release_artifacts_stay_bound_to_its_manifest(tmp_path):
    directory = derive(tmp_path / "data")
    criteria = directory / "criteria.jsonl"
    row = json.loads(criteria.read_text())
    row["constraints"][0]["evaluation_question"] = "Edited after the manifest was written?"
    criteria.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Dataset artifact hash mismatch: criteria.jsonl"):
        Dataset(directory)


def test_derived_release_accounting_must_match_its_contents(tmp_path):
    with pytest.raises(ValueError, match="Derived release accounting mismatch"):
        Dataset(derive(tmp_path / "data", accounting={**ACCOUNTING, "atomic_checks": 2}))


def test_release_without_a_parent_still_needs_the_frozen_manifest(tmp_path):
    directory = tmp_path / "data"
    create(directory)
    manifest = read(directory / "release_manifest.json")
    del manifest["synthetic"]
    write(directory / "release_manifest.json", manifest)
    assert derived_from(directory) is None
    with pytest.raises(ValueError, match="Release manifest hash mismatch"):
        Dataset(directory)


def test_prepare_never_fetches_the_frozen_release_into_a_derived_one(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    script = checkout / "benchmarks/agentic_if_realworld/prepare.py"
    script.parent.mkdir(parents=True)
    directory = derive(tmp_path / "release")

    def fetch(_directory):
        raise AssertionError("fetch_release must not run for a derived release")

    monkeypatch.setattr(prepare, "__file__", str(script))
    monkeypatch.setattr(prepare, "fetch_release", fetch)
    monkeypatch.setattr(prepare, "source_preflight", lambda *args, **kwargs: {})
    monkeypatch.delenv("AGENTIC_IF_SYNTHETIC", raising=False)
    monkeypatch.setenv("AGENTIC_IF_DATA_DIR", str(directory))
    monkeypatch.setenv("AGENTIC_IF_PREPARED_PATH", str(tmp_path / "prepared" / "prepared.jsonl"))
    configured = prepare.prepare()
    assert [json.loads(line) for line in configured.open()] == Dataset(directory).prepared()


def wrapper(*markers: str) -> dict:
    history = json.dumps([{"role": "user", "content": "hello"}])
    content = "Write the next turn.\n" + TOOLS_MARKER + "[]" + "".join(markers) + history
    return {"messages": [{"role": "user", "content": content}]}


def test_neutral_history_label_is_accepted_once():
    tools, history, quote = captured_context(wrapper(CONVERSATION_MARKER))
    assert tools == [] and history == [{"role": "user", "content": "hello"}] and quote == "[]"
    assert captured_context(wrapper(HISTORY_MARKER))[1] == history
    with pytest.raises(ValueError, match="ambiguous wrapper boundaries"):
        captured_context(wrapper(CONVERSATION_MARKER, HISTORY_MARKER))


def test_seed_schedules_rotate_only_the_seeds():
    assert phases() == phases(17) == PHASES
    assert phases(29) == ((29, 32768), (29, 65536), (29, 131072), (43, 131072), (17, 131072))
    assert phases(43) == ((43, 32768), (43, 65536), (43, 131072), (17, 131072), (29, 131072))
    for first_seed in (17, 29, 43):
        assert [cap for _, cap in phases(first_seed)] == [cap for _, cap in PHASES]
    with pytest.raises(ValueError, match="17, 29 or 43"):
        phases(5)
