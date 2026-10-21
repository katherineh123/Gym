# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparing the private instruction-following split: pinned rows and the task folders they reference."""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environments.agentic_if import prepare as prepare_module
from nemo_gym.config_types import DatasetConfig
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, seal_runtime_row


CHECKS = [{"id": "end", "kind": "exact", "parameters": {"value": "DONE"}, "scope": {"places": ["final"]}}]


def git(repository: Path, *args: str) -> str:
    command = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-C", str(repository), *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def flat_row(task_folder: str, *, text: str = "Fix it") -> dict:
    sealed = seal_runtime_row(
        {"task_name": "terminal-bench/a", "docker_image": "registry/a:1", "task_folder": task_folder}
        | {"responses_create_params": {"input": [{"role": "user", "content": text}]}},
        RuntimeSpec(harness="opencode", task_backend="tb21", checks=CHECKS),
    )
    return {"task_id": digest(sealed)} | sealed


@pytest.fixture
def tasks_repository(tmp_path):
    """A task source whose tip has moved past the pinned commit."""
    repository = tmp_path / "source"
    for folder in ("tasks/s/a", "tasks/s/b", "tasks/t/c"):
        (repository / folder / "tests").mkdir(parents=True)
        (repository / folder / "tests/test.sh").write_text(f"echo pinned {folder}\n")
    git(repository, "init", "--quiet")
    git(repository, "config", "uploadpack.allowFilter", "true")
    git(repository, "config", "uploadpack.allowAnySHA1InWant", "true")
    git(repository, "add", ".")
    git(repository, "commit", "--quiet", "-m", "pinned")
    pinned = git(repository, "rev-parse", "HEAD")
    (repository / "tasks/s/a/tests/test.sh").write_text("echo moved\n")
    git(repository, "commit", "--quiet", "-am", "moved")
    return repository, pinned


@pytest.fixture
def isolated(tmp_path, monkeypatch, tasks_repository):
    repository, pinned = tasks_repository
    monkeypatch.setattr(prepare_module, "GYM_ROOT", tmp_path)
    monkeypatch.setattr(prepare_module, "TASKS_CHECKOUT", tmp_path / "checkout")
    monkeypatch.setattr(prepare_module, "TASKS_REVISION", pinned)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", tmp_path / "data" / "instruction-following.jsonl")
    return f"file://{repository}"


def test_environment_config_declares_the_private_prepared_split():
    config = OmegaConf.load(prepare_module.ENVIRONMENT_DIR / "config.yaml")
    datasets = config.agentic_if_resources.resources_servers.agentic_if.datasets
    [raw_validation] = [dataset for dataset in datasets if dataset.type == "validation"]
    validation = DatasetConfig.model_validate(raw_validation)
    assert Path(validation.jsonl_fpath) == prepare_module.OUTPUT_FPATH.relative_to(prepare_module.GYM_ROOT)
    assert validation.prepare_script == Path("environments/agentic_if/prepare.py")


def test_rows_must_carry_their_digest_valid_receipts_and_a_checkout_folder(isolated):
    assert prepare_module._task_folders([flat_row("checkout/tasks/s/a")]) == ["tasks/s/a"]

    tampered = flat_row("checkout/tasks/s/a")
    tampered["responses_create_params"]["input"][0]["content"] = "Do something else"
    with pytest.raises(ValueError, match="content digest"):
        prepare_module._task_folders([tampered])
    resealed = {key: value for key, value in tampered.items() if key != "task_id"}
    with pytest.raises(ValueError, match="receipt"):
        prepare_module._task_folders([{"task_id": digest(resealed)} | resealed])
    for folder in ("/data/tasks/s/a", "checkout/../tasks/s/a", "elsewhere/tasks/s/a"):
        with pytest.raises(ValueError, match="is not a folder of checkout/"):
            prepare_module._task_folders([flat_row(folder)])


def test_checkout_holds_only_the_referenced_folders_at_the_pinned_commit(isolated):
    prepare_module._check_out(["tasks/s/a"], isolated)
    checkout = prepare_module.TASKS_CHECKOUT
    assert git(checkout, "rev-parse", "HEAD") == prepare_module.TASKS_REVISION
    assert (checkout / "tasks/s/a/tests/test.sh").read_text() == "echo pinned tasks/s/a\n"
    assert not (checkout / "tasks/s/b").exists() and not (checkout / "tasks/t").exists()

    # Preparing again for other rows moves the same checkout to exactly their folders.
    prepare_module._check_out(["tasks/t/c"], isolated)
    assert (checkout / "tasks/t/c/tests/test.sh").is_file() and not (checkout / "tasks/s/a").exists()


def test_a_folder_without_its_verifier_fails_preparation(isolated):
    with pytest.raises(RuntimeError, match="no tests/test.sh"):
        prepare_module._check_out(["tasks/s/missing"], isolated)


def test_prepare_writes_the_rows_only_after_the_pinned_artifact_and_its_folders_check_out(
    isolated, tmp_path, monkeypatch
):
    with pytest.raises(FileNotFoundError, match="--search-dir"):
        prepare_module.prepare(tasks_repository=isolated)

    rows = [json.dumps(flat_row(folder)) + "\n" for folder in ("checkout/tasks/s/a", "checkout/tasks/t/c")]
    artifact = "".join(rows)
    plugin = tmp_path / "plugin"
    (plugin / "rows").mkdir(parents=True)
    for path, row in zip(prepare_module.ROWS_PATHS, rows, strict=True):
        (plugin / path).write_bytes(gzip.compress(row.encode()))
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(plugin))
    monkeypatch.setattr(prepare_module, "ROWS_SHA256", hashlib.sha256(b"another release").hexdigest())
    with pytest.raises(RuntimeError, match="expected"):
        prepare_module.prepare(tasks_repository=isolated)
    assert not prepare_module.OUTPUT_FPATH.exists()

    monkeypatch.setattr(prepare_module, "ROWS_SHA256", hashlib.sha256(artifact.encode()).hexdigest())
    output = prepare_module.prepare(tasks_repository=isolated)
    assert output == prepare_module.OUTPUT_FPATH.absolute()
    assert output.read_text() == artifact
    assert (prepare_module.TASKS_CHECKOUT / "tasks/s/a/tests/test.sh").is_file()
    assert (prepare_module.TASKS_CHECKOUT / "tasks/t/c/tests/test.sh").is_file()
    assert [path.name for path in output.parent.iterdir()] == [output.name]
