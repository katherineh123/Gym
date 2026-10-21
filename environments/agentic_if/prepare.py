# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the agentic IF instruction-following benchmark.

The sealed rows are private. They ship separately as gzipped parts, ``rows/instruction-following-<n>.jsonl.gz``, under a
plugin root that ``gym eval prepare`` reaches through ``--search-dir``. Terminal rows reference task folders of the internal
terminal-bench-tasks repository by paths relative to the Gym root; preparation checks out exactly those folders at the
pinned commit.
"""

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from nemo_gym import _resolve_under_cwd_or_install
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import validate_runtime


ENVIRONMENT_DIR = Path(__file__).parent
GYM_ROOT = ENVIRONMENT_DIR.parents[1]
OUTPUT_FPATH = ENVIRONMENT_DIR / "data" / "instruction-following.jsonl"

# Version 2.0.0, a reviewed subset of the eval rows, split at a row boundary to fit repository file-size limits.
# The SHA-256 is of the uncompressed parts, concatenated in order.
ROWS_PATHS = (Path("rows/instruction-following-1.jsonl.gz"), Path("rows/instruction-following-2.jsonl.gz"))
ROWS_SHA256 = "0c673da2c3d4c8465c3854ffd4c292a120a16043bbd8b8aa8d8c4cd5f292709a"  # pragma: allowlist secret

TASKS_REPOSITORY = "https://gitlab-master.nvidia.com/nexus-team/terminal-bench-tasks.git"
TASKS_REVISION = "1bfa40874d684e359e7635739422326cfb39d71c"  # pragma: allowlist secret
TASKS_CHECKOUT = ENVIRONMENT_DIR / "terminal-bench-tasks"


def _task_folders(rows: list[dict]) -> list[str]:
    """Check each row's identity and receipts; return the repository folders terminal rows reference."""
    prefix = f"{TASKS_CHECKOUT.relative_to(GYM_ROOT).as_posix()}/"
    task_ids, folders = set(), set()
    for row in rows:
        content = {key: value for key, value in row.items() if key != "task_id"}
        if row.get("task_id") != digest(content):
            raise ValueError(f"Row {row.get('task_id')!r} does not carry its content digest as task_id")
        if row["task_id"] in task_ids:
            raise ValueError(f"Duplicate task_id {row['task_id']}")
        task_ids.add(row["task_id"])
        validate_runtime(content)
        if (folder := content.get("task_folder")) is None:
            continue
        path = PurePosixPath(folder)
        if not folder.startswith(prefix) or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"task_folder {folder!r} is not a folder of {prefix}")
        folders.add(folder.removeprefix(prefix))
    return sorted(folders)


def _git(*args: str) -> None:
    # Rows run prebuilt images, so the task folders' Git LFS build inputs are never needed.
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    try:
        subprocess.run(
            ["git", "-C", str(TASKS_CHECKOUT), *args],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"git {args[0]} failed: {error.stderr.strip() or error}") from error


def _check_out(folders: list[str], repository: str) -> None:
    """Check out only the referenced task folders, at the pinned commit."""
    TASKS_CHECKOUT.mkdir(parents=True, exist_ok=True)
    if not (TASKS_CHECKOUT / ".git").exists():
        _git("init", "--quiet")
        _git("remote", "add", "origin", repository)
    else:
        _git("remote", "set-url", "origin", repository)
    _git("fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", TASKS_REVISION)
    _git("sparse-checkout", "init", "--cone")
    _git("sparse-checkout", "set", *folders)
    _git("checkout", "--quiet", "--detach", TASKS_REVISION)
    missing = [folder for folder in folders if not (TASKS_CHECKOUT / folder / "tests/test.sh").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} task folders have no tests/test.sh at {TASKS_REVISION}: {missing[:3]}")


def prepare(tasks_repository: str = TASKS_REPOSITORY) -> Path:
    """Read the pinned rows and check out their task folders; the rows are written last."""
    sources = [_resolve_under_cwd_or_install(path) for path in ROWS_PATHS]
    if missing := [str(path) for path, source in zip(ROWS_PATHS, sources) if not source.is_file()]:
        raise FileNotFoundError(
            f"{', '.join(missing)} not found: pass the plugin root that holds them with --search-dir"
        )
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".instruction-following-", dir=OUTPUT_FPATH.parent) as directory:
        unpacked = Path(directory) / OUTPUT_FPATH.name
        with unpacked.open("wb") as file:
            for source in sources:
                with gzip.open(source, "rb") as compressed:
                    shutil.copyfileobj(compressed, file)
        with unpacked.open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != ROWS_SHA256:
            raise RuntimeError(f"The rows under {sources[0].parent} have SHA-256 {actual}, expected {ROWS_SHA256}")
        rows = [json.loads(line) for line in unpacked.read_text().splitlines() if line.strip()]
        _check_out(_task_folders(rows), tasks_repository)
        os.replace(unpacked, OUTPUT_FPATH)
    print(f"Prepared {len(rows)} agentic IF tasks at {OUTPUT_FPATH}")
    return OUTPUT_FPATH.absolute()


if __name__ == "__main__":
    prepare()
