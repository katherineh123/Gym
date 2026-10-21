# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install OpenCode in the task sandbox: the real installer with stubbed downloads and package managers."""

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from environment_servers.agentic_if import bootstrap
from environment_servers.agentic_if.drivers.opencode import OpenCodeDriver
from environment_servers.agentic_if.runtime import BoundSandbox
from environment_servers.agentic_if.tests.support import Provider, context
from nemo_gym.sandbox.providers import SandboxExecResult


BINARY = b"#!/bin/sh\necho 1.17.11\n"
PIN = hashlib.sha256(BINARY).hexdigest()
URL = "https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.17.11.tgz"
# Stands in for curl: logs the URL and copies the package of that name from the test root.
FAKE_CURL = """#!/bin/sh
while [ "$#" -gt 0 ]; do
  case "$1" in -o) output=$2; shift ;; https://*) url=$1 ;; esac
  shift
done
echo "$url" >> "$TEST_ROOT/download.log"
[ "${TEST_DOWNLOAD_FAIL:-0}" = 0 ] || exit 22
exec cp "$TEST_ROOT/${url##*/}" "$output"
"""
# Stands in for apt-get and apk: logs each call and "installs" curl from the test root.
FAKE_PACKAGE_MANAGER = """#!/bin/sh
echo "${0##*/} $*" >> "$TEST_ROOT/packages.log"
for package in "$@"; do
  if [ "$package" = curl ]; then cp "$TEST_ROOT/curl" "$TEST_ROOT/bin/curl"; fi
done
"""
FAKE_ID = '#!/bin/sh\necho "${TEST_UID:-0}"\n'
CA_BUNDLES = {"/etc/ssl/certs/ca-certificates.crt": "debian-ca.crt", "/etc/pki/tls/certs/ca-bundle.crt": "rhel-ca.crt"}


def isolate_ca_bundles(text: str, root: Path) -> str:
    """Point the installer's CA bundle checks at the test root instead of this machine's trust store."""
    for path, name in CA_BUNDLES.items():
        text = text.replace(path, str(root / name))
    return text


def npm_package(binary: bytes | None) -> bytes:
    """An npm tarball laid out like opencode-linux-x64's."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        files = [("package/package.json", b"{}")] + ([("package/bin/opencode", binary)] if binary is not None else [])
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """An isolated PATH: real core tools; a stubbed root check, apt-get and download; a CA bundle at the test root."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sh", "tar", "gzip", "sha256sum", "mkdir", "mv", "rm", "cp"):
        executable = shutil.which(name)
        if executable is None:
            pytest.skip(f"Installer test requires {name}")
        (bindir / name).symlink_to(executable)
    for name, script in {"curl": FAKE_CURL, "id": FAKE_ID, "apt-get": FAKE_PACKAGE_MANAGER}.items():
        (bindir / name).write_text(script)
        (bindir / name).chmod(0o755)
    shutil.copy(bindir / "curl", tmp_path / "curl")  # What the package manager installs.
    (tmp_path / CA_BUNDLES["/etc/ssl/certs/ca-certificates.crt"]).write_text("test certificate bundle")
    (tmp_path / "install_opencode.sh").write_text(isolate_ca_bundles(bootstrap.INSTALLER.read_text(), tmp_path))
    (tmp_path / URL.rsplit("/", 1)[1]).write_bytes(npm_package(BINARY))
    (tmp_path / "private").mkdir()
    return tmp_path, {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)}


def install(root: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    arguments = args or (str(root / "private/opencode"), "1.17.11", PIN)
    return subprocess.run(
        [str(root / "bin/sh"), str(root / "install_opencode.sh"), *arguments],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )


def test_installs_the_pinned_package_binary_and_nothing_else(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = install(root, env)
    assert result.returncode == 0, result.stderr
    assert (root / "download.log").read_text().splitlines() == [URL]
    assert [path.name for path in (root / "private").iterdir()] == ["opencode"]
    assert (root / "private/opencode").read_bytes() == BINARY
    assert not (root / "packages.log").exists()  # An image with the download tools stays unchanged.


def test_a_binary_that_misses_its_pin_is_never_installed(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / URL.rsplit("/", 1)[1]).write_bytes(npm_package(b"tampered"))
    result = install(root, env)
    assert result.returncode == 1
    assert f"not its pin {PIN}" in result.stderr
    assert not any((root / "private").iterdir())


def test_a_package_without_the_binary_installs_nothing(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / URL.rsplit("/", 1)[1]).write_bytes(npm_package(None))
    result = install(root, env)
    assert result.returncode == 1
    assert "has no package/bin/opencode" in result.stderr
    assert not any((root / "private").iterdir())


def test_a_failed_download_installs_nothing(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = install(root, env | {"TEST_DOWNLOAD_FAIL": "1"})
    assert result.returncode == 1
    assert f"could not download {URL}" in result.stderr
    assert not any((root / "private").iterdir())


@pytest.mark.parametrize(
    ("manager", "calls"),
    [
        ("apt-get", ["apt-get update", "apt-get install -y --no-install-recommends curl"]),
        ("apk", ["apk add --no-cache curl"]),
    ],
    ids=["apt-get", "apk"],
)
def test_as_root_missing_curl_is_installed_and_the_install_proceeds(
    sandbox: tuple[Path, dict[str, str]], manager: str, calls: list[str]
) -> None:
    root, env = sandbox
    (root / "bin/curl").unlink()
    (root / "bin/apt-get").rename(root / "bin" / manager)
    result = install(root, env)
    assert result.returncode == 0, result.stderr
    assert (root / "packages.log").read_text().splitlines() == calls
    assert (root / "download.log").read_text().splitlines() == [URL]
    assert (root / "private/opencode").read_bytes() == BINARY


def test_a_missing_ca_bundle_adds_ca_certificates(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / CA_BUNDLES["/etc/ssl/certs/ca-certificates.crt"]).unlink()
    result = install(root, env)
    assert result.returncode == 0, result.stderr
    assert (root / "packages.log").read_text().splitlines() == [
        "apt-get update",
        "apt-get install -y --no-install-recommends ca-certificates",
    ]
    assert (root / "private/opencode").read_bytes() == BINARY


@pytest.mark.parametrize(
    ("unavailable", "reason"),
    [("root", "requires root"), ("package-manager", "requires apt-get or apk")],
    ids=["non-root", "no-package-manager"],
)
def test_missing_curl_without_root_or_a_package_manager_names_the_remedy(
    sandbox: tuple[Path, dict[str, str]], unavailable: str, reason: str
) -> None:
    root, env = sandbox
    (root / "bin/curl").unlink()
    if unavailable == "root":
        env = env | {"TEST_UID": "1000"}
    else:
        (root / "bin/apt-get").unlink()
    result = install(root, env)
    assert result.returncode == 1
    assert "requires curl in the task image; preinstall it, or set binary_path" in result.stderr
    assert f"automatic installation {reason}" in result.stderr
    assert not (root / "packages.log").exists() and not (root / "download.log").exists()
    assert not any((root / "private").iterdir())


def test_sha256sum_stays_required_and_is_never_installed(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / "bin/sha256sum").unlink()
    (root / "bin/curl").unlink()
    result = install(root, env)
    assert result.returncode == 1
    assert "requires sha256sum in the task image; preinstall it, or set binary_path" in result.stderr
    assert not (root / "packages.log").exists()


@pytest.mark.parametrize(
    ("target", "version", "sha256", "message"),
    [
        ("private/opencode", "1.17.11", PIN, "absolute path"),
        (None, "latest", PIN, "exact version"),
        (None, "1.17.11", PIN.upper(), "hex digest"),
        (None, "1.17.11", PIN[:-1], "hex digest"),
    ],
    ids=["relative-target", "inexact-version", "uppercase-pin", "short-pin"],
)
def test_invalid_arguments_are_rejected_before_any_work(
    sandbox: tuple[Path, dict[str, str]], target: str | None, version: str, sha256: str, message: str
) -> None:
    root, env = sandbox
    result = install(root, env, target or str(root / "private/opencode"), version, sha256)
    assert result.returncode == 1
    assert message in result.stderr
    assert not (root / "download.log").exists()


class RecordingTransport:
    """The sandbox transport double: records each command and answers with the scripted result."""

    def __init__(self, failure: SandboxExecResult | None = None) -> None:
        self.calls: list[dict] = []
        self.uploads: list[tuple[Path, str]] = []
        self.failure = failure

    async def exec(self, *, command: str, user=None, cwd=None, timeout_s=None, env=None) -> SandboxExecResult:
        self.calls.append({"command": command, "user": user, "timeout_s": timeout_s})
        if self.failure is not None and command.startswith("sh -c "):
            return self.failure
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload(self, source: Path, destination: str) -> None:
        self.uploads.append((source, destination))


@pytest.mark.asyncio
async def test_without_a_host_copy_the_sandbox_installs_the_pin_as_root(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    transport = RecordingTransport()
    staged = await bootstrap.prepare_binary(
        BoundSandbox(transport, user="agent", workdir="/app", timeout_s=60),
        version="1.17.11",
        sha256=PIN,
        binary_path=None,
    )
    mkdir, installer, chmod = transport.calls
    assert mkdir["user"] == "agent" and staged.directory in mkdir["command"]
    assert (installer["user"], installer["timeout_s"]) == ("root", bootstrap.INSTALL_TIMEOUT_S)
    assert chmod == {"command": f"chmod 0755 {staged.directory}/opencode", "user": "root", "timeout_s": 30}
    assert not transport.uploads  # No host copy is involved.
    # The exact command the sandbox received runs the real installer.
    command = installer["command"].replace(f"{staged.directory}/opencode", str(root / "private/opencode"))
    command = isolate_ca_bundles(command, root)
    result = subprocess.run(
        [str(root / "bin/sh"), "-c", command], env=env, capture_output=True, text=True, errors="replace", timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert (root / "private/opencode").read_bytes() == BINARY


@pytest.mark.asyncio
async def test_a_failed_sandbox_install_surfaces_its_output_and_stages_nothing() -> None:
    message = "OpenCode installer: requires curl in the task image; preinstall it, or set binary_path"
    transport = RecordingTransport(failure=SandboxExecResult(stdout="", stderr=message, return_code=1))
    with pytest.raises(RuntimeError, match="installation failed \\(exit 1.*requires curl"):
        await bootstrap.prepare_binary(
            BoundSandbox(transport, user="agent", workdir="/app", timeout_s=60),
            version="1.17.11",
            sha256=PIN,
            binary_path=None,
        )
    assert not any(call["command"].startswith("chmod") for call in transport.calls)


@pytest.mark.asyncio
async def test_driver_without_a_host_copy_installs_before_running_opencode(tmp_path: Path) -> None:
    provider = Provider(child=False)
    driver = OpenCodeDriver(version="1.17.11", binary_sha256=PIN, context_limit=32000, max_turns=None)
    result = await driver.execute(context(tmp_path, provider))
    assert result.status == "completed"
    commands = [call["command"] for call in provider.calls]
    installer = next(index for index, command in enumerate(commands) if command.startswith("sh -c "))
    assert provider.calls[installer]["user"] == "root"
    assert installer < next(index for index, command in enumerate(commands) if "opencode --version" in command)
    assert not any(path.endswith("/opencode") for path in provider.files)
    launch = next(call for call in provider.calls if "opencode run " in call["command"])
    assert "steps" not in json.loads(launch["env"]["OPENCODE_CONFIG_CONTENT"])["agent"]["build"]
