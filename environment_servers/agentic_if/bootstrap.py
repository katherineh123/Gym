# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable harness-binary staging: a checked host copy, or the pinned package installed inside the sandbox."""

import asyncio
import hashlib
from pathlib import Path
from shlex import quote
from uuid import uuid4

from environment_servers.agentic_if.runtime import BoundSandbox
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxExecResult


INSTALLER = Path(__file__).with_name("install_opencode.sh")
# The download budget of upstream's in-sandbox installers (opencode_agent's sandbox_install_timeout_seconds).
INSTALL_TIMEOUT_S = 600


def binary_digest(path: Path) -> str:
    """Return the hex SHA-256 of a host file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class BinarySandbox:
    """Resolve harness commands without rewriting the user's prompt or changing HOME."""

    def __init__(self, sandbox: BoundSandbox, directory: str) -> None:
        self.sandbox, self.directory = sandbox, directory

    @property
    def transport(self) -> AsyncSandbox:
        """The staging transport for root-owned operations; harness commands use ``exec``."""
        return self.sandbox.transport

    async def upload(self, source: Path, destination: str) -> None:
        """Stage a host file in the sandbox."""
        await self.sandbox.upload(source, destination)

    async def download(self, source: str, destination: Path) -> None:
        """Collect a sandbox file on the host."""
        await self.sandbox.download(source, destination)

    async def exec(
        self,
        *,
        command: str,
        timeout_s: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxExecResult:
        """Run ``command`` as the task identity, with ``opencode`` resolving to the staged binary."""
        prefix = f'opencode() {{ {quote(self.directory + "/opencode")} "$@"; }}; '
        prefix += f"export PATH={quote(self.directory)}:$PATH; "
        env = {
            "XDG_CACHE_HOME": self.directory + "/cache",
            "XDG_CONFIG_HOME": self.directory + "/config",
            **(env or {}),
        }
        return await self.sandbox.exec(command=prefix + command, timeout_s=timeout_s, cwd=cwd, env=env)


async def install_opencode(transport: AsyncSandbox, target: str, *, version: str, sha256: str) -> None:
    """Download the pinned npm package inside the sandbox and install its checked ``bin/opencode`` at ``target``.

    ``install_opencode.sh`` runs as root, like the upload it replaces. It adds curl, tar, gzip or a CA bundle only when
    the image lacks them, as Gym's OpenCode agent does, and installs no binary if any step fails, so the next attempt
    simply tries again. Its message, with the installer's output, reaches the attempt's log.
    """
    command = " ".join(
        ["sh", "-c", quote(INSTALLER.read_text()), INSTALLER.name, quote(target), quote(version), quote(sha256)]
    )
    result = await transport.exec(command=command, user="root", timeout_s=INSTALL_TIMEOUT_S)
    if result.return_code != 0 or result.error_type:
        details = "\n".join(part for part in (result.stderr, result.stdout) if part)
        raise RuntimeError(
            f"OpenCode sandbox installation failed (exit {result.return_code}, "
            f"error_type={result.error_type}): {details[-4000:]}"
        )


async def prepare_binary(
    sandbox: BoundSandbox, *, version: str, sha256: str, binary_path: Path | None
) -> BinarySandbox:
    """Stage the pinned binary as root in a private directory, then execute under the bound task identity.

    A configured host ``binary_path`` is checked, then uploaded, and the task image always stays unchanged. Without
    one, the sandbox downloads and checks the pinned npm package itself; the image then stays unchanged unless the
    download needs curl, tar, gzip or a CA bundle that it lacks, which the installer adds as Gym's OpenCode agent does.
    No task files, account identity, or home ownership are modified.
    """
    if binary_path is not None and await asyncio.to_thread(binary_digest, binary_path) != sha256:
        raise ValueError("local OpenCode binary digest does not match its configured pin")
    directory = f"/tmp/nemo-gym-if-binary-{uuid4().hex}"
    result = await sandbox.exec(command=f"umask 077; mkdir -p {quote(directory)}", timeout_s=30)
    if result.return_code or result.error_type:
        raise RuntimeError("Could not prepare a private native harness directory")
    target = directory + "/opencode"
    if binary_path is None:
        await install_opencode(sandbox.transport, target, version=version, sha256=sha256)
    else:
        await sandbox.transport.upload(binary_path, target)
    result = await sandbox.transport.exec(command=f"chmod 0755 {quote(target)}", user="root", timeout_s=30)
    if result.return_code or result.error_type:
        raise RuntimeError("Could not stage the native harness executable")
    return BinarySandbox(sandbox, directory)
