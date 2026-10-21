#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install bin/opencode of the pinned opencode-linux-x64 npm package at TARGET inside the task sandbox, after checking
# it against its SHA-256 pin. The Environment Server runs this as root while it prepares an attempt
# (bootstrap.install_opencode). Task files and the task's PATH are left alone. Image packages are too, unless the
# download needs curl, tar, gzip or a CA bundle that the image lacks; then this adds them as Gym's OpenCode agent does
# (responses_api_agents/opencode_agent/install_opencode_runtime.sh). A failed run installs no binary.
# Usage: install_opencode.sh TARGET VERSION SHA256
set -eu

fail() {
  printf 'OpenCode installer: %s\n' "$*" >&2
  exit 1
}

[ "$#" -eq 3 ] || fail 'usage: install_opencode.sh TARGET VERSION SHA256'
target=$1
version=$2
sha256=$3
case "$target" in
  /*) ;;
  *) fail "TARGET must be an absolute path, got '$target'" ;;
esac
case "$version" in
  '' | *[!0-9.]*) fail "VERSION must be an exact version, got '$version'" ;;
esac
case "$sha256" in
  *[!0-9a-f]*) fail 'SHA256 must be a lowercase hex digest' ;;
esac
[ "${#sha256}" -eq 64 ] || fail 'SHA256 must be a lowercase hex digest'
remedy='preinstall it, or set binary_path to upload a host copy instead'
command -v sha256sum >/dev/null 2>&1 || fail "requires sha256sum in the task image; $remedy"
missing=
for tool in curl tar gzip; do
  command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
[ -s /etc/ssl/certs/ca-certificates.crt ] || [ -s /etc/pki/tls/certs/ca-bundle.crt ] \
  || missing="$missing ca-certificates"
if [ -n "$missing" ]; then
  [ "$(id -u)" = 0 ] || fail "requires$missing in the task image; $remedy (automatic installation requires root)"
  # $missing is deliberately unquoted: it is a list of package names.
  if command -v apk >/dev/null 2>&1; then
    apk add --no-cache $missing || fail "could not install$missing with apk"
  elif command -v apt-get >/dev/null 2>&1; then
    { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $missing; } \
      || fail "could not install$missing with apt-get"
  else
    fail "requires$missing in the task image; $remedy (automatic installation requires apt-get or apk)"
  fi
fi

scratch="$target.download"
rm -rf "$scratch"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
url="https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-$version.tgz"
curl -fsSL --retry 3 -o "$scratch/package.tgz" "$url" || fail "could not download $url"
# Written by this shell rather than unpacked, so the binary is owned by the installing user (root), not the archive's.
tar -xzOf "$scratch/package.tgz" package/bin/opencode >"$scratch/opencode" \
  || fail "opencode-linux-x64@$version has no package/bin/opencode"
actual=$(sha256sum "$scratch/opencode")
actual=${actual%% *}
[ "$actual" = "$sha256" ] \
  || fail "bin/opencode of opencode-linux-x64@$version has SHA-256 $actual, not its pin $sha256"
mv "$scratch/opencode" "$target"
