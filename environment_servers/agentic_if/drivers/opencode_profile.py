# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool-profile records of the ``if-profile.js`` OpenCode plugin, and the profile hash they carry.

The plugin applies each tool's configured argument schema inside OpenCode and records the model's configured
arguments with every call; the model gateway applies the descriptions. These helpers read back what the model wrote
and saw.
"""

import hashlib
import json
import re
from typing import Any, Mapping


def profile_sha256(profile: Mapping[str, Any]) -> str:
    """SHA-256 of the profile in canonical JSON (sorted keys, no whitespace); names the exact configuration."""
    body = {k: v for k, v in profile.items() if k != "profile_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


PROFILE_METADATA_KEY = "nemo_gym_profile"
PROFILE_STASH_KEY = "__nemo_gym_profile"


def model_visible_tool_input(state: Mapping[str, Any]) -> Any:
    """The arguments of an exported OpenCode tool part as the model wrote them.

    OpenCode stores a tool part's input after the plugin mapped it to the native form; the plugin records the model's
    configured arguments (extra arguments included) in the part's metadata. Without that record (no profile, or a tool
    that failed before the plugin's after-hook), the stored input is what the model sent.
    """
    metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
    record = metadata.get(PROFILE_METADATA_KEY)
    if isinstance(record, Mapping) and "configured_args" in record:
        return record["configured_args"]
    stash = _stash(state)
    if "configured_args" in stash:
        return stash["configured_args"]
    return state.get("input") or {}


def model_visible_tool_output(state: Mapping[str, Any]) -> Any:
    """The result of an exported OpenCode tool part as the model saw it.

    That is the output or, for a failed call, the error text, with native argument names renamed to the profile's as
    the plugin showed them (if-profile.js renameNative).
    """
    stamp = (state.get("metadata") or {}).get(PROFILE_METADATA_KEY) or {}
    if state.get("status") == "error" and isinstance(stamp.get("model_error"), str):
        return stamp["model_error"]
    if state.get("output") is not None:
        return state["output"]
    error = state.get("error")
    names = _stash(state).get("names")
    if isinstance(error, str) and isinstance(names, Mapping):
        for native, configured in names.items():
            if native != configured:
                error = re.sub(rf"\b{re.escape(native)}\b", str(configured), error)
    return error


def _stash(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """The plugin's record inside a reshaped call's stored input (if-profile.js STASH_KEY).

    It survives a call that fails inside the tool.
    """
    tool_input = state.get("input") if isinstance(state.get("input"), Mapping) else {}
    stash = tool_input.get(PROFILE_STASH_KEY)
    return stash if isinstance(stash, Mapping) else {}
