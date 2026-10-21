# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native tool selections; never executable config."""

from resources_servers.agentic_if.task_variants.catalog import NATIVE_TOOLS


def configurable_tools(tool_names: dict[str, str] | None) -> frozenset[str]:
    """Omission retains native tools; a mapping selects exactly its installed IDs."""
    native = NATIVE_TOOLS
    if tool_names is None:
        return native
    if not set(tool_names) <= native:
        raise ValueError(f"unsupported tool for opencode: {sorted(set(tool_names) - native)}")
    return frozenset(tool_names)
