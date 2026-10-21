# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native tool contracts describe the pinned binary, not hand-maintained guesses."""

from pathlib import Path

import yaml

from resources_servers.agentic_if.task_variants.catalog import NATIVE_ARGUMENTS, NATIVE_SCHEMA_SNAPSHOT, NATIVE_TOOLS


def test_all_pinned_opencode_tools_have_complete_schemas_and_derived_arguments():
    config = yaml.safe_load((Path(__file__).parents[2] / "configs" / "agentic_if.yaml").read_text())
    pinned = config["agentic_if_environment"]["environment_servers"]["agentic_if"]["opencode"]
    snapshot = NATIVE_SCHEMA_SNAPSHOT
    assert (snapshot["version"], snapshot["binary_sha256"]) == (pinned["version"], pinned["binary_sha256"])
    schemas = snapshot["schemas"]
    assert set(schemas) == NATIVE_TOOLS == set(NATIVE_ARGUMENTS)
    assert all(schema["type"] == "object" for schema in schemas.values())
    assert schemas["apply_patch"]["properties"]["patchText"]["type"] == "string"
    assert "background" not in schemas["task"]["properties"]
    assert schemas["read"]["properties"]["offset"]["minimum"] == 0
