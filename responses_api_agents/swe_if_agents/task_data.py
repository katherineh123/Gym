# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy IF uses the same SWE task contract; constraint metadata is additional data."""

from responses_api_agents.swe_agents.task_data import TaskData as SWETaskData


class TaskData(SWETaskData):
    """Retain the upstream task schema without duplicating or weakening its fields."""
