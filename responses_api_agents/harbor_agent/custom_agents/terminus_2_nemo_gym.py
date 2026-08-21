# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Any, override

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM, LLMBackend
from harbor.models.agent.context import AgentContext

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM
from responses_api_agents.harbor_agent.custom_envs.singularity.singularity import MemoryLimitExceededError


class Terminus2NemoGym(Terminus2):
    """Terminus2 variant that uses a NeMo Gym model server-compatible BaseLLM."""

    @staticmethod
    def name() -> str:
        return "terminus-2-nemo-gym"

    def __init__(
        self,
        *args: Any,
        llm: BaseLLM | None = None,
        responses_create_params: dict[str, Any] | None = None,
        nemo_model_server_timeout_sec: float = 120.0,
        **kwargs: Any,
    ) -> None:
        self._provided_llm = llm
        self._responses_create_params = responses_create_params
        self._nemo_model_server_timeout_sec = nemo_model_server_timeout_sec
        super().__init__(*args, **kwargs)

    @override
    def _init_llm(
        self,
        llm_backend: LLMBackend | str,
        model_name: str,
        temperature: float | None,
        collect_rollout_details: bool,
        llm_kwargs: dict[str, Any] | None,
        api_base: str | None,
        session_id: str | None,
        max_thinking_tokens: int | None,
        reasoning_effort: str | None,
        model_info: dict[str, Any] | None,
        use_responses_api: bool,
    ) -> BaseLLM:
        """Create the NeMo Gym LLM through current Terminus-2's LLM hook."""
        if self._provided_llm is not None:
            return self._provided_llm
        if api_base is None:
            raise ValueError("api_base is required for Terminus2NemoGym when llm is not provided")

        return NemoGymLLM(
            model_name=model_name,
            api_base=api_base,
            collect_rollout_details=collect_rollout_details,
            model_info=model_info,
            responses_create_params=self._responses_create_params,
            timeout_sec=self._nemo_model_server_timeout_sec,
        )

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Override run() to gracefully handle agent errors.

        The parent's run() has a finally block that saves rollout_details and
        dumps the trajectory before any exception propagates. By catching
        exceptions here, we let Harbor's trial system proceed normally with the
        verifier — returning the agent's conversation history from all completed
        turns (reward will be 0 for incomplete work) instead of crashing the
        entire rollout batch.
        """
        self._memory_limit_exceeded = False
        try:
            await super().run(instruction, environment, context)
        except MemoryLimitExceededError as e:
            self._memory_limit_exceeded = True
            self.logger.info(f"Agent error: {type(e).__name__}: {e}. Returning history from completed turns.")
        except Exception as e:
            self.logger.info(f"Agent error: {type(e).__name__}: {e}. Returning history from completed turns.")
        finally:
            self._attach_routed_experts_to_trajectory()
            self._write_agent_error_flags()

    def _attach_routed_experts_to_trajectory(self) -> None:
        """Add NeMo Gym routed experts to Harbor metrics.extra before Gym converts the trajectory."""
        llm = getattr(self, "_llm", None)
        if not isinstance(llm, NemoGymLLM):
            return

        modified = False
        for step in getattr(self, "_trajectory_steps", []):
            if getattr(step, "source", None) != "agent":
                continue
            metrics = getattr(step, "metrics", None)
            if metrics is None:
                continue

            routed_experts = llm.pop_routed_experts_for_rollout_details(
                getattr(metrics, "prompt_token_ids", None),
                getattr(metrics, "completion_token_ids", None),
                getattr(metrics, "logprobs", None),
            )
            if routed_experts is None:
                continue
            metrics_extra = metrics.extra or {}
            metrics_extra["routed_experts"] = routed_experts
            metrics.extra = metrics_extra
            modified = True

        if modified:
            self._dump_trajectory()

    def _write_agent_error_flags(self) -> None:
        """Write agent error flags to disk for app.py to pick up."""
        try:
            flags: dict[str, bool] = {
                "memory_limit_exceeded": self._memory_limit_exceeded,
            }
            llm = getattr(self, "_llm", None)
            if llm and isinstance(llm, NemoGymLLM):
                flags["context_length_exceeded"] = llm.context_length_exceeded
            (self.logs_dir / "agent_error_flags.json").write_text(json.dumps(flags))
        except Exception:
            pass  # Don't let flag-writing failures break the agent
