# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AdvancedIF rubric-judged instruction following.

AdvancedIF (He et al., "AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM
Instruction Following", arXiv:2511.10507) asks the policy for the final assistant turn of a conversation and has an
LLM judge check that reply against expert-written rubrics. The protocol follows the paper and its official scorer
(https://github.com/facebookresearch/AdvancedIF):

- One judge call per row carries every rubric. System-steerability rows use the system-steer prompt with its
  few-shot examples; the other categories use the user-instruction prompt.
- The reward is 1.0 only when the judge's ``SATISFIED_ALL_REQUIREMENTS`` field says yes.
- The paper's headline is the unweighted mean of the three category pass rates (``category_mean/...``).

The official judge prompts are CC BY-NC 4.0, so they are not in this repository: ``benchmarks/advanced_if/prepare.py``
downloads them at a pinned commit and this server reads them from ``judge_prompts_fpath``.
"""

import asyncio
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponse
from nemo_gym.metrics import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    observe_llm_judge_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture, call_judge, record_parsed_judge_verdict
from resources_servers.advanced_if.verifier_fixture import create_fixture_server, invoke_fixture


SYSTEM_STEERABILITY = "system_steerability"
CATEGORIES = ("complex_if", "carried_context", SYSTEM_STEERABILITY)

_SERVER_DIR = Path(__file__).resolve().parent
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL)
_LEADING_THINK_RE = re.compile(r"^.*?</(think|thinking)>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class AdvancedIfResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming = Field(
        description="Base Chat Completions request for the judge; `messages` is replaced by the rendered prompt."
    )
    judge_prompts_fpath: str = Field(
        description=(
            "JSON file with the official judge prompts, written by benchmarks/advanced_if/prepare.py. A relative path "
            "resolves against the working directory, then the repository root."
        )
    )
    judge_max_concurrency: int = Field(default=64, ge=1, description="Concurrent judge calls per server worker.")
    judge_max_attempts: int = Field(
        default=3,
        ge=1,
        description="Judge calls per row while the reply is not the JSON verdict object; then the row scores 0.",
    )


class AdvancedIfRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[int] = None
    category: str
    rubrics: List[str]


class AdvancedIfVerifyRequest(AdvancedIfRunRequest, BaseVerifyRequest):
    pass


class AdvancedIfVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    category: str
    extracted_answer: str
    policy_incomplete_reason: Optional[str] = None
    judge_output: Optional[str] = None
    judge_attempts: int
    judge_parse_failed: bool
    satisfied_all_requirements: Optional[str] = None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text") if isinstance(part, dict) else getattr(part, "text", None) for part in content]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def conversation_messages(input_items: Any) -> List[Dict[str, str]]:
    """Return the role and text of each message in a Responses `input`."""
    if isinstance(input_items, str):
        return [{"role": "user", "content": input_items}]
    messages = []
    for item in input_items:
        item = item if isinstance(item, dict) else item.model_dump()
        if item.get("role") is not None:
            messages.append({"role": item["role"], "content": _content_text(item.get("content"))})
    return messages


def render_judge_prompt(
    prompts: Dict[str, str], *, category: str, messages: List[Dict[str, str]], response_text: str, rubrics: List[str]
) -> str:
    """Fill the official template for the row's category the way the official scorer does."""
    last_user_turn = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    rubrics_text = json.dumps(rubrics, indent=4)
    if category == SYSTEM_STEERABILITY:
        system_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        return prompts["system_steer_judge_prompt"].format(
            few_shot_examples=prompts["system_steer_few_shot_examples"],
            system_prompt=system_prompt,
            user_prompt_last_turn=last_user_turn,
            response_text=response_text,
            rubrics_text=rubrics_text,
        )

    # Every message before the final user turn, numbered by exchange.
    lines, turn = [], 1
    for message in messages[:-1]:
        lines.append(f"{message['role']} [{turn}]: {message['content']}")
        if message["role"] == "assistant":
            turn += 1
    return prompts["if_judge_prompt"].format(
        full_conversation="\n".join(lines),
        user_prompt_last_turn=last_user_turn,
        response_text=response_text,
        rubrics_text=rubrics_text,
    )


def parse_judge_reply(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the judge's verdict object, or None when the reply is not one."""
    if not text:
        return None
    text = text.strip()
    fenced = _CODE_FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(verdict, dict) or not isinstance(verdict.get("SATISFIED_ALL_REQUIREMENTS"), str):
        return None
    return verdict


def extract_answer(response: NeMoGymResponse) -> str:
    """Return the final assistant message text without reasoning; an unclosed think block counts as no answer."""
    text = ""
    for item in reversed(response.output):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            text = _content_text(item.content)
            break
    if re.search(r"<think(ing)?>", text) and not re.search(r"</think(ing)?>", text):
        return ""
    return _LEADING_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", text)).strip()


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return _SERVER_DIR.parents[1] / candidate


def _score(rollout: Dict[str, Any]) -> Dict[str, float]:
    return {"reward": float(rollout.get("reward") or 0.0)}


class AdvancedIfResourcesServer(SimpleResourcesServer):
    config: AdvancedIfResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        prompts_path = _resolve_path(self.config.judge_prompts_fpath)
        if not prompts_path.exists():
            raise FileNotFoundError(
                f"AdvancedIF judge prompts not found at {prompts_path}. Run `gym eval prepare --benchmark advanced_if` "
                "to download them from the official repository."
            )
        self._judge_prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        return super().model_post_init(context)

    async def _judge(self, prompt: str) -> Optional[str]:
        params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
            {
                **self.config.judge_chat_completions_create_params.model_dump(exclude_unset=True),
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        async with self._judge_semaphore:
            completion = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=params,
                response_model=NeMoGymChatCompletion,
            )
        return completion.choices[0].message.content if completion.choices else None

    async def verify(self, body: AdvancedIfVerifyRequest) -> AdvancedIfVerifyResponse:
        answer = extract_answer(body.response)
        details = body.response.incomplete_details
        incomplete_reason = details.reason if details is not None else None
        result: Dict[str, Any] = dict(
            extracted_answer=answer,
            policy_incomplete_reason=incomplete_reason,
            judge_attempts=0,
            judge_parse_failed=False,
        )
        # A reply cut off by the output cap or with no visible text fails every rubric without a judge call.
        if incomplete_reason == "max_output_tokens" or not answer:
            return AdvancedIfVerifyResponse(**body.model_dump(), reward=0.0, **result)

        prompt = render_judge_prompt(
            self._judge_prompts,
            category=body.category,
            messages=conversation_messages(body.responses_create_params.input),
            response_text=answer,
            rubrics=body.rubrics,
        )

        async def grade() -> AdvancedIfVerifyResponse:
            verdict = None
            for attempt in range(1, self.config.judge_max_attempts + 1):
                judge_output = await self._judge(prompt)
                result.update(judge_attempts=attempt, judge_output=judge_output)
                verdict = parse_judge_reply(judge_output)
                if verdict is not None:
                    record_parsed_judge_verdict()
                    break
            if verdict is None:
                # The official scorer counts an unusable judge reply as a failed row.
                result["judge_parse_failed"] = True
                return AdvancedIfVerifyResponse(**body.model_dump(), reward=0.0, **result)

            # Only the overall field is scored: per-rubric answers are free text, often with no yes or no in them.
            satisfied = verdict["SATISFIED_ALL_REQUIREMENTS"]
            result["satisfied_all_requirements"] = satisfied
            reward = 1.0 if satisfied.strip().lower() == "yes" else 0.0
            return AdvancedIfVerifyResponse(**body.model_dump(), reward=reward, **result)

        return await observe_llm_judge_grading(grade)

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics = compute_pass_majority_metrics(tasks, score_fn=_score)[0]
        subset_metrics = compute_subset_metrics(tasks, subset_key="category", score_fn=_score)
        metrics.update(subset_metrics)

        # The paper's overall score: the unweighted mean of the three category scores.
        for key in subset_metrics:
            category, _, metric = key.partition("/")
            if category != CATEGORIES[0] or not metric.startswith("pass@"):
                continue
            values = [subset_metrics.get(f"{name}/{metric}") for name in CATEGORIES]
            if all(isinstance(value, (int, float)) for value in values):
                metrics[f"category_mean/{metric}"] = mean(values)

        rollouts = [rollout for task in tasks for rollout in task]
        if rollouts:
            metrics["judge_parse_failed_rate"] = mean(bool(r.get("judge_parse_failed")) for r in rollouts)
            metrics["empty_answer_rate"] = mean(not r.get("extracted_answer") for r in rollouts)
            for reason in ("max_output_tokens", "content_filter"):
                metrics[f"policy_{reason}_rate"] = mean(r.get("policy_incomplete_reason") == reason for r in rollouts)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key_metrics = highest_k_metrics(agent_metrics, "category_mean/pass@1[avg-of-{k}]", score_names=["reward"])
        key_metrics.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["reward"]))
        for category in CATEGORIES:
            key_metrics.update(
                highest_k_metrics(agent_metrics, f"{category}/pass@1[avg-of-{{k}}]", score_names=["reward"])
            )
        for name in (
            "judge_parse_failed_rate",
            "empty_answer_rate",
            "policy_max_output_tokens_rate",
            "policy_content_filter_rate",
        ):
            if name in agent_metrics:
                key_metrics[name] = agent_metrics[name]
        return key_metrics


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=AdvancedIfVerifyRequest,
    cases_path=_SERVER_DIR / "tests" / "verifier_cases.jsonl",
    invoke=invoke_fixture,
)


if __name__ == "__main__":
    AdvancedIfResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = AdvancedIfResourcesServer.run_webserver()  # noqa: F401
