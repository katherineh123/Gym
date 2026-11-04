# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ComplexConstraints rubric-judged instruction following.

ComplexConstraints (Mehta et al., "ComplexConstraints and Beyond: Expert Rubrics for RLVR", arXiv:2606.09118) gives
the policy one realistic request that carries many interdependent constraints and grades the reply against 10-40
expert-written rubric criteria. Grading follows the official harness (https://github.com/surge-ai/complex-constraints):

- One judge call per criterion. The judge sees the reply and that one criterion, never the prompt, and answers with a
  JSON object whose ``score`` is "0" or "1" (structured output). A criterion passes when the score is at least 1.
- A judge reply that does not parse is asked again, up to five calls. A criterion still unscored after that leaves the
  row out of the scores: the row goes to the failures sidecar, and a resume judges it again.
- ``reward`` is the task pass rate the leaderboard reports (1.0 only when every criterion passes); ``mean_criteria``
  is the fraction of criteria passed.

The official judge prompt and response schema carry no license, so they are not in this repository:
``benchmarks/complex_constraints/prepare.py`` downloads them at a pinned commit and this server reads them from
``judge_prompt_fpath``.
"""

import asyncio
import json
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from statistics import mean
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

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
from nemo_gym.verification import JudgeError, VerifierFixture, call_judge, record_parsed_judge_verdict
from resources_servers.complex_constraints.verifier_fixture import create_fixture_server, invoke_fixture


# The placeholders the official template uses; the reply is substituted first, then the criterion.
RESPONSE_PLACEHOLDER = "{{ rubric.response_to_grade }}"
CRITERION_PLACEHOLDER = "{{ rubric.criterion }}"
# The paper's constraint types, one per prompt; each is a metrics subset.
INSTRUCTION_TYPES = ("Conditional", "Implicit", "Multistep", "Negative")

_SERVER_DIR = Path(__file__).resolve().parent
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL)
_LEADING_THINK_RE = re.compile(r"^.*?</(think|thinking)>", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"\A```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\Z")


class ComplexConstraintsResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming = Field(
        description=(
            "Base Chat Completions request for the judge; the server sets `messages` and the structured-output "
            "`response_format`."
        )
    )
    judge_prompt_fpath: str = Field(
        description=(
            "JSON file with the official judge prompt template and response schema, written by "
            "benchmarks/complex_constraints/prepare.py. A relative path resolves against the working directory, then "
            "the repository root."
        )
    )
    judge_max_concurrency: int = Field(default=64, ge=1, description="Concurrent judge calls per server worker.")
    judge_max_attempts: int = Field(
        default=5,
        ge=1,
        description=(
            "Judge calls per criterion while the reply does not parse (the official scorer makes five). A criterion "
            "still unscored after that sends the row to the failures sidecar."
        ),
    )


class ComplexConstraintsRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = None
    instruction_type: Optional[str] = None
    rubrics: List[str] = Field(min_length=1)


class ComplexConstraintsVerifyRequest(ComplexConstraintsRunRequest, BaseVerifyRequest):
    pass


class CriterionGrade(BaseModel):
    passed: bool
    score: str
    rationale: str
    judge_attempts: int


class ComplexConstraintsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: str
    policy_incomplete_reason: Optional[str] = None
    mean_criteria: float
    criteria_passed: int
    criterion_grades: List[CriterionGrade]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text") if isinstance(part, dict) else getattr(part, "text", None) for part in content]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


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


def render_judge_prompt(template: str, *, response_text: str, criterion: str) -> str:
    """Fill the official template the way the official scorer does: plain substitution, reply first."""
    return template.replace(RESPONSE_PLACEHOLDER, response_text).replace(CRITERION_PLACEHOLDER, criterion)


def parse_criterion_verdict(text: Optional[str]) -> Optional[Tuple[Decimal, str]]:
    """Return the judge's (score, rationale), or None when the reply fails the official scorer's strict checks."""
    if text is None:
        return None
    cleaned = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", text.strip()))
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "score" not in payload or not isinstance(payload.get("rationale"), str):
        return None
    raw_score = payload["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float, str)):
        return None
    try:
        score = Decimal(str(raw_score).strip())
    except InvalidOperation:
        return None
    if not score.is_finite():
        return None
    return score, payload["rationale"]


def criterion_passes(score: Decimal) -> bool:
    """The official pass rule: the score rounded half-up to four places is at least 1."""
    return score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) >= 1


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return _SERVER_DIR.parents[1] / candidate


def _score(rollout: Dict[str, Any]) -> Dict[str, float]:
    return {"reward": float(rollout.get("reward") or 0.0), "mean_criteria": float(rollout.get("mean_criteria") or 0.0)}


class ComplexConstraintsResourcesServer(SimpleResourcesServer):
    config: ComplexConstraintsResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        prompt_path = _resolve_path(self.config.judge_prompt_fpath)
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"ComplexConstraints judge prompt not found at {prompt_path}. Run `gym eval prepare --benchmark "
                "complex_constraints` to download it from the official repository."
            )
        judge = json.loads(prompt_path.read_text(encoding="utf-8"))
        self._judge_template: str = judge["judge_prompt_template"]
        self._response_format = {"type": "json_schema", "json_schema": judge["response_schema"]}
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        return super().model_post_init(context)

    async def _judge(self, prompt: str) -> Optional[str]:
        params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
            {
                **self.config.judge_chat_completions_create_params.model_dump(exclude_unset=True),
                "messages": [{"role": "user", "content": prompt}],
                "response_format": self._response_format,
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

    async def _grade_criterion(self, answer: str, criterion: str) -> Optional[CriterionGrade]:
        prompt = render_judge_prompt(self._judge_template, response_text=answer, criterion=criterion)
        for attempt in range(1, self.config.judge_max_attempts + 1):
            verdict = parse_criterion_verdict(await self._judge(prompt))
            if verdict is not None:
                record_parsed_judge_verdict()
                score, rationale = verdict
                return CriterionGrade(
                    passed=criterion_passes(score), score=str(score), rationale=rationale, judge_attempts=attempt
                )
        return None

    async def verify(self, body: ComplexConstraintsVerifyRequest) -> ComplexConstraintsVerifyResponse:
        answer = extract_answer(body.response)
        details = body.response.incomplete_details
        result: Dict[str, Any] = dict(
            extracted_answer=answer, policy_incomplete_reason=details.reason if details is not None else None
        )
        # No visible reply satisfies no criterion; a reply cut off by the output cap is graded as written.
        if not answer:
            return ComplexConstraintsVerifyResponse(
                **body.model_dump(), reward=0.0, mean_criteria=0.0, criteria_passed=0, criterion_grades=[], **result
            )

        async def grade() -> ComplexConstraintsVerifyResponse:
            # return_exceptions lets every criterion settle before a judge failure propagates.
            outcomes = await asyncio.gather(
                *(self._grade_criterion(answer, criterion) for criterion in body.rubrics), return_exceptions=True
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            unscored = [index for index, grade in enumerate(outcomes) if grade is None]
            if unscored:
                # The official scorer leaves such a row out of the task pass rate; the sidecar does the same.
                raise JudgeError(
                    f"judge reply did not parse after {self.config.judge_max_attempts} attempts for criteria "
                    f"{unscored} of task {body.task_id}"
                )
            passed = sum(grade.passed for grade in outcomes)
            return ComplexConstraintsVerifyResponse(
                **body.model_dump(),
                reward=1.0 if passed == len(outcomes) else 0.0,
                mean_criteria=passed / len(outcomes),
                criteria_passed=passed,
                criterion_grades=outcomes,
                **result,
            )

        return await observe_llm_judge_grading(grade)

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics = compute_pass_majority_metrics(tasks, score_fn=_score)[0]
        metrics.update(compute_subset_metrics(tasks, subset_key="instruction_type", score_fn=_score))
        rollouts = [rollout for task in tasks for rollout in task]
        if rollouts:
            metrics["empty_answer_rate"] = mean(not r.get("extracted_answer") for r in rollouts)
            for reason in ("max_output_tokens", "content_filter"):
                metrics[f"policy_{reason}_rate"] = mean(r.get("policy_incomplete_reason") == reason for r in rollouts)
            attempts = [g["judge_attempts"] for r in rollouts for g in r.get("criterion_grades") or []]
            if attempts:
                metrics["judge_retry_rate"] = mean(attempt > 1 for attempt in attempts)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key_metrics = highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["reward", "mean_criteria"])
        for instruction_type in INSTRUCTION_TYPES:
            key_metrics.update(
                highest_k_metrics(agent_metrics, f"{instruction_type}/pass@1[avg-of-{{k}}]", score_names=["reward"])
            )
        for name in ("empty_answer_rate", "policy_max_output_tokens_rate", "policy_content_filter_rate"):
            if name in agent_metrics:
                key_metrics[name] = agent_metrics[name]
        return key_metrics


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=ComplexConstraintsVerifyRequest,
    cases_path=_SERVER_DIR / "tests" / "verifier_cases.jsonl",
    invoke=invoke_fixture,
)


if __name__ == "__main__":
    ComplexConstraintsResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = ComplexConstraintsResourcesServer.run_webserver()  # noqa: F401
