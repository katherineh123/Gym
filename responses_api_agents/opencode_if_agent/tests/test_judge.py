# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest


INSTRUCTIONS = [
    {
        "id": "finish",
        "taxonomy": ["IF-FORMAT"],
        "instruction_text": "End with FINISH.",
        "placement": {"surface": "user_prompt", "position": "end"},
        "rubric": "The final answer ends with FINISH.",
    }
]
EVIDENCE = {"messages": [{"role": "assistant", "content": "Done. FINISH"}], "complete": True}


class RemoteResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    async def json(self):
        return self.payload

    def release(self):
        pass


def completion(results, reason="stop"):
    return {"choices": [{"finish_reason": reason, "message": {"content": json.dumps({"judgments": results})}}]}


def judgment(status="pass", **extra):
    return {"instruction_id": "finish", "status": status, "rationale": "The answer ends correctly.", **extra}


@pytest.mark.asyncio
async def test_judge_uses_hosted_endpoint_and_returns_separate_structured_verdict(monkeypatch):
    from nemo_gym.task_variants import judge as module

    outgoing = []

    async def remote(method, url, **kwargs):
        outgoing.append((method, url, kwargs))
        return RemoteResponse(completion([judgment()]))

    monkeypatch.setattr(module, "request", remote)
    evaluator = module.RubricJudge(module.JudgeConfig(api_key="test-key"))
    result = await evaluator.grade(INSTRUCTIONS, EVIDENCE)
    assert result["status"] == "completed"
    assert result["judgments"][0]["status"] == "pass"
    assert "reward" not in result
    method, url, kwargs = outgoing[0]
    assert (method, url) == ("POST", "https://inference-api.nvidia.com/v1/chat/completions")
    assert kwargs["json"]["model"] == "nvidia/zai-org/glm-5.3"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert "The final answer ends with FINISH." in kwargs["json"]["messages"][1]["content"]
    assert "test-key" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        completion([judgment()], reason="length"),
        completion([]),
        completion([judgment(instruction_id="wrong")]),
        completion([judgment(), judgment()]),
        completion([judgment(status="maybe")]),
        {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
    ],
)
async def test_invalid_or_incomplete_judge_output_never_becomes_a_pass(monkeypatch, payload):
    from nemo_gym.task_variants import judge as module

    async def remote(*args, **kwargs):
        return RemoteResponse(payload)

    monkeypatch.setattr(module, "request", remote)
    result = await module.RubricJudge(module.JudgeConfig(api_key="test-key")).grade(INSTRUCTIONS, EVIDENCE)
    assert result["status"] == "error"
    assert result["judgments"] == []


@pytest.mark.asyncio
async def test_empty_constraints_skip_judging_and_missing_evidence_is_not_pass(monkeypatch):
    from nemo_gym.task_variants import judge as module

    async def no_request(*args, **kwargs):
        pytest.fail("no external call should be needed")

    monkeypatch.setattr(module, "request", no_request)
    evaluator = module.RubricJudge(module.JudgeConfig())
    assert (await evaluator.grade([], EVIDENCE))["status"] == "not_requested"
    assert (await evaluator.grade(INSTRUCTIONS, {"complete": False}))["status"] == "error"
    assert (await evaluator.grade(INSTRUCTIONS, EVIDENCE))["error"] == "missing_judge_api_key"


@pytest.mark.asyncio
async def test_oversize_evidence_is_rejected_not_silently_truncated():
    from nemo_gym.task_variants.judge import JudgeConfig, RubricJudge

    evaluator = RubricJudge(JudgeConfig(api_key="test-key", max_input_chars=100))
    result = await evaluator.grade(INSTRUCTIONS, EVIDENCE)
    assert result["error"] == "judge_input_too_large"


@pytest.mark.asyncio
async def test_network_error_does_not_leak_credentials(monkeypatch):
    from nemo_gym.task_variants import judge as module

    async def remote(*args, **kwargs):
        raise TimeoutError("some exception containing test-key")

    monkeypatch.setattr(module, "request", remote)
    result = await module.RubricJudge(module.JudgeConfig(api_key="test-key")).grade(INSTRUCTIONS, EVIDENCE)
    assert result["status"] == "error"
    assert "test-key" not in json.dumps(result)


@pytest.mark.asyncio
async def test_insufficient_evidence_is_a_per_instruction_error(monkeypatch):
    from nemo_gym.task_variants import judge as module

    async def remote(*args, **kwargs):
        return RemoteResponse(completion([judgment(status="error", rationale="Patch evidence unavailable.")]))

    monkeypatch.setattr(module, "request", remote)
    result = await module.RubricJudge(module.JudgeConfig(api_key="test-key")).grade(INSTRUCTIONS, EVIDENCE)
    assert result["status"] == "error"
    assert result["judgments"][0]["status"] == "error"
