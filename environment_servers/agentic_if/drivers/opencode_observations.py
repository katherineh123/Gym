# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correlate native execution with the actor's actual wire-visible observations."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy

from pydantic import JsonValue

from environment_servers.agentic_if.drivers.opencode_context import generated_content as _generated_content
from environment_servers.agentic_if.drivers.opencode_context import output_digest
from nemo_gym.openai_utils import NeMoGymResponseOutputItem
from nemo_gym.responses_converter import ResponsesConverter
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence, SessionStep
from resources_servers.agentic_if.task_variants.builder import content_text
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec


def prefix_model_calls(continuation: ContinuationSpec | None) -> dict[str, dict[str, JsonValue]]:
    """Read witnessed rejected calls from the frozen root, never infer their arguments."""
    result = {}
    if continuation is None:
        return result
    for message in continuation.prefix_export["messages"]:
        for part in message.get("parts", []):
            if part.get("type") != "tool":
                continue
            metadata = part.get("state", {}).get("metadata") or {}
            receipt = metadata.get("nemo_gym_model_call")
            if receipt is None:
                if (
                    part.get("tool") == "invalid"
                    or (metadata.get("nemo_gym_profile") or {}).get("dispatch_rejected") is True
                ):
                    raise ValueError("rejected prefix requires captured call evidence")
                continue
            identity = part.get("callID")
            function = receipt.get("function", {}) if isinstance(receipt, dict) else {}
            names = receipt.get("available_names") if isinstance(receipt, dict) else None
            if (
                not isinstance(identity, str)
                or not identity
                or identity in result
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
                or not isinstance(function.get("arguments"), str)
                or not isinstance(names, list)
                or any(not isinstance(name, str) or not name for name in names)
                or len(names) != len(set(names))
            ):
                raise ValueError("invalid or duplicate rejected-call evidence in prefix")
            result[identity] = deepcopy(receipt)
    return result


def preserve_rejected_calls(
    exports: dict[str, dict], observations: Sequence[Mapping[str, JsonValue]], continuation: ContinuationSpec | None
) -> None:
    """Attach captured model calls to enriched exports, not immutable native exports.

    Stock OpenCode replaces invalid calls with recovery inputs; alias-gate
    rejections keep native names. Keep the exact model call and that request's
    tool inventory for both, including malformed arguments and child sessions.
    """
    captured = {}
    for observation in observations:
        if "response" not in observation:
            continue
        for identity, function in _calls(observation).items():
            key = (observation["session_id"], identity)
            if key in captured:
                raise ValueError("ambiguous model call evidence")
            names = sorted(tool["function"]["name"] for tool in observation["request"].get("tools", []))
            captured[key] = {"function": deepcopy(function), "available_names": names}
    historical = prefix_model_calls(continuation)
    roots = [key for key, document in exports.items() if document.get("info", {}).get("parentID") is None]
    if historical and len(roots) != 1:
        raise ValueError("rejected-call evidence requires exactly one root session")
    for session, document in exports.items():
        for message in document.get("messages", []):
            for part in message.get("parts", []):
                if part.get("type") != "tool":
                    continue
                stamp = (part.get("state", {}).get("metadata") or {}).get("nemo_gym_profile") or {}
                if part.get("tool") != "invalid" and stamp.get("dispatch_rejected") is not True:
                    continue
                receipt = captured.get((session, part.get("callID")))
                if receipt is None and session in roots:
                    receipt = historical.get(part.get("callID"))
                if receipt is None:
                    raise ValueError("unobserved rejected model call in native export")
                part["state"].setdefault("metadata", {})["nemo_gym_model_call"] = deepcopy(receipt)


def _calls(observation: Mapping[str, JsonValue]) -> dict[str, dict[str, JsonValue]]:
    response = observation.get("response")
    if not isinstance(response, dict):
        raise ValueError("missing model response")
    choices = response.get("choices", [])
    if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
        raise ValueError("ambiguous model response")
    calls = choices[0]["message"].get("tool_calls") or []
    indexed = {}
    for call in calls:
        identity, function = call.get("id"), call.get("function")
        if not isinstance(identity, str) or not identity or identity in indexed or not isinstance(function, dict):
            raise ValueError("missing or duplicate model call identity")
        if not isinstance(function.get("name"), str) or not isinstance(function.get("arguments"), str):
            raise ValueError("invalid model call")
        indexed[identity] = function
    return indexed


def _response_output(observation: Mapping[str, JsonValue]) -> list[NeMoGymResponseOutputItem]:
    _calls(observation)  # Validate the captured response shape before projection.
    message = deepcopy(observation["response"]["choices"][0]["message"])
    if message.get("role") != "assistant" or not isinstance(message.get("content") or "", str):
        raise ValueError("unsupported model message")
    converter = ResponsesConverter(return_token_id_information=True)
    reasoning = message.pop("reasoning_content", None)
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise ValueError("unsupported model reasoning")
        message["content"] = converter._wrap_reasoning_in_think_tags([reasoning]) + (message.get("content") or "")
    return converter.postprocess_assistant_message_dict(message)


def _without_recovered_rejections(
    observations: Sequence[Mapping[str, JsonValue]],
) -> list[Mapping[str, JsonValue]]:
    """Omit only proven HTTP rejections recovered by the same later native turn.

    Raw observations stay intact. Missing/partial captures, unrecovered errors,
    and multiple successful responses remain visible to the evidence checks.
    """
    completed, retained = set(), []
    for observation in reversed(observations):
        key = (observation.get("session_id"), observation.get("assistant_message_id"))
        identified = all(isinstance(value, str) and value for value in key)
        error = observation.get("transport_error")
        if (
            identified
            and key in completed
            and "response" not in observation
            and isinstance(error, dict)
            and error.get("phase") == "http_rejected"
            and type(error.get("status")) is int
            and 400 <= error["status"] <= 599
        ):
            continue
        retained.append(observation)
        if identified and "response" in observation:
            try:
                _calls(observation)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            completed.add(key)
    return list(reversed(retained))


def bind_observations(
    evidence: SessionEvidence,
    observations: Sequence[Mapping[str, JsonValue]],
    *,
    prefix_messages: frozenset[tuple[str, str]] = frozenset(),
) -> SessionEvidence:
    """Keep execution facts and literal actor names distinct, without inventing calls.

    System text is recorded where it was observed, never applied retroactively to
    a restored prefix. Missing/mismatched evidence cannot produce a complete grade.
    """
    result = evidence.model_copy(deep=True)
    issues: list[str] = []
    indexed = {}
    compactions = {}
    native_compactions = {
        (record["session_id"], record["message_id"]): record for record in result.facts.get("native_compactions", [])
    }
    for observation in _without_recovered_rejections(observations):
        key = (observation.get("session_id"), observation.get("assistant_message_id"))
        if not all(isinstance(value, str) and value for value in key):
            issues.append("unidentified_model_observation")
            continue
        if observation.get("source") == "compaction":
            try:
                if key in compactions or key not in native_compactions:
                    raise ValueError("unknown or duplicate native summary")
                if observation.get("request", {}).get("tools") or _calls(observation):
                    raise ValueError("internal summary may not execute tools")
                if output_digest(_response_output(observation)) != native_compactions[key]["output_sha256"]:
                    raise ValueError("native and captured summaries disagree")
                compactions[key] = observation
            except (KeyError, TypeError, ValueError, AttributeError):
                issues.append(f"inconsistent_compaction_observation:{key[0]}:{key[1]}")
            continue
        if key in indexed:
            issues.append(f"ambiguous_model_observation:{key[0]}:{key[1]}")
        indexed[key] = observation
    for key in native_compactions.keys() - compactions.keys() - prefix_messages:
        issues.append(f"missing_compaction_observation:{key[0]}:{key[1]}")
    result.facts["observed_native_compactions"] = [
        {"session_id": session, "message_id": message} for session, message in compactions
    ]
    steps = []
    seen, last_systems, execution_tools = set(), {}, {}
    for step in result.steps:
        key = (step.session_id, step.id)
        if step.role != "assistant" or key in prefix_messages:
            steps.append(step)
            continue
        observation = indexed.get(key)
        if observation is None:
            issues.append(f"missing_model_observation:{step.session_id}:{step.id}")
            steps.append(step)
            continue
        seen.add(key)
        try:
            calls = _calls(observation)
            if set(calls) != {call.call_id for call in step.calls}:
                raise ValueError("native and model tool calls disagree")
            for call in step.calls:
                wire = calls[call.call_id]
                call.model_tool = wire["name"]
                try:
                    call.model_args = json.loads(wire["arguments"])
                except json.JSONDecodeError:
                    # Invalid JSON is an observable rejected call, not a reason
                    # to substitute native default arguments into the trajectory.
                    call.model_args = wire["arguments"]
                logical = observation.get("execution_tools", {}).get(wire["name"])
                if logical is None and call.rejected:
                    # OpenCode's own invalid handler is an execution receipt for
                    # an unknown model attempt, not a selected executable tool.
                    logical = "invalid"
                if not isinstance(logical, str) or not logical:
                    raise ValueError("missing selected tool binding")
                call.tool = logical
                execution_tools[(step.session_id, call.call_id)] = logical
            request = observation["request"]
            systems = [
                content_text(item.get("content"))
                for item in request.get("messages", [])
                if item.get("role") == "system"
            ]
            if systems != last_systems.get(step.session_id, []):
                for index, text in enumerate(systems):
                    steps.append(
                        SessionStep(
                            id=f"{step.id}:system:{index}", session_id=step.session_id, role="system", text=text
                        )
                    )
                last_systems[step.session_id] = systems
        except (KeyError, TypeError, ValueError, AttributeError):
            issues.append(f"inconsistent_model_observation:{step.session_id}:{step.id}")
        steps.append(step)
    for key in indexed.keys() - seen - prefix_messages:
        issues.append(f"model_observation_missing_native_message:{key[0]}:{key[1]}")
    for step in steps:
        if step.role == "tool" and (step.session_id, step.call_id) in execution_tools:
            step.tool = execution_tools[step.session_id, step.call_id]
    result.steps = steps
    if issues:
        if result.status == "complete":
            result.status = "incomplete"
        result.issues = list(dict.fromkeys(result.issues + issues))
    return result


def _compaction_text(text: str) -> str:
    """Stock OpenCode 1.17.11's 2,000 UTF-16-unit tool-context truncation.

    This validates an already captured receipt; it never supplies missing text.
    JavaScript slices may split a surrogate pair, so preserve that exact value.
    """
    units = text.encode("utf-16-le", errors="surrogatepass")
    if len(units) <= 4000:
        return text
    prefix = units[:4000].decode("utf-16-le", errors="surrogatepass")
    return prefix + f"\n[Tool output truncated for compaction: omitted {len(units) // 2 - 2000} chars]"


def project_output(
    output: Sequence[NeMoGymResponseOutputItem],
    observations: Sequence[Mapping[str, JsonValue]],
    *,
    session_id: str,
    compaction_messages: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[list[NeMoGymResponseOutputItem], list[str]]:
    """Project captured call/result text and audit the exported assistant output.

    Native exports remain execution receipts, not proof of what the model saw.
    In particular, history hooks can change a failed tool's visible result after
    OpenCode stored it. Missing or changing observations are explicit capture
    gaps, never reconstructed from execution logs. Sessions are kept separate.
    """
    calls, tool_outputs, internal_outputs = {}, {}, {}
    native_outputs = {
        item.call_id: item.output for item in output if getattr(item, "type", None) == "function_call_output"
    }
    generated, issues = [], []
    for observation in _without_recovered_rejections(observations):
        if observation.get("session_id") != session_id:
            continue
        if observation.get("source") == "compaction":
            # A native compaction can consume the last tool result instead of
            # the next actor request. Accept only an independently audited
            # summary's exact native result or pinned native truncation. Project
            # only that captured view, never omitted bytes or a replacement for
            # an earlier actor observation. Full native receipts stay on disk.
            key = (session_id, observation.get("assistant_message_id"))
            if key in compaction_messages and "response" in observation:
                for message in observation.get("request", {}).get("messages", []):
                    identity = message.get("tool_call_id")
                    content = message.get("content")
                    if (
                        message.get("role") == "tool"
                        and identity in calls
                        and isinstance(content, str)
                        and isinstance(native_outputs.get(identity), str)
                        and content in (native_outputs[identity], _compaction_text(native_outputs[identity]))
                    ):
                        try:
                            content.encode("utf-8")
                        except UnicodeEncodeError:
                            # A split surrogate cannot cross Gym's response
                            # boundary. Keep the capture gap, not a fabricated
                            # replacement character or a falsely complete row.
                            continue
                        internal_outputs[identity] = content
            continue
        for message in observation.get("request", {}).get("messages", []):
            if message.get("role") != "tool" or message.get("tool_call_id") not in calls:
                # A continuation's old prefix is context, not fresh output.
                continue
            identity, content = message["tool_call_id"], message.get("content")
            if isinstance(content, list) and all(
                isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                for part in content
            ):
                content = "".join(part["text"] for part in content)
            if not isinstance(content, str):
                issues.append(f"unsupported_tool_observation:{identity}")
            elif identity in tool_outputs and tool_outputs[identity] != content:
                issues.append(f"changed_tool_observation:{identity}")
            else:
                tool_outputs[identity] = content
        try:
            observed = _calls(observation)
            if calls.keys() & observed.keys():
                raise ValueError("duplicate model call identity")
            calls.update(observed)
            generated.extend(_response_output(observation))
        except (TypeError, ValueError, AttributeError):
            issues.append("invalid_model_output_capture")
            continue
    tool_outputs = internal_outputs | tool_outputs
    result = deepcopy(list(output))
    result_ids = set()
    for item in result:
        if getattr(item, "type", None) == "function_call" and item.call_id in calls:
            item.name = calls[item.call_id]["name"]
            item.arguments = calls[item.call_id]["arguments"]
        elif getattr(item, "type", None) == "function_call_output":
            if item.call_id in result_ids:
                issues.append(f"duplicate_exported_result:{item.call_id}")
            result_ids.add(item.call_id)
            if item.call_id not in tool_outputs:
                issues.append(f"missing_tool_observation:{item.call_id}")
            else:
                item.output = tool_outputs[item.call_id]
    if set(tool_outputs) != result_ids:
        issues.append("captured_and_exported_tool_results_disagree")
    if _generated_content(result) != _generated_content(generated):
        issues.append("captured_and_exported_assistant_output_disagree")
    return result, list(dict.fromkeys(issues))
