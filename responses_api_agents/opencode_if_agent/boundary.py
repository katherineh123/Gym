# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reversible model-boundary tool aliases; native OpenCode still executes its own tools."""

import codecs
import json
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any


def model_request(
    payload: dict[str, Any], *, tool_names: dict[str, str], system_text: str, system_prefix: str = ""
) -> dict[str, Any]:
    """Apply actor-facing names to schemas/history without changing tool arguments or issue text."""
    result = deepcopy(payload)
    tools = result.get("tools", [])
    functions = [tool["function"] for tool in tools if tool.get("type") == "function"]
    available = {function["name"] for function in functions}
    if tools and not set(tool_names) <= available:
        raise ValueError(
            f"requested tool aliases unavailable in this native session: {sorted(set(tool_names) - available)}"
        )
    mapped = [tool_names.get(name, name) for name in available]
    if len(mapped) != len(set(mapped)):
        raise ValueError("tool alias collision with native/MCP tool")
    for function in functions:
        function["name"] = tool_names.get(function["name"], function["name"])
    for message in result.get("messages", []):
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            if "name" in function:
                function["name"] = tool_names.get(function["name"], function["name"])
        if message.get("role") == "tool" and "name" in message:
            message["name"] = tool_names.get(message["name"], message["name"])
    choice = result.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        function = choice["function"]
        function["name"] = tool_names.get(function["name"], function["name"])
    additions = [system_text] if system_text else []
    if tool_names:
        additions.append(
            "Tool names in this session (same behavior and arguments): "
            + "; ".join(f"use {alias} for the {native} tool" for native, alias in sorted(tool_names.items()))
            + ". Use the names in the supplied tool registry."
        )
    if additions or system_prefix:
        messages = result.setdefault("messages", [])
        suffix = "\n\n".join(additions)
        if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
            messages[0]["content"] = "\n\n".join(p for p in (system_prefix, messages[0]["content"], suffix) if p)
        else:
            messages.insert(0, {"role": "system", "content": "\n\n".join(p for p in (system_prefix, suffix) if p)})
    return result


def native_response(payload: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    """Translate model-emitted aliases back to native names before OpenCode dispatches."""
    result = deepcopy(payload)
    inverse = {alias: native for native, alias in names.items()}
    for choice in result.get("choices", []):
        for call in choice.get("message", {}).get("tool_calls", []):
            function = call.get("function", {})
            if "name" in function:
                function["name"] = inverse.get(function["name"], function["name"])
    return result


class ToolStreamRewriter:
    """Buffer only function names (which may span SSE events), never the generated arguments."""

    def __init__(self, names: dict[str, str]) -> None:
        self.inverse = {alias: native for native, alias in names.items()}
        self.pending: dict[tuple[int, int], dict[str, Any]] = {}
        self.started: set[tuple[int, int]] = set()

    def _flush(self, key: tuple[int, int]) -> dict[str, Any]:
        call = self.pending.pop(key)
        name = call["function"]["name"]
        if not name:
            raise ValueError("streamed tool arguments arrived without a function name")
        call["function"]["name"] = self.inverse.get(name, name)
        self.started.add(key)
        return call

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate one Chat Completions chunk, retaining usage and other fields verbatim."""
        result = deepcopy(event)
        for choice in result.get("choices", []):
            delta = choice.get("delta", {})
            calls = []
            for call in delta.pop("tool_calls", []):
                key = (choice.get("index", 0), call["index"])
                function = call.get("function", {})
                if key in self.started:
                    if function.get("name"):
                        raise ValueError("function name continued after arguments started")
                    calls.append(call)
                    continue
                buffered = self.pending.setdefault(key, {"index": call["index"], "function": {"name": ""}})
                buffered.update({k: v for k, v in call.items() if k != "function"})
                buffered["function"]["name"] += function.get("name", "")
                if len(buffered["function"]["name"]) > 256:
                    raise ValueError("streamed function name too long")
                if function.get("arguments"):
                    buffered["function"]["arguments"] = function["arguments"]
                    calls.append(self._flush(key))
            if choice.get("finish_reason") is not None:
                calls.extend(self._flush(key) for key in list(self.pending) if key[0] == choice.get("index", 0))
            if calls:
                delta["tool_calls"] = calls
        return [result]


async def rewrite_sse(source: AsyncIterator[bytes], names: dict[str, str]) -> AsyncIterator[bytes]:
    """Translate complete SSE events across arbitrary UTF-8/network boundaries."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    writer = ToolStreamRewriter(names)
    done = False
    async for part in source:
        buffer += decoder.decode(part)
        # Normalize after accumulating: CRLF itself can be split across network reads.
        buffer = buffer.replace("\r\n", "\n")
        if len(buffer) > 4 * 1024 * 1024:
            raise ValueError("model SSE event exceeds 4 MiB")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            data = "\n".join(line[5:].lstrip(" ") for line in frame.split("\n") if line.startswith("data:"))
            if not data:
                yield (frame + "\n\n").encode()
            elif data == "[DONE]":
                if writer.pending:
                    raise ValueError("model stream ended with unfinished tool names")
                done = True
                yield b"data: [DONE]\n\n"
            else:
                for event in writer.feed(json.loads(data)):
                    yield ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode()
    buffer += decoder.decode(b"", final=True)
    if buffer.strip() or not done:
        raise ValueError("model SSE stream ended before a complete DONE event")
