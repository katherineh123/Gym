# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental model transport codecs; no IF execution or grading ownership."""

import codecs
import json
from collections.abc import AsyncIterator, Callable
from copy import deepcopy

from pydantic import JsonValue

from environment_servers.agentic_if.model_boundary import PreparedRequest


class _ChatStream:
    def __init__(self, prepared: PreparedRequest | None) -> None:
        self.prepared = prepared
        self.pending: dict[tuple[int, int], dict] = {}
        self.started: set[tuple[int, int]] = set()
        self.metadata: dict = {}
        self.choices: dict[int, dict] = {}
        self.calls: dict[tuple[int, int], dict] = {}

    def _flush(self, key: tuple[int, int]) -> dict:
        call = self.pending.pop(key)
        if not call["function"]["name"]:
            raise ValueError("streamed tool arguments have no function name")
        if self.prepared is not None:
            wrapped = self.prepared.native_reply({"choices": [{"message": {"tool_calls": [call]}}]})
            call = wrapped["choices"][0]["message"]["tool_calls"][0]
        self.started.add(key)
        return call

    def feed(self, raw: dict[str, JsonValue]) -> dict[str, JsonValue]:
        self.metadata.update({key: value for key, value in raw.items() if key != "choices"})
        result = deepcopy(raw)
        for choice in raw.get("choices", []):
            index = choice.get("index", 0)
            target = self.choices.setdefault(index, {"index": index, "message": {}})
            if choice.get("finish_reason") is not None:
                target["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta", {})
            for key, value in delta.items():
                if key in {"content", "refusal", "reasoning_content"} and isinstance(value, str):
                    target["message"][key] = target["message"].get(key, "") + value
                elif key != "tool_calls":
                    target["message"][key] = value
            for call in delta.get("tool_calls") or []:
                assembled = self.calls.setdefault((index, call["index"]), {"function": {}})
                assembled.update({k: v for k, v in call.items() if k not in {"index", "function"}})
                for key, value in call.get("function", {}).items():
                    assembled["function"][key] = assembled["function"].get(key, "") + value
        for choice in result.get("choices", []):
            delta = choice.get("delta", {})
            calls = []
            for call in delta.pop("tool_calls", None) or []:
                key = (choice.get("index", 0), call["index"])
                function = call.get("function", {})
                if key in self.started:
                    if function.get("name"):
                        raise ValueError("function name continued after arguments started")
                    calls.append(call)
                    continue
                pending = self.pending.setdefault(key, {"index": call["index"], "function": {"name": ""}})
                pending.update({k: v for k, v in call.items() if k != "function"})
                pending["function"]["name"] += function.get("name", "")
                if len(pending["function"]["name"]) > 256:
                    raise ValueError("streamed function name too long")
                if function.get("arguments"):
                    pending["function"]["arguments"] = function["arguments"]
                    calls.append(self._flush(key))
            if choice.get("finish_reason") is not None:
                calls.extend(self._flush(key) for key in list(self.pending) if key[0] == choice.get("index", 0))
            if calls:
                delta["tool_calls"] = calls
        return result

    def response(self) -> dict[str, JsonValue]:
        if self.pending or not self.choices or any("finish_reason" not in choice for choice in self.choices.values()):
            raise ValueError("Chat stream has no complete terminal choice")
        for index, choice in self.choices.items():
            calls = [call for (owner, _), call in sorted(self.calls.items()) if owner == index]
            if calls:
                choice["message"]["tool_calls"] = calls
        return self.metadata | {"object": "chat.completion", "choices": list(self.choices.values())}


async def rewrite_stream(
    source: AsyncIterator[bytes],
    *,
    prepared: PreparedRequest | None,
    observe: Callable[[dict[str, JsonValue]], None],
) -> AsyncIterator[bytes]:
    """Preserve streaming and observe model-visible responses before alias reversal."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer, terminal = "", False
    chat = _ChatStream(prepared)
    async for part in source:
        buffer = (buffer + decoder.decode(part)).replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            if len(frame) > 4 * 1024 * 1024:
                raise ValueError("model SSE event exceeds 4 MiB")
            data = "\n".join(line[5:].lstrip(" ") for line in frame.split("\n") if line.startswith("data:"))
            if not data:
                yield (frame + "\n\n").encode()
                continue
            if data == "[DONE]":
                observe(chat.response())
                terminal = True
                yield b"data: [DONE]\n\n"
                continue
            outgoing = chat.feed(json.loads(data))
            lines = [line for line in frame.split("\n") if not line.startswith("data:")]
            lines.append("data: " + json.dumps(outgoing, ensure_ascii=False))
            yield ("\n".join(lines) + "\n\n").encode()
        if len(buffer) > 4 * 1024 * 1024:
            raise ValueError("model SSE event exceeds 4 MiB")
    buffer += decoder.decode(b"", final=True)
    if buffer.strip() or not terminal:
        raise ValueError("model stream has no complete terminal event")
