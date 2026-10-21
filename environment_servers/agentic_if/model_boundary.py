# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One actor-facing presentation contract over OpenCode's native Chat Completions transport."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from pydantic import JsonValue

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_if.task_variants.builder import content_text
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


# Only these keywords contain subschemas. Other dictionaries (const/default/
# examples, including values with a key named "type") are literal data.
SCHEMA_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
SCHEMA_ARRAYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
SCHEMA_SINGLES = {
    "items",
    "additionalItems",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
}


def effective_tool_names(
    available: list[str], requested: dict[str, str], *, allow_missing: bool = False
) -> dict[str, str]:
    """{tool: alias} of the requested tools the request offers; a missing one raises unless `allow_missing`."""
    bindings = {}
    for logical, alias in requested.items():
        if logical in available:
            bindings[logical] = alias
        elif not allow_missing:
            raise ValueError(f"unavailable logical tool: {logical}")
    return bindings


def _parameter_schema(schema: dict, path: tuple[str, ...]) -> dict:
    node, remaining = schema, path
    while remaining:
        keyword, *tail = remaining
        if not isinstance(node, dict) or keyword not in node:
            raise ValueError("unknown parameter schema")
        child = node[keyword]
        if keyword in SCHEMA_MAPS:
            if not tail or not isinstance(child, dict) or tail[0] not in child:
                raise ValueError("unknown parameter schema property")
            node, remaining = child[tail[0]], tail[1:]
        elif keyword in SCHEMA_ARRAYS or (keyword == "items" and isinstance(child, list)):
            if not tail or not isinstance(child, list) or not tail[0].isascii() or not tail[0].isdigit():
                raise ValueError("invalid parameter schema array index")
            index = int(tail[0])
            if str(index) != tail[0] or index >= len(child):
                raise ValueError("unknown parameter schema array element")
            node, remaining = child[index], tail[1:]
        elif keyword in SCHEMA_SINGLES:
            node, remaining = child, tail
        else:
            raise ValueError("target is not a parameter schema location")
    if not isinstance(node, dict):
        raise ValueError("parameter schema must be an object to carry a description")
    return node


def describe_parameters(schema: dict, descriptions: dict[str, str]) -> dict:
    """Replace descriptions on existing parameter-schema nodes, leaving meaning intact.

    Keys are top-level property names or JSON pointers to nested schema objects,
    including local $defs/definitions. No references are followed and no schema
    nodes are created; literal examples/defaults are never schema targets.
    """
    result = deepcopy(schema)
    seen: set[tuple[str, ...]] = set()
    for key, text in descriptions.items():
        path = (
            tuple(part.replace("~1", "/").replace("~0", "~") for part in key[1:].split("/"))
            if key.startswith("/")
            else ("properties", key)
        )
        if not path or path in seen:
            raise ValueError(f"invalid or duplicate parameter target: {key}")
        seen.add(path)
        node = _parameter_schema(result, path)
        node["description"] = text
    return result


def _native_text(content: JsonValue) -> str:
    """Project native text blocks without applying the authoring-row schema."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(
        not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str)
        for part in content
    ):
        raise ValueError("native user prompts must contain only protocol-native text blocks")
    return "".join(part["text"] for part in content)


def _text_slice(content: JsonValue, start: int, end: int) -> JsonValue:
    """Split merged user turns while retaining native block metadata."""
    if isinstance(content, str):
        return content[start:end]
    blocks, offset = [], 0
    for part in content:
        length = len(part["text"])
        left, right = max(start - offset, 0), min(end - offset, length)
        if left < right:
            blocks.append({**deepcopy(part), "text": part["text"][left:right]})
        offset += length
    return blocks


@dataclass(frozen=True)
class PreparedRequest:
    """An immutable per-request mapping; concurrent children cannot overwrite it."""

    payload: dict[str, JsonValue]
    names: dict[str, str]
    allowed_names: frozenset[str]
    execution_tools: dict[str, str]
    restricted: bool
    observe_dispatch: Callable[[str, str, str, bool], None] | None = None

    def native_reply(self, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Restore dispatch labels without replacing native validation/recovery.

        OpenCode owns rejection; its before-tool hook checks alias provenance.
        The harness produces its own error; this layer never manufactures a tool
        result, repairs arguments, or decides whether the episode continues.
        """
        result = deepcopy(payload)
        inverse = {visible: native for native, visible in self.names.items()}
        calls = [
            (call.get("id"), call["function"])
            for choice in result.get("choices", [])
            for call in (choice.get("message") or {}).get("tool_calls") or []
        ]
        for call_id, call in calls:
            name = call.get("name")
            if not name:
                continue
            allowed = name in self.allowed_names
            if allowed:
                native = inverse.get(name, name)
            elif name.lower() in self.allowed_names:
                # OpenCode repairs a name by lowercasing it. Translate an alias
                # without performing the repair: the native harness still
                # receives a mismatched-capitalization name and owns that step.
                lower = name.lower()
                target = inverse.get(lower, lower)
                native = target.upper() if target != lower else name
                self.execution_tools[name] = self.execution_tools[lower]
                allowed = True
            else:
                native = name
            if self.observe_dispatch is not None:
                self.observe_dispatch(call_id, name, native, allowed)
            call["name"] = native
        return result


class ModelBoundary:
    """Resolve literal prompts and the selected native registry for one attempt.

    This layer never changes a tool's arguments or executes a tool. Installed
    interface hooks run inside the native harness.
    """

    def __init__(self, spec: RuntimeSpec, params: NeMoGymResponseCreateParamsNonStreaming) -> None:
        self.spec = spec.model_copy(deep=True)
        messages = params.model_dump(mode="json")["input"]
        self.users = [
            {"role": "user", "content": content_text(item["content"])}
            for item in messages
            if item.get("role") == "user"
        ]
        systems = [content_text(item["content"]) for item in messages if item.get("role") == "system"]
        self.system = systems[0] if systems else None
        self.max_output_tokens = params.max_output_tokens
        self.interfaces = installed_interfaces(spec.tools.tool_interfaces)

    def prepare(
        self, payload: dict[str, JsonValue], *, child_session: bool = False, require_user_prompt: bool = True
    ) -> PreparedRequest:
        """Prepare one actor request, without mutating its native transcript."""
        result = deepcopy(payload)
        tools = result.get("tools", [])
        available = [tool.get("function", {}).get("name") for tool in tools]
        if any(not isinstance(name, str) or not name for name in available) or len(set(available)) != len(available):
            raise ValueError("native tool registry contains missing or duplicate names")
        settings = self.spec.tools
        if settings.tool_names is None:
            selected = set(available)
        else:
            selected = set(effective_tool_names(available, settings.tool_names, allow_missing=child_session))
        unexpected = set(available) - selected
        if unexpected:
            raise ValueError(
                f"OpenCode exposed unexpected tools: {sorted(unexpected)}; check native permission configuration"
            )
        result["tools"] = tools
        targets = (
            set(settings.tool_names or {})
            | set(settings.tool_descriptions)
            | set(settings.tool_parameter_descriptions)
        )
        requested = {name: (settings.tool_names or {}).get(name, name) for name in targets}
        bindings = effective_tool_names(available, requested, allow_missing=child_session)
        visible = [bindings.get(name, name) for name in selected]
        if len(set(visible)) != len(visible):
            raise ValueError("tool alias collision")
        logical_by_visible = {(settings.tool_names or {}).get(name, name): name for name in targets}
        for tool in tools:
            definition = tool["function"]
            native = definition["name"]
            alias = bindings.get(native, native)
            logical = logical_by_visible.get(alias, native)
            if "description" in self.interfaces.get(logical, {}):
                definition["description"] = self.interfaces[logical]["description"]
            if logical in settings.tool_descriptions:
                definition["description"] = settings.tool_descriptions[logical]
            if logical in settings.tool_parameter_descriptions:
                definition["parameters"] = describe_parameters(
                    definition.get("parameters", {}), settings.tool_parameter_descriptions[logical]
                )
            definition["name"] = alias
        for message in result.get("messages", []):
            if not isinstance(message, dict):
                raise ValueError("native actor messages must be structured objects")
            for call in message.get("tool_calls", []):
                call["function"]["name"] = bindings.get(call["function"]["name"], call["function"]["name"])
            if message.get("role") == "tool":
                # Gym correlates results by tool_call_id; redundant native names
                # are not a separate execution route.
                message.pop("name", None)
        choice = result.get("tool_choice")
        if isinstance(choice, dict):
            named = choice.get("function", choice)
            if named.get("name"):
                native = named["name"]
                if native not in selected:
                    raise ValueError("tool_choice names an unselected tool")
                named["name"] = bindings.get(native, native)
        self._prompts(result, child_session=child_session, require_user_prompt=require_user_prompt)
        if self.max_output_tokens is not None:
            current = result.get("max_tokens")
            result["max_tokens"] = (
                min(current, self.max_output_tokens) if current is not None else self.max_output_tokens
            )
        return PreparedRequest(
            result,
            bindings,
            frozenset(visible),
            {
                bindings.get(native, native): logical_by_visible.get(bindings.get(native, native), native)
                for native in selected
            },
            settings.tool_names is not None,
        )

    def _prompts(self, payload: dict[str, JsonValue], *, child_session: bool, require_user_prompt: bool) -> None:
        messages = payload.get("messages", [])
        if not child_session:
            combined = "\n\n".join(item["content"] for item in self.users)
            candidates = [
                (index, _native_text(item.get("content")))
                for index, item in enumerate(messages)
                if item.get("role") == "user"
            ]
            matches = [(index, text) for index, text in candidates if text == combined]
            if not matches and combined:
                matches = [(index, text) for index, text in candidates if combined in text]
            if not matches and require_user_prompt:
                raise ValueError("native actor history is missing the literal task/continuation prompt")
            # The native harness can surround the task with its own framing.
            # Keep that framing verbatim. Only undo the joining of authored
            # turns when there are multiple turns.
            if matches and len(self.users) > 1:
                index, text = matches[-1] if self.spec.continuation is not None else matches[0]
                original = messages[index]
                position, start = text.index(combined), 0
                restored = []
                for turn, user in enumerate(self.users):
                    position += len(user["content"])
                    end = len(text) if turn == len(self.users) - 1 else position
                    restored.append({**deepcopy(original), "content": _text_slice(original["content"], start, end)})
                    position += 2  # The join separator is not an authored turn.
                    start = position
                messages[index : index + 1] = restored
        if self.system is not None:
            # One literal contract for every model call. This also covers child
            # requests and explicit empty text. Omission is a no-op; native user
            # framing, observations and tools are untouched.
            messages = [{"role": "system", "content": self.system}] + [
                item for item in messages if item.get("role") not in {"system", "developer"}
            ]
            payload.pop("instructions", None)
        payload["messages"] = messages
