// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The request-kind plugin as staged next to upstream's assistant-message plugin (run by test_opencode_hooks.py).
import assert from "node:assert/strict";
import { test } from "node:test";
import { pathToFileURL } from "node:url";

const plugin = await import(pathToFileURL(process.env.IF_REQUEST_KIND_PLUGIN).href);
const { IfRequestKind } = plugin;
const header = "X-OpenCode-Assistant-Message-Id";
const kind = "X-OpenCode-Request-Kind";
const message = (id = "a") => ({
  id, sessionID: "s", parentID: "u", role: "assistant", agent: "build", modelID: "m", providerID: "p",
  time: { created: 1 },
});
const input = { sessionID: "s", message: { id: "u" }, agent: "build", model: { id: "m", providerID: "p" } };
const emit = (hooks, info) => hooks.event({ event: { type: "message.updated", properties: { info } } });
const headers = (hooks, update = {}) => {
  const output = { headers: {} };
  hooks["chat.headers"]({ ...input, ...update }, output);
  return output.headers;
};

test("only the wrapper is exported, so OpenCode registers one plugin", () => {
  assert.deepEqual(Object.keys(plugin), ["IfRequestKind"]);
});

test("internal compaction is identified separately from task-solving replies", () => {
  const compaction = IfRequestKind();
  emit(compaction, { ...message(), agent: "compaction" });
  assert.deepEqual(headers(compaction, { agent: "compaction" }), { [header]: "a", [kind]: "compaction" });
  const actor = IfRequestKind();
  emit(actor, message());
  assert.deepEqual(headers(actor), { [header]: "a", [kind]: "actor" });
  // Without exactly one saved assistant (a title call, or two candidates) neither header is guessed.
  assert.deepEqual(headers(compaction, { agent: "title" }), {});
  emit(actor, message("b"));
  assert.deepEqual(headers(actor), {});
});
