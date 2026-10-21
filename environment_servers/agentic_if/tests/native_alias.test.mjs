// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";

const seen = [];
const server = http.createServer(async (req, res) => {
  assert.equal(req.headers.authorization, "Bearer attempt-only");
  let body = "";
  for await (const chunk of req) body += chunk;
  const call = JSON.parse(body);
  seen.push(call);
  res.setHeader("content-type", "application/json");
  if (call.call_id === "missing") { res.writeHead(409).end("{}"); return; }
  res.end(JSON.stringify({allowed: call.session_id === "root", model_name: call.session_id === "root" ? "ReadSource" : "read", available_names: ["ReadSource"]}));
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
process.env.NEMO_GYM_IF_TOOL_CALL_URL = `http://127.0.0.1:${server.address().port}/tool-call`;
process.env.NEMO_GYM_IF_TOOL_CALL_TOKEN = "attempt-only";
try {
  const {IfProfile} = await import(process.env.IF_PROFILE_PLUGIN);
  const hook = await IfProfile({directory: process.env.IF_TEST_ROOT});
  const invoke = (sessionID, callID, tool = "read") => hook["tool.execute.before"]({sessionID, callID, tool}, {args: {filePath: "/app/file"}});
  await assert.rejects(invoke("child", "same"), /read.*unavailable.*ReadSource/s);
  await invoke("root", "same");
  await assert.rejects(invoke("root", "missing"), /provenance unavailable/i);
  await invoke("root", "native", "bash");
  assert.equal(seen.length, 3); // No callback for unchanged native tools.
  const messages = {messages: [{info: {role: "assistant", sessionID: "child"}, parts: [{type: "tool", tool: "read", sessionID: "child", callID: "same", state: {status: "error", input: {}, error: "read unavailable"}}]}]};
  await hook["experimental.chat.messages.transform"]({}, messages);
  assert.match(messages.messages[0].parts[0].state.error, /^read.*unavailable/s);
  const records = fs.readFileSync(process.env.NEMO_GYM_OPENCODE_PROFILE_APPLIED, "utf8").trim().split("\n").map(JSON.parse);
  assert.equal(records.filter(x => x.kind === "alias_rejected" && x.session_id === "child" && x.call_id === "same").length, 1);
} finally {
  await new Promise(resolve => server.close(resolve));
}
console.log("native alias guard passed");
