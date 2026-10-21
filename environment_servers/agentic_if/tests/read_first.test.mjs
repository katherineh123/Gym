// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const directory = process.env.IF_TEST_ROOT;
const target = "file$&.txt";
const file = path.join(directory, target);
fs.writeFileSync(file, "original");
fs.writeFileSync(process.env.NEMO_GYM_OPENCODE_PLANTS, "[]");
const {IfProfile} = await import(process.env.IF_PROFILE_PLUGIN);
const hook = await IfProfile({directory});
let seq = 0;
const before = (h, tool, args, sessionID = "root", callID = String(++seq)) => {
  const input = {tool, sessionID, callID};
  return h["tool.execute.before"](input, {args}).then(() => input);
};
const after = async (h, input) => {
  const output = {output: "actual result", metadata: {}};
  await h["tool.execute.after"](input, output);
  return output;
};
const edit = {filePath: target, oldString: "original", newString: "updated"};
await assert.rejects(before(hook, "edit", {...edit}), {message: `Read ${target} first.`});
await assert.rejects(before(hook, "write", {filePath: target, content: "overwrite"}), /before overwriting/);
await before(hook, "write", {filePath: "new.txt", content: "new"});
// Merely starting a read (or failing it) does not count.
await before(hook, "read", {filePath: target});
await assert.rejects(before(hook, "edit", {...edit}), /first/);
await after(hook, await before(hook, "bash", {command: `cat '${target}'`}));
await assert.rejects(before(hook, "edit", {...edit}), /first/);
const read = await after(hook, await before(hook, "read", {filePath: target}));
assert.deepEqual(read.metadata.nemo_gym_profile.read_first, {
  file, sha256: crypto.createHash("sha256").update("original").digest("hex")
});
await before(hook, "edit", {...edit});
fs.writeFileSync(file, "external update");
await assert.rejects(before(hook, "edit", {...edit}), /again/);
// No changed-case configured for write; successful writes refresh the known hash.
const writing = await before(hook, "write", {filePath: target, content: "new content"});
fs.writeFileSync(file, "new content");
await after(hook, writing);
await before(hook, "edit", {...edit});
await assert.rejects(before(hook, "edit", {...edit}, "child"), /first/);
// Canonical file identity resolves symlinks; session/call identities do not collide.
fs.symlinkSync(file, path.join(directory, "alias.txt"));
const rootRead = await before(hook, "read", {filePath: "alias.txt"}, "root", "shared");
fs.writeFileSync(path.join(directory, "other.txt"), "other");
const childRead = await before(hook, "read", {filePath: "other.txt"}, "child", "shared");
await after(hook, childRead);
await after(hook, rootRead);
await before(hook, "edit", {...edit});
await assert.rejects(before(hook, "edit", {...edit}, "child"), /first/);
// Refusal does not consume a matching plant. Read then edit fires it once.
fs.writeFileSync(process.env.NEMO_GYM_OPENCODE_PLANTS, JSON.stringify([
  {id: "edit-plant", match: {tool: "edit"}, action: "fail", text: "PLANTED"}
]));
const guardedPlant = await IfProfile({directory});
await assert.rejects(before(guardedPlant, "edit", {...edit}), /first/);
await after(guardedPlant, await before(guardedPlant, "read", {filePath: target}));
await assert.rejects(before(guardedPlant, "edit", {...edit}), {message: "PLANTED"});
// A restored prefix retains completed read receipts, not failed reads.
fs.writeFileSync(process.env.NEMO_GYM_OPENCODE_PLANTS, "[]");
const resumed = await IfProfile({directory});
const history = [{info: {role: "assistant", sessionID: "resumed"}, parts: [
  {type: "tool", tool: "read", callID: "old", sessionID: "resumed", state: {
    status: "completed", input: {filePath: target}, metadata: {nemo_gym_profile: {read_first: {
      file, sha256: crypto.createHash("sha256").update("new content").digest("hex")
    }}}
  }}
]}];
await resumed["experimental.chat.messages.transform"]({}, {messages: history});
await before(resumed, "edit", {...edit}, "resumed");
await assert.rejects(before(resumed, "edit", {...edit}, "new-child"), /first/);
history[0].info.sessionID = "failed-prefix";
history[0].parts[0].sessionID = "failed-prefix";
history[0].parts[0].state.status = "error";
await resumed["experimental.chat.messages.transform"]({}, {messages: history});
await assert.rejects(before(resumed, "edit", {...edit}, "failed-prefix"), /first/);
// Directory reads do not become file-read receipts.
const directoryRead = await after(resumed, await before(resumed, "read", {filePath: directory}));
assert.equal(directoryRead.metadata.nemo_gym_profile.read_first, undefined);
// Interface mappings still guard the actual native path; aliases must not
// rewrite literal guard errors imported from an earlier hook version.
const configured = JSON.parse(fs.readFileSync(process.env.NEMO_GYM_OPENCODE_PROFILE));
configured.tools = {read: {parameters: {type: "object", properties: {path: {type: "string"}}, required: ["path"]},
  args_map: {filePath: "path"}}};
configured.visible_tools = {read: "ReadSource", edit: "EditSource", write: "WriteSource"};
fs.writeFileSync(process.env.NEMO_GYM_OPENCODE_PROFILE, JSON.stringify(configured));
const mapped = await IfProfile({directory});
await after(mapped, await before(mapped, "read", {path: target}));
await before(mapped, "edit", {...edit});
const oldError = [{info: {role: "assistant", sessionID: "old"}, parts: [{type: "tool", tool: "edit",
  callID: "old-rejection", state: {status: "error", input: {...edit}, error: `Read ${target} first.`}}]}];
await mapped["experimental.chat.messages.transform"]({}, {messages: oldError});
assert.equal(oldError[0].parts[0].state.error, `Read ${target} first.`);
console.log("read-first native behavior passed");
