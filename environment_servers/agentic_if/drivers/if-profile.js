// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Apply a tool profile inside OpenCode through its own plugin hooks:
//   - tool.definition: each profiled tool gets the profile's JSON Schema;
//   - tool.execute.before: the model's (configured) arguments are mapped to the native tool's, in place;
//   - tool.execute.after: the configured arguments and the profile hash are recorded in the tool part's metadata;
//   - experimental.chat.messages.transform: every past call of a reshaped tool is shown in the configured form.
// The profile is a JSON file named by NEMO_GYM_OPENCODE_PROFILE. Without it, the plugin does nothing.
// The plugin keeps no state between runs: everything it needs is in the profile and in the stored tool parts.
//
// Planted tool failures (an internal JSON array named by NEMO_GYM_OPENCODE_PLANTS): each plant matches
// the calls of one native tool, optionally with one native argument tested against a regular expression, and acts on
// its nth matching call only. Its action is "fail": the call does not run and the model sees the plant's text as the
// call's error (thrown before any mapping, like the plugin's own rejections). Only calls that pass the profile's argument
// check count. One plant acts on a call: the first, in list order, whose nth match it is. Each firing is appended to
// NEMO_GYM_OPENCODE_PLANTS_FIRED, after one "loaded" line; plants that cannot be loaded (any other action included) leave
// an "error" line instead and none is applied (evidence is then incomplete).
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const PROFILE_PATH = process.env.NEMO_GYM_OPENCODE_PROFILE;
const APPLIED_PATH = process.env.NEMO_GYM_OPENCODE_PROFILE_APPLIED || "/tmp/nemo-gym-opencode-profile-applied.jsonl";
const PLANTS_PATH = process.env.NEMO_GYM_OPENCODE_PLANTS;
const PLANTS_FIRED_PATH = process.env.NEMO_GYM_OPENCODE_PLANTS_FIRED || "/tmp/nemo-gym-opencode-plants-fired.jsonl";
const META_KEY = "nemo_gym_profile";
// Kept in the stored input of every reshaped call: the model's configured arguments and the native->configured name
// map. The native tool ignores it (OpenCode's argument schemas strip unknown keys). It survives a call that fails
// inside the tool, which never reaches tool.execute.after, so the record and the model's view stay the configured form.
const STASH_KEY = "__nemo_gym_profile";

function load() {
  if (!PROFILE_PATH) return null;
  return JSON.parse(fs.readFileSync(PROFILE_PATH, "utf8"));
}

function getPath(obj, path) {
  let cur = obj;
  for (const key of path.split(".")) {
    if (cur === null || typeof cur !== "object" || !(key in cur)) return undefined;
    cur = cur[key];
  }
  return cur;
}

// configured -> native: every native argument from its configured path, plus the fixed values.
function toNative(tool, configured) {
  const native = {};
  for (const [nativeKey, path] of Object.entries(tool.args_map || {})) {
    const value = getPath(configured, path);
    if (value !== undefined) native[nativeKey] = value;
  }
  Object.assign(native, tool.fixed || {});
  return native;
}

// The configured schema is only described to the model (the AI SDK does not enforce a plain JSON Schema), so the plugin
// checks the model's arguments itself: required properties, JSON types, enums, and no properties the schema does not
// declare (unless it sets additionalProperties: true), so a recorded call always matches the schema the model was shown.
const TYPES = {
  string: (v) => typeof v === "string",
  number: (v) => typeof v === "number" && Number.isFinite(v),
  integer: (v) => Number.isInteger(v),
  boolean: (v) => typeof v === "boolean",
  object: (v) => v !== null && typeof v === "object" && !Array.isArray(v),
  array: (v) => Array.isArray(v),
  null: (v) => v === null,
};

function schemaErrors(schema, value, path = "") {
  if (!schema || typeof schema !== "object") return [];
  const where = path || "input";
  const types = schema.type === undefined ? [] : [].concat(schema.type);
  if (types.length && !types.some((t) => TYPES[t] && TYPES[t](value))) {
    return [`${where} must be ${types.join(" or ")}`];
  }
  if (schema.enum && !schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(value))) {
    return [`${where} must be one of ${JSON.stringify(schema.enum)}`];
  }
  const errors = [];
  if (TYPES.object(value)) {
    for (const key of schema.required || []) {
      if (!(key in value)) errors.push(`${path ? path + "." : ""}${key} is missing`);
    }
    for (const [key, sub] of Object.entries(schema.properties || {})) {
      if (key in value) errors.push(...schemaErrors(sub, value[key], path ? `${path}.${key}` : key));
    }
    if (schema.properties && schema.additionalProperties !== true) {
      for (const key of Object.keys(value)) {
        if (!(key in schema.properties)) errors.push(`${path ? path + "." : ""}${key} is not an accepted argument`);
      }
    }
  }
  if (Array.isArray(value) && schema.items) {
    value.forEach((item, i) => errors.push(...schemaErrors(schema.items, item, `${where}[${i}]`)));
  }
  return errors;
}

// The workspace diff after a file-changing call, recorded so that a later replay can cut the session after any completed
// step and restore the files exactly. Computed against a temporary copy of the git index, so the agent's own
// `git status` is unchanged; keep deletions unstaged so the diff records them too.
const FILE_TOOLS = new Set(["edit", "write", "bash"]);

function git(dir, args, env) {
  const r = spawnSync("git", ["-C", dir, ...args], {
    encoding: "utf8",
    env: { ...process.env, ...(env || {}) },
    maxBuffer: 256 * 1024 * 1024,
  });
  if (r.status !== 0) throw new Error(`git ${args.join(" ")}: ${r.stderr}`);
  return r.stdout;
}

function workspaceDiff(dir) {
  const gitDir = git(dir, ["rev-parse", "--absolute-git-dir"]).trim();
  const index = path.join(os.tmpdir(), `nemo-gym-index-${process.pid}-${crypto.randomUUID()}`);
  try {
    const source = path.join(gitDir, "index");
    if (fs.existsSync(source)) {
      fs.copyFileSync(source, index);
      // Keep Git's racy-entry check: a fresh index time can hide same-size edits
      // made in the second of the last index write.
      const { atime, mtime } = fs.statSync(source);
      fs.utimesSync(index, atime, mtime);
    }
    git(dir, ["add", "-N", "--ignore-removal", "."], { GIT_INDEX_FILE: index });
    return git(dir, ["diff", "--binary", "--full-index"], { GIT_INDEX_FILE: index });
  } finally {
    fs.rmSync(index, { force: true });
  }
}

// Only exact, data-free diagnostics from the pinned native implementation are
// rewritten. File paths, suggestions, snippets and third-party errors are data.
const EDIT_DIAGNOSTICS = new Set([
  "filePath is required",
  "No changes to apply: oldString and newString are identical.",
  "oldString cannot be empty when editing an existing file. Provide the exact text to replace, or use write for an intentional full-file replacement.",
  "Refusing replacement because the matched span is much larger than oldString. Re-read the file and provide the full exact oldString for the intended replacement.",
  "Could not find oldString in the file. It must match exactly, including whitespace, indentation, and line endings.",
  "Found multiple matches for oldString. Provide more surrounding context to make the match unique.",
]);

function renameNative(tool, toolID, text, visibleTools) {
  const names = tool?.args_map || {};
  if (toolID === "read" && names.offset && /^Offset \d+ is out of range for this file \(\d+ lines\)$/.test(text)) {
    return names.offset + text.slice("Offset".length);
  }
  if (toolID !== "edit" || !EDIT_DIAGNOSTICS.has(text)) return text;
  let result = text.replace(/\b(?:filePath|oldString|newString)\b/g, word => names[word] || word);
  if (visibleTools !== undefined) {
    const recommendation = ", or use write for an intentional full-file replacement.";
    result = result.replace(recommendation, () => visibleTools.write
      ? `, or use ${visibleTools.write} for an intentional full-file replacement.`
      : ".");
  }
  return result;
}

// Hints OpenCode appends to a successful result that name a native argument. Only these exact phrases are renamed:
// the rest of a result is file or command output and stays byte-identical.
function renameHints(tool, toolID, text, metadata) {
  if (typeof text !== "string") return text;
  const offset = tool?.args_map?.offset;
  if (toolID !== "read" || !offset || !metadata?.display?.truncated) return text;
  if (metadata.display.type === "directory") {
    return text.replace(
      /(\n\n\(Showing \d+ of \d+ entries\. Use ')offset(' parameter to read beyond entry \d+\)\n<\/entries>)$/,
      (_match, before, after) => before + offset + after,
    );
  }
  if (metadata.display.type !== "file") return text;
  // The pinned read tool emits this footer outside its line-numbered file data.
  // Do not rewrite the same phrase in file bytes, reminders or shell streams.
  return text.replace(
    /(\n\n\((?:Output capped at [^\n]+\. Showing lines \d+-\d+|Showing lines \d+-\d+ of \d+)\. Use )offset(=\d+ to continue\.\)\n<\/content>)(?=\n\n<system-reminder>|$)/,
    (_match, before, after) => before + offset + after,
  );
}

function presentedError(profile, toolID, text, mapped, args) {
  if (typeof text !== "string" || (profile.literal_errors || []).includes(text)) return text;
  // Older frozen prefixes predate guard-error receipts. Their exact authored
  // refusal is already model-facing text, not a native diagnostic to rename.
  if (profile.read_first?.[toolID]) {
    const definition = profile.tools?.[toolID];
    const native = definition?.args_map ? toNative(definition, args || {}) : args;
    if (typeof native?.filePath === "string" && Object.values(profile.read_first[toolID]).some(
      message => message.replaceAll("{{path}}", () => native.filePath) === text)) return text;
  }
  if (profile.visible_tools === undefined) return mapped ? renameNative(profile.tools[toolID], toolID, text) : text;
  const visible = profile.visible_tools?.[toolID] || toolID;
  if (text.startsWith(`${visible} failed: `)) return text;
  const denied = "The user has specified a rule which prevents you from using this specific tool call.";
  if (text.startsWith(denied)) return `${visible} failed: ${denied}`;
  // Only native-generated argument errors are renamed. Nonzero command output
  // is a completed tool result and never enters this path.
  let detail = text.replace(/^The \S+ tool was called with invalid arguments: /, "Invalid arguments: ");
  detail = renameNative(mapped ? profile.tools[toolID] : undefined, toolID, detail, profile.visible_tools);
  return `${visible} failed: ${detail}`;
}

function record(entry) {
  try {
    fs.appendFileSync(APPLIED_PATH, JSON.stringify(entry) + "\n");
  } catch {
    // The record is evidence only; never fail a run over it.
  }
}

function resolveFile(directory, given) {
  const absolute = path.resolve(directory, given);
  try { return fs.realpathSync(absolute); } catch { return absolute; }
}

function isRegularFile(file) {
  try { return fs.statSync(file).isFile(); } catch { return false; }
}

function contentSha256(file) {
  try { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); } catch { return null; }
}

function recordPlant(entry) {
  try {
    fs.appendFileSync(PLANTS_FIRED_PATH, JSON.stringify(entry) + "\n");
  } catch {
    // Never let the record change what the model sees; a missing "loaded" line fails the run on the agent's side.
  }
}

function loadPlants() {
  if (!PLANTS_PATH) return [];
  try {
    const plants = JSON.parse(fs.readFileSync(PLANTS_PATH, "utf8")).map((plant, index) => {
      if (plant.action !== "fail") throw new Error(`unsupported plant action: ${plant.action}`);
      const arg = plant.match.arg;
      return {
        index,
        id: plant.id,
        tool: plant.match.tool,
        field: arg ? arg.field : null,
        regex: arg ? new RegExp(arg.regex) : null,
        nth: plant.nth ?? 1,
        action: plant.action,
        text: plant.text,
        matched: 0,
      };
    });
    recordPlant({ kind: "loaded", plants: plants.length });
    return plants;
  } catch (error) {
    recordPlant({ kind: "error", message: String((error && error.message) || error) });
    return [];
  }
}

// The plant that acts on this call, if any. `native` holds the call's native arguments; a plant with an argument test
// matches only when that argument is present and its value (a string as is, anything else as JSON) matches the regex.
function duePlant(plants, toolID, native) {
  let due = null;
  for (const plant of plants) {
    if (plant.tool !== toolID) continue;
    if (plant.regex) {
      const value = native && typeof native === "object" ? native[plant.field] : undefined;
      if (value === undefined || value === null) continue;
      if (!plant.regex.test(typeof value === "string" ? value : JSON.stringify(value))) continue;
    }
    plant.matched += 1;
    if (plant.matched === plant.nth && due === null) due = plant;
  }
  return due;
}

function recordFiring(plant, input, callNumber) {
  recordPlant({
    kind: "fired",
    index: plant.index,
    ...(plant.id ? {event_id: plant.id} : {}),
    tool: input.tool,
    action: plant.action,
    call_number: callNumber,
    call_id: input.callID ?? null,
    session_id: input.sessionID ?? null,
  });
}

export const IfProfile = async (input) => {
  const directory = (input && input.directory) || process.cwd();
  const profile = load();
  if (!profile) return {};
  record({kind: "loaded", profile_sha256: profile.profile_sha256});
  const tools = profile.tools || {};
  const reshaped = (id) => tools[id] && tools[id].args_map;
  const key = value => JSON.stringify([value.sessionID, value.callID]);
  const pending = new Map(); // session + call ID -> configured arguments
  const readFirst = profile.read_first;
  const readFiles = new Map(); // session + call -> canonical native file
  const known = new Map(); // session -> file -> last observed content hash (or null if unreadable)
  const seeded = new Set();
  const guardErrors = new Map();
  const knownIn = sessionID => {
    if (!known.has(sessionID)) known.set(sessionID, new Map());
    return known.get(sessionID);
  };
  const readFirstCheck = (input, native) => {
    if (!readFirst || !["read", "edit", "write"].includes(input.tool)) return null;
    const given = native?.filePath;
    if (typeof given !== "string" || !given) return null;
    const file = resolveFile(directory, given);
    const cases = readFirst[input.tool];
    if (cases && isRegularFile(file)) {
      const seen = knownIn(input.sessionID);
      let message = !seen.has(file) ? cases.unread : null;
      if (seen.has(file) && cases.changed && seen.get(file) !== null) {
        const now = contentSha256(file);
        if (now !== null && now !== seen.get(file)) message = cases.changed;
      }
      if (message) {
        const text = message.replaceAll("{{path}}", () => given);
        guardErrors.set(key(input), text);
        record({kind: "tool_receipt", session_id: input.sessionID, call_id: input.callID, native_tool: input.tool,
          metadata: {model_error: text, read_first_rejected: true}});
        throw new Error(text);
      }
    }
    return file;
  };
  const recorded = new Set();
  const errorPresentations = new Map();
  const rejectedAliases = new Map();
  const prefixSession = process.env.NEMO_GYM_IF_PREFIX_SESSION;
  // A restored prefix keeps its frozen historical receipts.
  const prefixReceipts = profile.prefix_receipts || [];
  const plants = loadPlants();
  let calls = 0; // every tool call this run, in the order OpenCode starts them (call_number in a firing record)

  async function checkAlias(input) {
    const visible = profile.visible_tools?.[input.tool];
    if (!profile.check_aliases || !visible || visible === input.tool) return;
    let verdict;
    try {
      const url = process.env.NEMO_GYM_IF_TOOL_CALL_URL;
      const token = process.env.NEMO_GYM_IF_TOOL_CALL_TOKEN;
      if (!url || !token) throw new Error("missing route");
      const response = await fetch(url, {
        method: "POST", headers: {"content-type": "application/json", authorization: `Bearer ${token}`},
        body: JSON.stringify({session_id: input.sessionID, call_id: input.callID, native_tool: input.tool}),
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) throw new Error("unavailable route");
      verdict = await response.json();
      if (typeof verdict.allowed !== "boolean" || typeof verdict.model_name !== "string" ||
          !Array.isArray(verdict.available_names) || !verdict.available_names.every(x => typeof x === "string")) {
        throw new Error("invalid receipt");
      }
    } catch {
      throw new Error("Tool call provenance unavailable; execution was not started.");
    }
    if (!verdict.allowed) {
      const error = `${verdict.model_name} is unavailable. Available tools: ${verdict.available_names.join(", ") || "none"}.`;
      rejectedAliases.set(key(input), error);
      record({kind: "alias_rejected", session_id: input.sessionID, call_id: input.callID,
        native_tool: input.tool, model_name: verdict.model_name, error});
      throw new Error(error);
    }
  }

  const hooks = {
    "tool.definition": async (input, output) => {
      const tool = tools[input.toolID];
      if (!tool) return;
      if (tool.parameters) output.jsonSchema = structuredClone(tool.parameters);
      if (!recorded.has("tool:" + input.toolID)) {
        recorded.add("tool:" + input.toolID);
        record({
          kind: "tool",
          profile_sha256: profile.profile_sha256,
          tool: input.toolID,
          description: output.description,
          jsonSchema: output.jsonSchema ?? null,
        });
      }
    },

    "tool.execute.before": async (input, output) => {
      await checkAlias(input);
      calls += 1;
      const tool = tools[input.tool];
      if (reshaped(input.tool)) {
        const errors = schemaErrors(tool.parameters, output.args);
        if (errors.length) {
          // The same wording as OpenCode's own invalid-arguments error (tool/tool.ts InvalidArgumentsError).
          throw new Error(
            `The ${input.tool} tool was called with invalid arguments: ${errors.join("; ")}.\n` +
              "Please rewrite the input so it satisfies the expected schema.",
          );
        }
      }
      // Compute once, before counting an event: a call the read-first guard
      // refuses must not consume an occurrence.
      const native = reshaped(input.tool) ? toNative(tool, output.args) : output.args;
      const readFile = readFirstCheck(input, native);
      const plant = duePlant(plants, input.tool, native);
      if (plant) {
        // Before any mapping: the stored input stays what the model sent, and the text is shown exactly.
        recordFiring(plant, input, calls);
        throw new Error(plant.text);
      }
      if (readFile) readFiles.set(key(input), readFile);
      if (!reshaped(input.tool)) return;
      const configured = JSON.parse(JSON.stringify(output.args));
      // OpenCode calls the tool with the same object it hands us, so change it in place.
      for (const key of Object.keys(output.args)) delete output.args[key];
      Object.assign(output.args, native, { [STASH_KEY]: { configured_args: configured, names: tool.args_map } });
      pending.set(key(input), configured);
    },

    "tool.execute.after": async (input, output) => {
      const configured = pending.get(key(input));
      pending.delete(key(input));
      let diff = null;
      if (FILE_TOOLS.has(input.tool)) {
        try {
          diff = workspaceDiff(directory);
        } catch {
          diff = null; // not a git repository: no replay record
        }
      }
      if (configured && output && typeof output.output === "string") {
        output.output = renameHints(tools[input.tool], input.tool, output.output, output.metadata);
      }
      const readFile = readFiles.get(key(input));
      readFiles.delete(key(input));
      let readRecord = null;
      // Native failures never call this hook.
      if (readFile && isRegularFile(readFile)) {
        readRecord = {file: readFile, sha256: contentSha256(readFile)};
        knownIn(input.sessionID).set(readFile, readRecord.sha256);
      }
      if (!configured && !profile.profile_sha256 && diff === null && !readRecord) return;
      output.metadata = output.metadata && typeof output.metadata === "object" ? output.metadata : {};
      output.metadata[META_KEY] = {
        profile_sha256: profile.profile_sha256 ?? null,
        ...(configured ? { configured_args: configured } : {}),
        ...(readRecord ? { read_first: readRecord } : {}),
        ...(diff !== null
          ? { workspace_diff: diff, workspace_diff_sha256: crypto.createHash("sha256").update(diff).digest("hex") }
          : {}),
      };
      record({kind: "tool_receipt", session_id: input.sessionID, call_id: input.callID, native_tool: input.tool,
        metadata: output.metadata[META_KEY]});
    },

    "experimental.chat.messages.transform": async (_input, output) => {
      const messages = output.messages || [];
      for (const message of output.messages || []) {
        for (const part of message.parts || []) {
          if (part.type !== "tool" || !part.state) continue;
          const sessionID = part.sessionID ?? message.info?.sessionID;
          if (part.state.status === "error" && sessionID === prefixSession) {
            const prior = prefixReceipts.find(row => row.call_id === part.callID && row.native_tool === part.tool);
            if (prior) {
              part.state.metadata ??= {};
              part.state.metadata[META_KEY] ??= structuredClone(prior.metadata);
            }
          }
          const guardError = guardErrors.get(key({sessionID, callID: part.callID}));
          if (part.state.status === "error" && guardError) {
            part.state.error = guardError;
            continue;
          }
          const rejectedAlias = rejectedAliases.get(key({sessionID: part.sessionID ?? message.info?.sessionID, callID: part.callID}));
          if (part.state.status === "error" && rejectedAlias) {
            part.state.error = rejectedAlias;
            continue;
          }
          // Invalid-tool recovery is presented by the gateway, which owns the
          // original model call and its actual per-request (including child)
          // tool inventory. A profile-wide list here would be misleading.
          const stamp = part.state.metadata && part.state.metadata[META_KEY];
          const stash = part.state.input && part.state.input[STASH_KEY];
          if (stamp && stamp.configured_args) {
            part.state.input = stamp.configured_args;
          } else if (stash && stash.configured_args) {
            part.state.input = stash.configured_args;
          }
          if (part.state.status === "error" && (typeof stamp?.model_error === "string" || stamp?.dispatch_rejected)) {
            part.state.error = stamp.model_error ?? stamp.error;
            continue;
          }
          // Otherwise the plugin rejected the call before mapping it: the stored input is exactly what the model sent
          // (possibly native names), shown as is next to its error.
          // An error OpenCode wrote for a call that reached the tool names native arguments ("oldString and newString are
          // identical"). Only those calls carry the stash; the plugin's own rejections are already in the profile's names.
          if (part.state.status === "error" && (profile.visible_tools !== undefined || reshaped(part.tool))) {
            const nativeError = part.state.error;
            const modelError = presentedError(profile, part.tool, nativeError, !!stash, part.state.input);
            part.state.error = modelError;
            // History transforms do not persist into OpenCode's native export.
            // Keep an exact receipt rather than reconstruct the displayed error
            // later, including when compaction is its only model consumer.
            if (typeof sessionID === "string" && typeof part.callID === "string" && modelError !== nativeError) {
              const identity = key({sessionID, callID: part.callID});
              const receipt = {kind: "tool_presentation", session_id: sessionID, call_id: part.callID,
                native_tool: part.tool, native_error: nativeError, model_error: modelError};
              const encoded = JSON.stringify(receipt);
              if (errorPresentations.get(identity) !== encoded) {
                record(receipt);
                errorPresentations.set(identity, encoded);
              }
            }
          }
        }
      }
      // Imported native completed-tool metadata survives OpenCode restore.
      // Seed each session once; later calls update state only after execution.
      if (readFirst && messages.length) {
        const sessionID = messages[0].info?.sessionID ?? messages[0].parts?.[0]?.sessionID;
        if (sessionID && !seeded.has(sessionID)) {
          seeded.add(sessionID);
          const seen = knownIn(sessionID);
          for (const message of messages) {
            for (const part of message.parts || []) {
              if ((part.sessionID ?? message.info?.sessionID) !== sessionID) continue;
              const state = part.type === "tool" && part.state;
              const rec = state?.status === "completed" && state.metadata?.[META_KEY]?.read_first;
              if (rec && typeof rec.file === "string") seen.set(rec.file, typeof rec.sha256 === "string" ? rec.sha256 : null);
            }
          }
        }
      }
    },
  };
  return hooks;
};
