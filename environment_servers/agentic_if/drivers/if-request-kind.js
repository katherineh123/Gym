// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Upstream's assistant-message plugin (opencode_agent/assistant_message_header.js, staged next to this file as
// assistant-message-header.mjs) plus the X-OpenCode-Request-Kind header that only the IF model gateway reads.
// OpenCode registers every exported function of a plugin module, so this module exports the wrapper alone.
import { AssistantMessageHeader } from "./assistant-message-header.mjs";

export const IfRequestKind = (...args) => {
  const hooks = AssistantMessageHeader(...args);
  const assistantHeader = hooks["chat.headers"];
  return {
    ...hooks,
    "chat.headers"(input, output) {
      assistantHeader(input, output);
      // Upstream sets the id only when exactly one saved assistant owns the call; the kind follows the same rule.
      if (output.headers["X-OpenCode-Assistant-Message-Id"] === undefined) return;
      // Compaction has a saved assistant too, but is not a task-solving turn.
      output.headers["X-OpenCode-Request-Kind"] = input.agent === "compaction" ? "compaction" : "actor";
    },
  };
};
