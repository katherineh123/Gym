# Agentic IF Environment Server

This Environment Server runs one sealed agentic instruction-following task in the OpenCode harness. It seeds the
task's sandbox through the task resources server (Scale-SWE or Terminal-Bench 2.1), runs OpenCode there with its
model calls routed to the policy model, and records the session as evidence.

Each sandbox downloads the pinned OpenCode itself unless `AGENTIC_IF_OPENCODE_BINARY` names a host copy to upload.

The paired Resources Server in `resources_servers/agentic_if` validates each row and grades the evidence. See its
README for settings and run instructions.
