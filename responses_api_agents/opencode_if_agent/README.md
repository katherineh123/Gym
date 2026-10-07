# OpenCode instruction-following variants

A thin adapter around Gym's **real OpenCode CLI**, using OpenSandbox and the existing
Scale-SWE or SWE-rebench resource server. A shared builder composes prompt-family,
tool-name, and instruction variations before the rollout. Coding-task verification
and hosted rubric judging remain separate.

## Generate and run

Start with an existing native resource-server row; do not rename dataset fields or
convert one dataset's verifier contract into another's. The public example rows
already in Gym are suitable inputs:

```bash
python -m nemo_gym.task_variants.generate \
  --input resources_servers/scale_swe/data/example.jsonl \
  --specs responses_api_agents/opencode_if_agent/data/specs.json \
  --agent-name opencode_if_agent --output /path/to/scale-if.jsonl

# Set NVIDIA_API_KEY, OPENSANDBOX_DOMAIN and OPENSANDBOX_API_KEY privately.
gym env start --config responses_api_agents/opencode_if_agent/configs/scale_swe.yaml

# In another terminal, after server startup:
gym eval run --no-serve --agent opencode_if_agent \
  --input /path/to/scale-if.jsonl --output /path/to/scale-if-rollouts.jsonl \
  --num-repeats 1 --concurrency 2
```

For rebench, use `resources_servers/swe_rebench/data/example.jsonl` and
`configs/swe_rebench.yaml`. The source examples use DockerHub images. Existing ECR
references are preserved too; this adapter neither mirrors nor rebuilds images.
Remote sandboxes must reach the Gym agent/model endpoints. `use_absolute_ip: true`
advertises a host address but does not establish routing or firewall access.

Both configs inherit upstream OpenCode permissions and the original resource
server settings, including anti-cheating. The IF server uses **one worker**, with
concurrent run-specific routes; scale via collection concurrency or separate
instances, not multiple workers sharing an in-memory route table.

## Variant contract

Each specification selects `harness: opencode`, a prompt family, a tool-name mapping,
and zero or more instructions. The generator expands the explicit task × spec
product, refuses duplicate variants/overwrites, and never resamples at rollout time.
`seed` is recorded experiment provenance, not a hidden runtime RNG. Legacy
OpenHands agent classes are rejected, not represented as OpenCode personas.

```json
{
  "tool_names": {"bash": "run_command"},
  "instructions": [{
    "id": "explain-shell",
    "instruction_text": "Explain the purpose of each ${tool:bash} call in the accompanying message.",
    "placement": {"surface": "user_prompt", "position": "end"},
    "rubric": "Every shell call needs an accompanying explanation of its purpose. An unexplained call fails. With no shell calls, mark not applicable.",
    "taxonomy": ["IF-REASONACT"]
  }]
}
```

System and user instructions can coexist. `start`/`end` specify placement around
the selected prompt. Tool-description placement adds `"tool": "bash"` and uses
`"surface": "tool_description"`; the logical tool name stays `bash` even when its
model-visible alias differs. Tool output and replay/mid-task injection are not
supported in this initial fresh-task runtime.

Prompt families can override `system` and/or `user`. User replacement requires
explicit `task_rules`, a `problem_statement` in the base row, and `${issue}` plus
`${rules}` placeholders: existing templated prompts are never heuristically stripped.
Other supported placeholders are `${workdir}` and `${tool:bash}` (and other native
logical tools). `${workdir}` requires an explicit base-row workdir; no missing path
is silently substituted. Templates are plain text, not executable Jinja. Replacing the
system family uses OpenCode's native build-agent prompt setting; OpenCode retains
its environment/reminder handling. With no override, the native prompt is retained.

Tool aliases are applied to model-visible schemas, tool history and tool choices,
then reversed in the response before native dispatch, including streamed tool names.
Arguments, tool outputs and arbitrary issue/source text are not rewritten. Unsupported
or colliding aliases fail explicitly. No rubric or golden patch enters the actor prompt.

Validation catches structural/profile incompatibilities, not every semantic conflict
between natural-language instructions. Review authored IF constraints against the
task's rules before collecting data; deliberate instruction-hierarchy conflict
experiments need their own explicitly designed rubrics.

## Results and limits

- `reward` and resource-specific test fields remain the native coding-task results.
- `if_result` contains hosted GLM 5.3 judgments: `pass`, `fail`, `not_applicable`, or
  `error`. Empty instruction lists skip judging. Missing/truncated/oversized evidence
  and failed API calls never become passes. No implicit combined training reward is
  imposed; downstream training must choose its reward policy explicitly.
- `task_variant` freezes the source digest, choices, rendered-input digest, taxonomy,
  rubrics and stable variant ID. `variant_receipt` adds an attempt ID, OpenCode version,
  the actual first tool-bearing actor request (not an auxiliary title request), and
  hashes/count of all model requests. Receipts
  are private rollout artifacts; they contain task text/tool schemas, not API keys.
- Exported response/trajectory tool names use model-visible aliases. Raw OpenCode
  logs and observation artifacts retain native names. Session/assistant-message
  correlation headers are forwarded to preserve Gym's model-call attribution.
- Delegated `task` calls require complete, gap-free child-session trajectory evidence
  for IF judging; otherwise IF returns an error rather than overlooking child actions.
  Upstream observation capture currently needs `python3` in the task image. Root-only
  rollouts can still be graded from their native export when optional telemetry is
  unavailable. Keep this distinction when selecting images or auditing SFT data.
- The judge uses `https://inference-api.nvidia.com/v1/chat/completions` with
  `nvidia/zai-org/glm-5.3`. Endpoint/model/token budget/concurrency can be overridden
  under `judge`. It is a hosted model, not a locally deployed GPU judge.

This is not a complete port of every legacy prompt/persona combination. Choose
native-compatible families and explicit tool placeholders; do not feed old
OpenHands prompts/tool names unmodified. Historical IF metadata can be migrated:

```bash
python -m nemo_gym.task_variants.migrate_charlie \
  --params /private/original.params.json --rows /private/original.jsonl \
  --output-dir /private/new-rubric-snapshot
```

Migration preserves saved wording, original positions, continuation prefixes and
provenance; it never mutates source artifacts or claims old SIF/replay rows are
native-runnable. Generate new runnable variants from OCI-backed native dataset rows.

```bash
python -m pytest responses_api_agents/opencode_if_agent/tests -q
```
