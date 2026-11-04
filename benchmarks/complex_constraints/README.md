# ComplexConstraints

[ComplexConstraints](https://surgehq.ai/blog/complexconstraints-a-benchmark-for-entangled-instruction-following)
(Mehta et al., Surge AI, [arXiv:2606.09118](https://arxiv.org/abs/2606.09118)) measures instruction following on
realistic professional requests: work rotas, procurement plans, crisis memos, lesson plans. Each of the 75 prompts
carries many constraints at once, and they interact: conditional rules, multistep work, things the reply must not do,
and requirements the user only implies. Expert-written rubrics check every constraint, 10 to 40 criteria per prompt
(1,559 in total, median 19). A task passes only when the reply satisfies every criterion.

| Constraint type (`instruction_type`) | Prompts |
|---|---|
| Negative | 22 |
| Multistep | 20 |
| Conditional | 19 |
| Implicit | 14 |

- Integration profile: `custom-gym-verifier`
- Scorer: [`resources_servers/complex_constraints`](../../resources_servers/complex_constraints/README.md)

## Provenance

- **Data:** the 75-prompt benchmark set, license CC BY 4.0 (paper, "Data Availability"). Surge published it as
  [`surgeai/ComplexConstraints`](https://huggingface.co/datasets/surgeai/ComplexConstraints) (one CSV,
  `ComplexConstraints_benchmark_set.csv`), but that repository no longer serves anonymous users. `prepare.py` reads the
  copy of that CSV saved as `data/ComplexConstraints.xlsx` in the MIT-licensed repository
  [`smarinacseas/failure-mode-id`](https://github.com/smarinacseas/failure-mode-id/blob/49eb91959b7c969adab9a0bad85002647be7e6f9/data/ComplexConstraints.xlsx)
  at commit `49eb91959b7c969adab9a0bad85002647be7e6f9`. It checks the file's sha256 and the published counts (75
  prompts, `CIF-001` to `CIF-075`, 1,559 criteria). The copy has the official file's 45 columns: `benchmark_id`,
  `prompt`, `use_case`, `instruction_type`, `prompt_style` and `criterion_1` to `criterion_40`. All 75 rows are used,
  in file order.
- **Official data instead of the copy:** with a Hugging Face token that has access to the official dataset
  (`HF_TOKEN`), run `COMPLEX_CONSTRAINTS_DATA_SOURCE=huggingface gym eval prepare --benchmark complex_constraints`.
  `prepare.py` then reads `ComplexConstraints_benchmark_set.csv` at revision
  `e9625c6f635f42b72cb85a04c2be64746f945126` and checks the same counts. Each prepared row records its source in
  `data_source`.
- **Judge prompt:** `JUDGE_PROMPT_TEMPLATE` and `RESPONSE_SCHEMA` from the official harness,
  [`surge-ai/complex-constraints` `scorer.py`](https://github.com/surge-ai/complex-constraints/blob/9ef1dd73d20030be94bdc399fd1c6ef7970f68d1/src/complex_constraints/scorer.py)
  at commit `9ef1dd73d20030be94bdc399fd1c6ef7970f68d1`. That repository carries no license, so the prompt is not
  committed here. `prepare.py` checks the file's sha256 and reads both values from its syntax tree without importing or
  running it.
- `prepare.py` writes `data/complex_constraints_benchmark.jsonl` and `data/complex_constraints_judge_prompt.json`;
  neither is committed.

## Task contract

- **Model input:** `responses_create_params.input` is the dataset's `prompt` as the only user message. No system
  prompt or template is added, as in the official harness.
- **Hidden from the model:** the criteria (`rubrics`) and the labels (`use_case`, `instruction_type`, `prompt_style`).
- **Graded output:** the text of the final assistant message. Reasoning items and `<think>` blocks are not shown to
  the judge.

## Scoring

The judge is **Gemini 3.5 Flash**, the official harness's default (`DEFAULT_JUDGE_MODEL = "google/gemini-3.5-flash"`).
The paper does not name the judge behind its benchmark results; it names GPT-5-mini only as the reward judge in its
RL training. The request matches the official scorer:

- One Chat Completions call per criterion. The judge sees the reply and that one criterion, never the prompt.
- The rendered template is the only user message, with structured output (`response_format` `json_schema`, the
  official `individual_criteria_score` schema: `rationale` and `score`, a string `"0"` or `"1"`). No temperature,
  token cap or reasoning effort is set, so the provider's defaults apply.
- A criterion passes when its `score`, rounded half-up to four places, is at least 1. A reply that fails the official
  strict parse is requested again, up to five calls in all.
- **Reward:** 1.0 when every criterion passes, else 0.0 (the leaderboard's task pass rate). `mean_criteria` is the
  fraction of criteria passed (the paper's mean per-criterion pass rate).

### Differences from the official harness

1. **No visible reply.** A reply with no visible text (only reasoning, an unclosed `<think>` block, or nothing) scores
   0 on both metrics without judge calls. The official harness judges the empty string against each criterion, which
   can pass criteria that only forbid something. Such rows are counted in `empty_answer_rate`. A reply cut off by the
   output cap is graded as written, as in the official harness, and counted in `policy_max_output_tokens_rate`.
2. **Unscored criteria.** When a criterion is still unparseable after five calls, the official harness leaves the row
   out of the task pass rate but keeps its scored criteria in `mean_criteria`. Gym masks the row and routes it to the
   failures sidecar, so it leaves both metrics until a resume judges it again.
3. **Judge outages.** A judge call that fails (transport, timeout or HTTP error after retries) also masks the row
   instead of scoring it.

## Metrics

| Metric | Meaning |
|---|---|
| `pass@1[avg-of-k]/reward` | **Headline.** Task pass rate: the share of tasks whose reply satisfies every criterion (the official `all_pass`). |
| `pass@1[avg-of-k]/mean_criteria` | Mean fraction of criteria satisfied per reply (the official `mean_criteria`). |
| `<instruction_type>/pass@1[avg-of-k]/reward` | Task pass rate within one constraint type. |
| `empty_answer_rate`, `policy_max_output_tokens_rate`, `policy_content_filter_rate`, `judge_retry_rate` | Replies with no visible text or cut off, and judge replies that had to be requested again. |

Pass rates are percentages; the `*_rate` metrics are fractions of rollouts (of criterion grades for
`judge_retry_rate`). With `num_repeats` k > 1, pass@1[avg-of-k] averages over the k samples of each task, as the
official harness's `--epochs` with the mean reducer does.

## Running

```bash
gym eval prepare --benchmark complex_constraints
gym eval run --benchmark complex_constraints --model-type vllm_model --config /abs/path/models.yaml \
    --split benchmark --concurrency 32 --output results/complex_constraints/<model>/rollouts.jsonl
```

`models.yaml` sets the policy and the judge. Keep it outside the repository, because it names your API keys. For
example, with NVIDIA's inference API:

```yaml
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: ${oc.env:NVIDIA_API_KEY}
policy_model_name: <policy model id>
policy_model:
  responses_api_models:
    vllm_model:
      uses_interleaved_reasoning: false
      return_token_id_information: false
judge_base_url: https://inference-api.nvidia.com/v1
judge_api_key: ${oc.env:NVIDIA_API_KEY}
judge_model_name: gcp/google/gemini-3.5-flash
```

Without `judge_*` settings the judge is `gemini-3.5-flash` on Google's OpenAI-compatible endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai`); set `judge_api_key` to a Gemini API key. Each task makes
one policy call and one judge call per criterion, 1,559 judge calls per pass over the benchmark.

Verification uses only the saved reply (`REVERIFY_MODE` is stateless), so a run can be judged again by another judge
without new policy calls. Point `judge_*` at the other judge in a second config and write to a new output:

```bash
gym eval reverify --benchmark complex_constraints --model-type vllm_model \
    --config /abs/path/models.yaml --config /abs/path/other_judge.yaml \
    --inputs results/complex_constraints/<model>/rollouts_materialized_inputs.jsonl \
    --rollouts results/complex_constraints/<model>/rollouts.jsonl \
    --output results/complex_constraints/<model>_<judge>/rollouts.jsonl
```

## Reference results

Surge's [leaderboard](https://surgehq.ai/benchmarks/complex-constraints) (task pass rate; it states no date, judge or
sampling settings) lists, among others, GPT 6 Astra (Max reasoning) 70.8, Claude Opus 5.5 (Adaptive/Max) 64.4,
GLM 5.3 (Max reasoning) 54.4, Kimi K3 (Max reasoning) 51.1 and Qwen 3.8 Max (xHigh reasoning) 46.3. The paper's
Table 1 is an earlier snapshot of the same leaderboard (June 3, 2026; top score 40.4).
