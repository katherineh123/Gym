# AdvancedIF

[AdvancedIF](https://arxiv.org/abs/2511.10507) (He et al., Meta Superintelligence Labs, with data from Surge AI)
measures instruction following with expert-written rubrics. The model writes the final assistant turn of a
conversation, and an LLM judge checks that turn against every rubric for the conversation. A task passes only when
the judge finds all rubrics satisfied.

| Category (`category`) | Dataset label (`benchmark_name`) | Tasks | What the model must follow |
|---|---|---|---|
| `complex_if` | `complex_if_single_turn_v5` | 402 | Six or more stacked instructions in a single user turn |
| `carried_context` | `carried_context_multi_turn_eval_v5` | 736 | Instructions spread over a multi-turn conversation |
| `system_steerability` | `system_steerability_v2` | 507 | Rules in a system prompt, over a multi-turn conversation |

- Integration profile: `custom-gym-verifier`
- Scorer: [`resources_servers/advanced_if`](../../resources_servers/advanced_if/README.md)

## Provenance

- **Data:** [`facebook/AdvancedIF`](https://huggingface.co/datasets/facebook/AdvancedIF) at revision
  `e20cba9b94b59c027dfab00b29244e8bc42e4ab4`, the single file `if_oss_full_data.csv`. All 1,645 rows are used,
  with no filtering, in file order (`task_id` is the row index). License: CC BY-NC 4.0.
- **Judge prompts:** `JUDGE_PROMPT`, `SYSTEM_STEER_JUDGE_PROMPT` and `STEER_FEW_SHOT_EXAMPLES` from the official scorer,
  [`facebookresearch/AdvancedIF` `judge.py`](https://github.com/facebookresearch/AdvancedIF/blob/f9d30137c4139d4d9af260ae28108b5afae828c0/judge.py)
  at commit `f9d30137c4139d4d9af260ae28108b5afae828c0`. License: CC BY-NC 4.0.
- Neither is committed. `prepare.py` downloads both at the pinned revisions. It checks the sha256 of `judge.py` and
  reads the three strings from its syntax tree without importing or running it. It writes
  `data/advanced_if_benchmark.jsonl` and `data/advanced_if_judge_prompts.json`.

## Task contract

- **Model input:** `responses_create_params.input` is the dataset's `conversation_history` unchanged. That is every
  prior user and assistant turn, ending with the final user turn; `system_steerability` rows start with their system
  message. No prompt template or extra system prompt is added. The prior assistant turns are fixed text from the
  dataset and are not regenerated.
- **Hidden from the model:** the rubrics (`rubrics`) and the category.
- **Graded output:** the text of the final assistant message. Reasoning items and `<think>` blocks are not shown to
  the judge.

## Scoring

The judge is **o3-mini**, as the paper specifies (section 3.2, "we choose o3-mini as our rubric verifier"); the
official scorer pins `o3-mini-2025-01-31`. The request matches the official scorer: one Chat Completions call per
task, the rendered prompt as the only user message, `response_format: {"type": "json_object"}`,
`max_completion_tokens: 32768`, and no temperature, seed or reasoning effort (o3-mini's default effort is medium).

- `system_steerability` tasks use the system-steer prompt with its worked examples. The judge sees the system prompt,
  the final user turn, the reply and the rubrics.
- The other two categories use the user-instruction prompt. The judge sees the earlier turns as `role [n]: text`, the
  final user turn, the reply and the rubrics.
- **Reward:** 1.0 when the judge's `SATISFIED_ALL_REQUIREMENTS` is `yes`, else 0.0. The range is [0, 1]; higher is
  better.

### Differences from running the official scorer on the Hugging Face data

1. **Judge routing.** The official script uses the system-steer prompt only for rows labeled
   `if_system_steerability_oss`. The Hugging Face rows are labeled `system_steerability_v2`, so the script as released
   judges them with the user-instruction prompt. The official README, and the `SystemSteerIFRubricsJudge` docstring
   ("Used for system_steerability_v2 task"), say system-steerability rows use the system-steer prompt; Gym follows
   them.
2. **Unusable judge replies.** A reply that is not the JSON verdict object is requested again, up to 3 calls, and a
   JSON object inside a Markdown code fence is accepted. The official script makes one call and scores the row 0.
   Rows that still fail score 0 and are counted in `judge_parse_failed_rate`.
3. **Unfinished policy replies.** A reply cut off by the output-token cap, or one with no visible text, scores 0
   without a judge call. This includes replies an endpoint's content filter stopped before any text. These rows
   are counted in `policy_max_output_tokens_rate`, `policy_content_filter_rate` and `empty_answer_rate`.
4. **Judge outages.** A judge call that fails (transport, timeout or HTTP error after retries) masks the row instead
   of scoring it 0. Rerun those rows with `gym eval reverify --judge-failed-only`.

## Metrics

| Metric | Meaning |
|---|---|
| `category_mean/pass@1[avg-of-k]/reward` | **Headline.** Unweighted mean of the three category pass rates, as reported in the paper. |
| `<category>/pass@1[avg-of-k]/reward` | Pass rate within one category. |
| `pass@1[avg-of-k]/reward` | Pass rate over all 1,645 tasks with equal weight per task. Mercor's AdvancedIF leaderboard reports this aggregation. |
| `judge_parse_failed_rate`, `empty_answer_rate`, `policy_max_output_tokens_rate`, `policy_content_filter_rate` | Rows that scored 0 without a usable verdict, and why. |

Pass rates are percentages. With `num_repeats` k > 1, pass@1[avg-of-k] averages over the k samples of each task.

## Running

```bash
gym eval prepare --benchmark advanced_if
gym eval run --benchmark advanced_if --model-type vllm_model --config /abs/path/models.yaml \
    --split benchmark --concurrency 32 --max-output-tokens 64000 \
    --output results/advanced_if/<model>/rollouts.jsonl
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
      extra_body:
        reasoning_effort: high
judge_base_url: https://inference-api.nvidia.com/v1
judge_api_key: ${oc.env:NVIDIA_API_KEY}
judge_model_name: azure/openai/o3-mini
```

Without `judge_*` settings the judge defaults to `o3-mini-2025-01-31` on `https://api.openai.com/v1`; set
`judge_api_key` to an OpenAI key. Each task makes one policy call and one to three judge calls. The
system-steerability judge prompt carries about 7,000 tokens of worked examples, so a full pass sends roughly 6M
input tokens to the judge.

## Reference results

AdvancedIF paper (arXiv v2, Table 3; o3-mini judge; "Thinking" mode):

| Model | CIF | CC | SS | Avg |
|---|---|---|---|---|
| GPT-5 | 86.9 | 73.9 | 72.8 | 77.9 |
| Gemini 3 Pro | 83.4 | 73.3 | 67.3 | 74.7 |
| Gemini 2.5 Pro | 78.5 | 67.1 | 59.5 | 68.4 |
| Claude 4 Sonnet | 75.9 | 61.8 | 53.8 | 63.8 |
| GPT-4.1 | 72.1 | 57.1 | 59.4 | 62.9 |

The paper does not publish the evaluated models' sampling settings.
