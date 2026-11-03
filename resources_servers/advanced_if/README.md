# advanced_if resources server

Scores the final assistant turn of an [AdvancedIF](https://arxiv.org/abs/2511.10507) conversation with the
benchmark's rubric judge. The benchmark workload, data preparation and run commands are in
[`benchmarks/advanced_if`](../../benchmarks/advanced_if/README.md).

## Verification

- `verify()` takes the last assistant message, without reasoning items or `<think>` blocks. A reply cut off by the
  output-token cap, or one with no visible text, scores 0 without a judge call.
- It renders the official judge prompt for the row's `category` and sends it as one Chat Completions user message
  to `judge_model_server` (default request: JSON mode, `max_completion_tokens: 32768`, no temperature).
  - `complex_if` and `carried_context` rows use the user-instruction prompt. It shows every turn before the final user
    turn as `role [n]: text`, then the final user turn, the reply and the rubrics as indented JSON.
  - `system_steerability` rows use the system-steer prompt with its four worked examples. It shows the system prompt,
    the final user turn, the reply and the rubrics.
- The reward is 1.0 when the judge's `SATISFIED_ALL_REQUIREMENTS` field is `yes` (case-insensitive), else 0.0.
- A judge reply that is not the JSON verdict object is requested again, up to `judge_max_attempts` calls. After
  that the row scores 0 and `judge_parse_failed` is true; the official scorer also counts these rows as failed.
  A failed judge call (transport, timeout, HTTP error) raises, so Gym masks the row and `gym eval reverify` can
  judge it again.
- Only the overall field is scored. The judge's per-rubric answers (`rubrics_check`, kept in `judge_output`) are
  free text and often contain no yes or no, for example "The response maintains a pleasant tone throughout." or
  "No emojis are used." for a rubric that is met.

## Metrics

`compute_metrics` reports pass@k for `reward`, overall and per `category`. `category_mean/...` is the unweighted
mean of the three category scores, which is the paper's headline number. It also reports `judge_parse_failed_rate`, `empty_answer_rate`, and the share of replies the endpoint left
incomplete because of the output cap (`policy_max_output_tokens_rate`) or a content filter
(`policy_content_filter_rate`). Each response records that reason in `policy_incomplete_reason`.

## Configuration

| Field | Meaning |
|---|---|
| `judge_model_server` | Model server that judges; the default config points at `advanced_if_judge_model`, an `openai_model` pass-through. |
| `judge_chat_completions_create_params` | Base judge request; `messages` is replaced per row. |
| `judge_prompts_fpath` | JSON file with the official judge prompts, written by `benchmarks/advanced_if/prepare.py`. |
| `judge_max_concurrency` | Concurrent judge calls per server worker (default 64). |
| `judge_max_attempts` | Judge calls per row while the reply is unparseable (default 3). |

## Licensing

- Code: Apache-2.0.
- Data and judge prompts: the AdvancedIF dataset (`facebook/AdvancedIF`) and the official judge prompts
  (`facebookresearch/AdvancedIF`) are CC BY-NC 4.0. Neither is committed to this repository; the benchmark's prepare
  step downloads both at pinned revisions.
- `data/example.jsonl` and `tests/judge_prompts.json` are synthetic stand-ins written for Gym (Apache-2.0).
