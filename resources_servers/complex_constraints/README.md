# complex_constraints resources server

Grades replies to [ComplexConstraints](../../benchmarks/complex_constraints/README.md) prompts against their expert
rubrics with an LLM judge, following the official harness
([`surge-ai/complex-constraints`](https://github.com/surge-ai/complex-constraints)).

## Verification

`verify()` takes the text of the final assistant message (reasoning and `<think>` blocks removed) and makes one judge
call per criterion in `rubrics`, all in parallel. Each call sends the official template, filled with the reply and one
criterion, as the only user message, with the official structured-output schema as `response_format`. The judge never
sees the prompt.

- A criterion passes when the judge's `score`, rounded half-up to four places, is at least 1.
- `reward` is 1.0 when every criterion passes, else 0.0. `mean_criteria` is the fraction passed and
  `criterion_grades` holds each criterion's score, rationale and number of judge calls, in `rubrics` order.
- A judge reply that is not a JSON object with a numeric `score` and a string `rationale` is requested again, up to
  `judge_max_attempts` calls (5, as in the official scorer). If a criterion is still unscored, `verify()` raises
  `JudgeError`, so the row is masked and goes to the failures sidecar instead of being scored.
- A reply with no visible text scores 0 without judge calls. A reply cut off by the output cap is graded as written;
  `policy_incomplete_reason` records why the policy stopped.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `judge_model_server` | `complex_constraints_judge_model` | Model server for the judge (Gemini 3.5 Flash by default). |
| `judge_chat_completions_create_params` | `messages: []` | Base judge request; the server sets `messages` and `response_format`. |
| `judge_prompt_fpath` | `benchmarks/complex_constraints/data/complex_constraints_judge_prompt.json` | Official template and schema, written by `gym eval prepare --benchmark complex_constraints`. |
| `judge_max_concurrency` | 64 | Concurrent judge calls per server worker. |
| `judge_max_attempts` | 5 | Judge calls per criterion while the reply does not parse. |

## Data

`data/example.jsonl` holds five Gym-authored tasks in the prepared row format (`task_id`, `use_case`,
`instruction_type`, `prompt_style`, `rubrics`, `responses_create_params`). They are not from the benchmark. The
benchmark rows come from `benchmarks/complex_constraints/prepare.py`.

## Tests

`tests/` runs without a live judge: `verifier_fixture.py` answers each criterion's judge call from canned scores, and
`tests/judge_prompt.json` is a Gym-authored stand-in for the official template and schema.

```bash
gym env test complex_constraints
```
