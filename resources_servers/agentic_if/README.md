# Agentic IF

Measures whether a coding agent follows instructions while it completes a real repository (Scale-SWE) or terminal
(Terminal-Bench 2.1) task in OpenCode. The environment server in `environment_servers/agentic_if` runs each episode;
this resources server validates each sealed row and grades the recorded session against the row's checks.

## Settings

- `policy_base_url`, `policy_api_key` and `policy_model_name`, in `env.yaml` or as overrides.
- `judge_api_key` for the judge model server, and `judge_base_url` to use another endpoint; the same way.
- `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`.
- `AGENTIC_IF_OPENCODE_BINARY`, optional: the `package/bin/opencode` file of
  https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.17.11.tgz. It must match `binary_sha256` in
  `configs/agentic_if.yaml`. Unset, each task sandbox downloads and checks it, which needs access to
  registry.npmjs.org; as with Gym's OpenCode agent, missing curl, tar or gzip are installed first.

## Run

The `instruction-following` split (type `validation`) is private: its rows ship separately under a plugin root, which
preparation reads through `--search-dir`. Preparation also checks out the terminal task folders the rows use from the
internal `nexus-team/terminal-bench-tasks` repository, so it needs clone access. Where only SSH works, add
`+prepare_script_args.tasks_repository=ssh://git@gitlab-master.nvidia.com:12051/nexus-team/terminal-bench-tasks.git`.

```bash
gym eval prepare --search-dir <plugin root> --config environments/agentic_if/config.yaml
gym eval run --environment agentic_if --split validation \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
```

The `example` split, `data/example.jsonl`, uses public tasks. Its Terminal-Bench rows need the task folders that
`benchmarks/terminal_bench_2_1/prepare.py` clones, so run that script first.

## Metrics

- `reward`: the task's own verifier result. Instruction following does not change it.
- `mean/if_all`: the share of graded rollouts that pass every applicable check.
- `mean/if_frac`: the share of applicable checks passed, averaged over graded rollouts.
- `mean/task_and_if`: the share of graded rollouts that pass both the task and every check.
- `if/graded`, `if/not_applicable` and `if/error` count rollouts; only graded rollouts enter the rates.
