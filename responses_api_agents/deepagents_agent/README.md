# Deep Agents Agent

Runs LangChain Deep Agents inside a NeMo Gym response API agent server.

This integration follows the Hermes agent shape: the agent points at Gym's
configured model server, uses the harness' own tools, converts the harness
trajectory back into Gym response output items, then verifies through the
configured resources server. It intentionally does not bridge Gym dataset tools
into LangChain tools.

## Quick Start

```bash
ng_run "+config_paths=[responses_api_agents/deepagents_agent/configs/deepagents_agent.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
  +agent_name=deepagents_agent \
  +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
  +output_jsonl_fpath=deepagents_agent_rollout.jsonl \
  +limit=1
```

## Configuration

```yaml
deepagents_agent:
  responses_api_agents:
    deepagents_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_verifier
      model_server:
        type: responses_api_models
        name: policy_model
      max_turns: 30
      concurrency: 32
      backend: state
      system_prompt: null
```

| field | default | description |
|-------|---------|-------------|
| `max_turns` | `30` | Used to derive the LangGraph recursion limit when `recursion_limit` is unset. |
| `recursion_limit` | `null` | Direct LangGraph recursion limit override. |
| `concurrency` | `32` | Max simultaneous `/run` calls. |
| `temperature` | `1.0` | Forwarded to Gym's model server unless the request overrides it. |
| `system_prompt` | `null` | Custom Deep Agents system prompt prefix. |
| `backend` | `state` | Deep Agents backend. `state` supports virtual files; `local_shell` also enables `execute`. |
| `backend_root_dir` | `null` | Root directory for `local_shell`. |
| `backend_virtual_mode` | `false` | Passed to `LocalShellBackend` for path handling. |
| `backend_inherit_env` | `false` | If `true`, local shell commands inherit process environment variables. |
| `execute_timeout` | `60` | Local shell command timeout in seconds. |
| `skills` | `null` | Optional Deep Agents skill source paths. |
| `memory` | `null` | Optional Deep Agents memory source paths. |
| `chat_template_kwargs` | thinking enabled, no thinking truncation | Passed through Gym model server metadata for vLLM chat-template control. |

## Notes

Deep Agents' built-in tools are the only tools exposed to the model. Any
`responses_create_params.tools` from the dataset are ignored by the LangChain
harness and are preserved only on the returned Gym response object.

The adapter registers a Gym harness profile with Deep Agents that disables the
summarization middleware, mirroring Hermes' compression-disabled behavior.
