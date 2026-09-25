# local-llms

Ploinky repository with local LLM agents. Each agent runs a specific model role inside the shared `assistos/local-llms` Docker image, managed by Ploinky's manifest-driven startup and MCP tool surface.

## Agents

| Agent | Role | Default Model | Backend | RAM Band | Default Enabled |
|---|---|---|---|---|---|
| `local-llms-manager` | Model registration, startup scripts, profile lifecycle | — | — | 256 MB | Yes |
| `language-translation` | Multilingual text translation | facebook/m2m100_418M | transformers_seq2seq | 2-8 GB | No |
| `relevance` | Passage relevance scoring/reranking | Qwen/Qwen3-Reranker-0.6B | reranker | 2-4 GB | No |
| `function-selection` | Function/tool routing | functiongemma | ollama | 1-2 GB | Yes (no-wait) |
| `function-invocation` | Tool calls with simple parameters | qwen3.5:0.8b | ollama | 2-4 GB | Yes (no-wait) |
| `tool-composition-local` | Short tool chains | qwen3.5:2b | ollama | 4-6 GB | No |
| `base-local` | General writing and simple plans | qwen3.5:4b | ollama | 6-8 GB | No |
| `local` | Tool-based workflows | qwen3.5:4b | ollama | 8-12 GB | No |
| `planning-local` | DSL planning, reports, specifications | qwen3.5:9b | ollama | 12-16 GB | No |
| `validated-planning-local` | Plans with validation and controlled repair | qwen3.5:9b | ollama | 16-24 GB | No |
| `adaptive-local` | Replanning after execution and errors | qwen3.5:27b | ollama | 24-32 GB | No |
| `coding-local` | Agentic coding in a local repository | qwen3.6:27b | ollama | 30-48 GB | No |

## Default Startup

When `local-llms-manager` is enabled as the static agent:

1. Manager starts and becomes MCP-ready immediately.
2. `function-selection` and `function-invocation` are launched as `no-wait` dependencies — manager readiness does not block on them.
3. Other agents must be explicitly enabled by the operator.

## Backend Families

| Backend | API Shape | Description |
|---|---|---|
| `ollama` | OpenAI-compatible chat | Ollama model server for GGUF/safetensors models |
| `llama_cpp` | OpenAI-compatible chat | llama.cpp HTTP server for GGUF models |
| `transformers_seq2seq` | Translation API | HuggingFace encoder-decoder translation service |
| `reranker` | Scoring API | Cross-encoder/sentence-transformers scoring service |
| `vllm` | OpenAI-compatible chat | vLLM server (experimental, requires GPU) |
| `sglang` | OpenAI-compatible chat | SGLang server (experimental, requires GPU) |
| `lmstudio_llmster` | OpenAI-compatible chat | LM Studio headless daemon (experimental) |

## Model Download Strategy

- Model weights are NOT baked into the Docker image.
- Default-enabled agents pull their default model on first start (via `start-agent.sh`).
- Additional Ollama models are downloaded through the explicit `start_model` MCP tool.
- `register_model` records catalog/registry metadata; it does not download weights by itself.
- Catalog entries may keep the requested model ID and provide a backend-specific `backendModel` alias when the runnable name differs, such as `google/translategemma-4b-it` -> `translategemma:4b` for Ollama.
- All model storage is shared per backend under `.ploinky/data/local-llms/` to avoid duplicate downloads.

## Storage Layout

```
.ploinky/data/local-llms/
├── registry.json              # Dynamic model/profile registrations
├── scripts/<profile-id>/      # Generated startup scripts
│   └── start.sh
├── ollama/                    # Shared Ollama model root
│   └── models/
├── llama-cpp/models/          # Shared GGUF model cache
├── transformers/cache/        # Shared HuggingFace cache
├── vllm/cache/                # vLLM HuggingFace cache (experimental)
├── sglang/cache/              # SGLang HuggingFace cache (experimental)
├── lmstudio/models/           # LM Studio models (experimental)
└── agents/<agent-id>/         # Per-agent runtime scratch
```

## GPU and Resource Limitations

Ploinky's current `containerSecurity` only supports `privileged: true`. GPU passthrough is not modeled in the manifest contract.

**Operator requirements for GPU:**
- Docker: `--gpus all` or `--gpus '"device=0"'` must be added manually at container runtime.
- Podman with NVIDIA CDI: `--device nvidia.com/gpu=all` and NVIDIA Container Toolkit CDI setup.

Agents with `cpuSafe: false` (adaptive-local, coding-local) will be extremely slow without GPU acceleration. The startup scripts check available memory and refuse obviously unsafe defaults when possible.

## LM Studio / llmster

LM Studio's headless daemon (`llmster`) runs through the GPU agent `local-llm`, not through these legacy agents. There it is an on-demand runner for **internal use only**. It stays off unless the deployment's operator turns it on (`ploinky var LOCAL_LLM_LMSTUDIO internal-use`, then a restart of local-llm). An admin installs it from the pinned download after accepting LM Studio's Terms. It is never part of any image. See `local-llm/docs/specs/DS000-vision-and-scope.md` and `local-llm/docs/specs/DS004-on-demand-runners.md`.

The legacy catalog's `lmstudio_llmster` backend entry (`experimental: true`, `scripts/runners/lmstudio-llmster.sh`) is an older placeholder from before that agent. No legacy agent uses it, and the legacy image does not contain llmster.

## MCP Tool Surfaces

**Manager** (`local-llms-manager`):
- `list_available_models` — Catalog + registry models
- `register_model` — Add model to workspace registry
- `generate_startup_script` — Generate startup script under `.ploinky/data/local-llms/scripts/`
- `register_agent_profile` — Register a runtime profile
- `promote_agent_profile` — Promote profile to first-class agent (dev operation, requires restart)
- `status_model` — Check model/backend status

**Model agents** (function-selection, function-invocation, etc.):
- `list_available_models` — Agent-scoped model list
- `start_model` — Start or switch model
- `stop_model` — Stop model
- `status_model` — Check model/backend status
- `register_model` — Register model scoped to this agent

**Translation agent** also exposes: `translate`
**Relevance agent** also exposes: `rerank`

## How to Register Models and Generate Startup Scripts

```
# Register a new model via MCP tool
register_model(modelName="my-custom-model", backend="ollama", agentId="base-local")

# Generate a startup script for a profile
generate_startup_script(profileId="my-profile", agentId="base-local", modelName="my-custom-model")
# Output: .ploinky/data/local-llms/scripts/my-profile/start.sh

# Register a full agent profile
register_agent_profile(profileId="my-agent", agentId="base-local", models=["model-a", "model-b"], defaultModel="model-a")
```

## Development

```sh
# Validate catalog and manifests (no model download required)
node --test tests/validate.mjs

# Build the shared image
docker build -t assistos/local-llms:latest .
```
