---
title: DS006-coding-style
summary: Coding conventions, source layout, language rules and test organization for local-llm, the repository's only agent; carries forward the repository coding-style rules of the retired shared-image agents.
---

# DS006 Coding Style

## Introduction

This specification is the coding-style authority for the local-llms repository, which holds one agent, `local-llm`. It carries forward the repository's coding-style specification from before the twelve shared-image agents were retired (repository `docs/specs/DS001-coding-style.md` at commit [`03697765`](https://github.com/AssistOS-AI/local-llms/blob/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs/DS001-coding-style.md)). The rules that applied only to that code (its shell dispatcher, its Flask services, its single validation file) are recorded there and below; everything else applies to local-llm. `AGENTS.md` points here. The engine that runs the models lives in the Ploinky-Worker repository (pworker), which this repository pins (DS000, DS001); its subtree keeps its own style, in [pworker DS006](https://github.com/AssistOS-AI/Ploinky-Worker/blob/4668a3f0e2cae0c85463945def068741dc8c2b68/docs/specs/DS006-engine-coding-style.md) at the pinned commit `4668a3f0e2cae0c85463945def068741dc8c2b68`.

## Core Content

### Languages and runtimes

- **Node.js 24, ES modules** (`.mjs`) for everything in this repository's agent: the controller process, the control socket, the tools, the chat responder, the binding to the engine, the hardware limits and the IDE plugins' JavaScript. The controller itself and the runner adapters are the engine's. The agent's source is mounted at `/code` and runs unchanged on amd64 and arm64 (DS005); nothing branches on the CPU architecture, except that the agent's runner lock for the running platform is chosen as data (DS004).
- **No Python in the agent.** Runners that are Python programs (vLLM, TabbyAPI) are installed on demand from the image's runner lock (DS004) and never edited here.
- **Shell** only where a script is unavoidable. Any shell script is POSIX `sh` with `set -e`, quotes every variable reference, writes errors to stderr with a `[component]` prefix, and produces JSON only through `jq`, never by string concatenation. The image's build steps live in `container-image-builds`, not here.

### Source layout

```
local-llms/
  AGENTS.md / README.md / dependencies.md
  fileSizesCheck.sh
  docs/handoff-local-llms.md
  .github/workflows/runner-lock-check.yml       the CI install check of the agent's runner lock (DS004)
  local-llm/
    manifest.json, mcp-config.json
    package.json, package-lock.json, .npmrc       the pworker engine pin and its install settings (DS001)
    catalog/models.json                           seed catalog (DS002)
    catalog/runners.lock.linux-{amd64,arm64}.json the agent's own runner lock per platform (DS004)
    src/main.mjs                                  controller process and AgentServer (DS001)
    src/controlHandlers.mjs, controlSocket.mjs    the control socket and its operations on the controller (DS001)
    src/chatResponder.mjs                         the chat endpoint (DS001)
    src/testPrompt.mjs, promptStream.mjs          the admin test prompt and its stream reader (DS001)
    src/drainBudget.mjs                           the agent's stages of the drain deadline (DS001)
    src/agentHost.mjs                             the container's paths and environment overrides, the seed catalog and lock files (DS001)
    src/ploinkyMessages.mjs                       the engine's messages in Ploinky wording (DS001)
    src/engineOptions.mjs                         createAgentController and the engine's tables bound to this agent (DS001)
    src/hardwareLimits.mjs                        the limits plug-in and the engine functions bound to it (DS003)
    src/controller/ploinkyBudget.mjs              the container's cgroup and MPS budgets (DS003)
    src/controller/vllmMpsQualification.mjs       the vLLM MPS qualification and its tuple (DS003)
    tools/                                        MCP tool entry point, the CI install check, the vLLM MPS calibration
    IDE-plugins/                                  the Local LLMs dashboard and Settings
    tests/*.test.mjs
    docs/specs/                                   DS000-DS006 and matrix.md
```

### Conventions

- **Validate at the boundary and fail closed.** Tool input has strict schemas (`additionalProperties: false`); runner parameters are validated against the runner's schema for the hardware profile; identifiers use restrictive patterns (ids `[a-z0-9][a-z0-9._-]{1,63}`, request ids `[A-Za-z0-9_-]{8,128}`), which keeps path traversal and injection out of argv, file paths and registry entries. Values that cannot be read (a GPU's memory, `/proc/meminfo`) mean "unknown" and refuse, never a default.
- **No shell between input and a process.** Runners and helpers are spawned with an argv array, never through a shell, and get a minimal environment (DS003): never the agent's secrets or tokens.
- **Errors carry a code.** Controller errors are the engine's `LocalLlmError` (`pworker/local-llm`) with a stable `code` (for example `admission_incompatible`, `runner_unavailable`) and a message that says what to do.
- **Nothing unpinned at run time.** Downloads are pinned by commit, size and sha256 (models, DS002) or by URL, size and sha256 (runners, DS004).
- **No per-runner branches** in the controller, the tools or the dashboard: a runner is an adapter plus data (DS000).
- **The engine is reached by its exports.** Code and tests import `pworker/local-llm`, `pworker/local-llm/chat`, `pworker/local-llm/runner-process` and `pworker/local-llm/runner-install-check` wherever the stable index exports the name. Only what it does not export is imported from the unstable `pworker/local-llm/internal/*` subpath, which maps to `lib/local-llm/*` and carries no compatibility promise; the list is in DS001, and a pin bump re-checks it.
- **No path of the engine's own files.** Nothing calls the engine's seed-catalog or lock loaders without naming the file, and nothing imports an agent-lock helper from the engine (`no-hidden-defaults.test.mjs`, DS001).
- **English** for all documentation, specifications and comments. Comments say why, not what.

### JSON

Manifests, MCP configuration, the catalog and locks are JSON, never YAML, formatted with 2-space indentation. Catalog entries conform to the engine's catalog schema (`catalog/schema.json` below `lib/local-llm` in the pinned package) and to its `validateModel`.

### Tests

Tests use `node:test` and `node:assert/strict`, with no external framework, one file per area under `local-llm/tests/`. They run against the installed engine package, so the dependency is installed first, from the repository root: `npm --prefix local-llm ci`, then `node --test 'local-llm/tests/*.test.mjs'` (a bare directory argument does not work on Node 24). The install needs the network and `git`, because `pworker` is a git dependency (DS001); the tests after it need no Docker, GPU or network: hardware, downloads, runners and time are injected. The committed `.npmrc` keeps the install free of scripts and of symbolic links. Characterization tests pin today's exact behaviour and change only with a deliberate change, which the commit names. The tests that bind the agent to the engine are `agent-host`, `ploinky-wording`, `production-composition`, `ploinky-budget-state`, `no-hidden-defaults`, `overview-serialized` and `runner-install-check-entry`, and the goldens `overview-golden-a84617d.json` and `overview-cpu-golden-b2b387f.json` are compared through `gpu-profiles-unchanged`, `cpu-host-unchanged` and `overview-serialized`. A few tests need Linux (abstract Unix sockets, `/proc`).

### File size

`fileSizesCheck.sh` reports files over 500 lines (warning) and 800 lines (large). Prefer splitting a module that grows past that by concern. Among the agent's `.mjs` sources only the evidence tool `tools/vllm_mps_calibration.mjs` is over 800 lines (the dashboard component and several tests are too); the controller's single command queue, which was the known exception, moved to the engine.

### Git policy

Branches must not use the `codex/` prefix. Commits must not include AI assistant co-author trailers or tool attribution; commit metadata appears human-authored.

### Rules of the retired code

For the record, the retired shared-image agents' rules were: runtime logic in POSIX shell (`start-agent.sh`, `dispatcher.sh`, runner scripts), Python services as single-file Flask applications, every test in `tests/validate.mjs`, shell scripts under 500 lines and Python services under 150. They no longer apply because that code is gone (DS000).

### Rationale

#### Coding-style authority

The repository's own specification set described only the retired agents and was removed with them. `AGENTS.md` makes the coding-style authority part of the mandatory reading order, so it moves here, numbered after local-llm's specifications, instead of disappearing.

local-llm is Node.js ESM with strict validation at every boundary, JSON configuration, `node:test` tests that need no hardware, and human-authored commits.
