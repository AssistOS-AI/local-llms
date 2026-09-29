---
id: DS006
title: Coding Style
status: accepted
owner: local-llm
summary: Coding conventions, source layout, language rules and test organization for local-llm, the repository's only agent; carries forward the repository coding-style rules of the retired shared-image agents.
---

# DS006 Coding Style

## Introduction

This specification is the coding-style authority for the local-llms repository, which holds one agent, `local-llm`. It carries forward the repository's coding-style specification from before the twelve shared-image agents were retired (repository `docs/specs/DS001-coding-style.md` at commit [`03697765`](https://github.com/AssistOS-AI/local-llms/blob/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs/DS001-coding-style.md)). The rules that applied only to that code (its shell dispatcher, its Flask services, its single validation file) are recorded there and below; everything else applies to local-llm. `AGENTS.md` points here.

## Core Content

### Languages and runtimes

- **Node.js 24, ES modules** (`.mjs`) for everything in the agent: the controller, the tools, the chat responder, the runner adapters and the IDE plugins' JavaScript. The agent's source is mounted at `/code` and runs unchanged on amd64 and arm64 (DS005); nothing branches on the CPU architecture.
- **No Python in the agent.** Runners that are Python programs (vLLM, TabbyAPI) are installed on demand from the image's runner lock (DS004) and never edited here.
- **Shell** only where a script is unavoidable. Any shell script is POSIX `sh` with `set -e`, quotes every variable reference, writes errors to stderr with a `[component]` prefix, and produces JSON only through `jq`, never by string concatenation. The image's build steps live in `container-image-builds`, not here.

### Source layout

```
local-llms/
  CLAUDE.md / AGENTS.md / README.md
  fileSizesCheck.sh
  docs/handoff-local-llms.md
  local-llm/
    manifest.json, mcp-config.json, package.json
    catalog/models.json, catalog/schema.json      seed catalog (DS002)
    src/main.mjs                                  controller process and AgentServer (DS001)
    src/controlHandlers.mjs                       the control socket's operations on the controller (DS001)
    src/controller/                               state, admission, profiles, downloads, runner processes
    src/runners/                                  one adapter per runner
    tools/                                        MCP tool entry points and the CI install check
    IDE-plugins/                                  the Local LLMs dashboard and Settings
    tests/*.test.mjs
    docs/specs/                                   DS000-DS006 and matrix.md
```

### Conventions

- **Validate at the boundary and fail closed.** Tool input has strict schemas (`additionalProperties: false`); runner parameters are validated against the runner's schema for the hardware profile; identifiers use restrictive patterns (ids `[a-z0-9][a-z0-9._-]{1,63}`, request ids `[A-Za-z0-9_-]{8,128}`), which keeps path traversal and injection out of argv, file paths and registry entries. Values that cannot be read (a GPU's memory, `/proc/meminfo`) mean "unknown" and refuse, never a default.
- **No shell between input and a process.** Runners and helpers are spawned with an argv array, never through a shell, and get a minimal environment (DS003): never the agent's secrets or tokens.
- **Errors carry a code.** Controller errors are `LocalLlmError` with a stable `code` (for example `admission_incompatible`, `runner_unavailable`) and a message that says what to do.
- **Nothing unpinned at run time.** Downloads are pinned by commit, size and sha256 (models, DS002) or by URL, size and sha256 (runners, DS004).
- **No per-runner branches** in the controller, the tools or the dashboard: a runner is an adapter plus data (DS000).
- **English** for all documentation, specifications and comments. Comments say why, not what.

### JSON

Manifests, MCP configuration, the catalog and locks are JSON, never YAML, formatted with 2-space indentation. Catalog entries conform to `catalog/schema.json` and to `validateModel`.

### Tests

Tests use `node:test` and `node:assert/strict`, with no external framework, one file per area under `local-llm/tests/`. They run from the repository root with `node --test 'local-llm/tests/*.test.mjs'` (a bare directory argument does not work on Node 24) and need no Docker, GPU or network: hardware, downloads, runners and time are injected. Characterization tests pin today's exact behaviour and change only with a deliberate change, which the commit names.

### File size

`fileSizesCheck.sh` reports files over 500 lines (warning) and 800 lines (large). Prefer splitting a module that grows past that by concern; `src/controller/deployments.mjs` is the known exception, the controller's single command queue.

### Git policy

Branches must not use the `codex/` prefix. Commits must not include AI assistant co-author trailers or tool attribution; commit metadata appears human-authored.

### Rules of the retired code

For the record, the retired shared-image agents' rules were: runtime logic in POSIX shell (`start-agent.sh`, `dispatcher.sh`, runner scripts), Python services as single-file Flask applications, every test in `tests/validate.mjs`, shell scripts under 500 lines and Python services under 150. They no longer apply because that code is gone (DS000).

## Decisions & Questions

### Question #1: Why keep a coding-style specification inside local-llm's set?

Response: The repository's own specification set described only the retired agents and was removed with them. `AGENTS.md` makes the coding-style authority part of the mandatory reading order, so it moves here, numbered after local-llm's specifications, instead of disappearing.

## Conclusion

local-llm is Node.js ESM with strict validation at every boundary, JSON configuration, `node:test` tests that need no hardware, and human-authored commits.
