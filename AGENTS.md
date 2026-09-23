# AGENTS.md

## Scope

This repository provides twelve Ploinky agents that run local large language models as MCP servers inside a shared Docker image (`assistos/local-llms`). One manager agent handles model registration and profile lifecycle; eleven model-serving agents cover translation, relevance scoring, function routing, general chat, planning, and coding.

It also holds `local-llm/`, a separate GPU-backed agent with its own image (`container-image-builds/images/local-llm`), its own Node.js source and tests, and its own specifications (`local-llm/docs/specs/matrix.md`, numbered independently of `docs/specs/`). The rules below about the shared image, shell-only runtime logic and the DS000–DS011 sequence apply to the twelve shared-image agents; `local-llm/` is tested with `node --test local-llm/tests/`.

## Mandatory Reading Order

1. `CLAUDE.md` — Git policy and repository-level instructions.
2. `docs/specs/DS001-coding-style.md` — Coding style, source layout, language rules, and test organization.
3. `docs/specs/matrix.md` (via `docs/specsLoader.html?spec=matrix.md`) — Full specification index.
4. `docs/index.html` — HTML documentation with architecture, agent table, and runtime details.

## Repository Rules

- **DS specifications are the source of truth.** When source code changes, both the HTML documentation and the DS specifications must be updated to reflect the change.
- **Coding style authority** is `docs/specs/DS001-coding-style.md`. Shell scripts use POSIX `sh` with `set -e`. Python services are single-file Flask apps. Tests use `node:test`. All JSON uses 2-space indentation.
- **All documentation, specifications, and comments must be written in English.**
- **DS numbering must remain gap-free.** The current sequence runs DS000 through DS011.
- **Decisions & Questions** in DS files use numbered question subchapters. Rationale lives inside the affected DS files rather than in a separate decision log.
- **Git policy**: No `codex/` branch prefixes. No AI assistant co-author trailers. Commits appear human-authored.
- **Test command**: `node --test tests/validate.mjs` (123 tests, no Docker or network required).
- **Additional checks**: `sh -n scripts/*.sh scripts/runners/*.sh` and `python3 -m py_compile scripts/services/*.py`.

## Runtime Defaults

- **Shared image**: `assistos/local-llms:latest`
- **Start command**: `/opt/local-llms/scripts/start-agent.sh`
- **Volume mount**: `.ploinky/data/local-llms` → `/data/local-llms`
- **Default startup**: Manager enables `function-selection` and `function-invocation` with `no-wait`.

## Key Paths

- `Dockerfile` — Shared Docker image build.
- `catalog/backends.json` — Backend definitions (ports, probes, storage).
- `catalog/agents/*.json` — Per-agent model catalog with schemas under `catalog/schemas/`.
- `scripts/start-agent.sh` — Startup orchestration (resource check → runner → readiness → AgentServer).
- `scripts/dispatcher.sh` — MCP tool dispatcher (envelope unwrap, validation, routing, registry).
- `scripts/runners/*.sh` — Per-backend runner scripts.
- `scripts/services/*.py` — Python Flask services (translation, reranker).
- `tests/validate.mjs` — Structural and behavioral test suite.
- `docs/index.html` — HTML documentation entry point.
- `docs/specs/matrix.md` — Specification matrix.
- `docs/specsLoader.html` — Specs viewer.
