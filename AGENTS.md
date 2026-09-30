# AGENTS.md

## Scope

This repository provides one Ploinky agent, `local-llm/`: it runs open-weight LLMs on the machine's NVIDIA GPU when one is usable, and on the CPU otherwise, with runners chosen per model (llama.cpp, ik_llama.cpp, Ollama, vLLM, TabbyAPI, and LM Studio for internal use only), on x86 hosts with a GPU of their own and on NVIDIA DGX Spark (arm64, unified memory). Its image is built in `container-image-builds/images/local-llm` (amd64 `Dockerfile`, arm64 `Dockerfile.arm64`, published as one multi-arch index). The twelve shared-image agents the repository held before were retired; `local-llm/docs/specs/DS000-vision-and-scope.md` says what went with them.

## Mandatory Reading Order

1. `CLAUDE.md` — Git policy and repository-level instructions.
2. `local-llm/docs/specs/DS006-coding-style.md` — Coding style, source layout, language rules, and test organization.
3. `local-llm/docs/specs/matrix.md` — Full specification index (DS000–DS006).

## Repository Rules

- **DS specifications are the source of truth.** When source code changes, the DS specifications under `local-llm/docs/specs/` must be updated in the same change. There is no separate HTML documentation.
- **Coding style authority** is `local-llm/docs/specs/DS006-coding-style.md`. The agent is Node.js ESM; tests use `node:test`; all JSON uses 2-space indentation.
- **All documentation, specifications, and comments must be written in English.**
- **DS numbering must remain gap-free.** The sequence runs DS000 through DS006.
- **Decisions & Questions** in DS files use numbered question subchapters. Rationale lives inside the affected DS files rather than in a separate decision log.
- **Git policy**: No `codex/` branch prefixes. No AI assistant co-author trailers. Commits appear human-authored.
- **Test command**, from the repository root: `node --test 'local-llm/tests/*.test.mjs'` (or `npm --prefix local-llm test`). A bare directory argument does not work on Node 24. No Docker, GPU or network is required.

## Key Paths

- `local-llm/manifest.json` — the agent's manifest (image pin, GPU declaration, chat endpoint).
- `local-llm/src/` — controller, runner adapters, chat responder.
- `local-llm/catalog/models.json` — seed model catalog (schema `local-llm.catalog/v3`).
- `local-llm/catalog/runners.lock.linux-{amd64,arm64}.json` — the agent's own runner lock per platform (DS004); `.github/workflows/runner-lock-check.yml` installs its entries in CI.
- `local-llm/IDE-plugins/` — the Local LLMs dashboard and Settings.
- `local-llm/tests/` — tests.
- `local-llm/docs/specs/` — specifications.
- `docs/handoff-local-llms.md` — short handoff for a new session.
