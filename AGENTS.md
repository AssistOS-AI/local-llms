# AGENTS.md

## Scope

This repository provides one Ploinky agent, `local-llm/`: it runs open-weight LLMs on NVIDIA through CUDA, on discrete AMD Radeon through Vulkan RADV, on Apple GPUs through Vulkan Venus in a libkrun Podman machine, and on the CPU otherwise. Runners are chosen per model (llama.cpp, its Vulkan build, ik_llama.cpp, Ollama, vLLM, TabbyAPI, and LM Studio for internal use only). Vulkan profiles offer only llama.cpp's Vulkan build; AMD support is unit-tested and unverified on hardware, and Venus uses a provisional memory cap. Its image is built in `container-image-builds/images/local-llm`, with preserved amd64 and arm64 CUDA parents plus a Vulkan layer, published as one multi-arch index. The twelve shared-image agents the repository held before were retired; `local-llm/docs/specs/DS000-vision-and-scope.md` says what went with them.

## Mandatory Reading Order

1. `local-llm/docs/specs/DS006-coding-style.md` — Coding style, source layout, language rules, and test organization.
2. `local-llm/docs/specs/matrix.md` — Full specification index (DS000–DS006).

## Repository Rules

- **DS specifications are the source of truth.** When source code changes, the DS specifications under `local-llm/docs/specs/` must be updated in the same change. There is no separate HTML documentation.
- **Coding style authority** is `local-llm/docs/specs/DS006-coding-style.md`. The agent is Node.js ESM; tests use `node:test`; all JSON uses 2-space indentation.
- **All documentation, specifications, and comments must be written in English.**
- **DS numbering must remain gap-free.** The sequence runs DS000 through DS006.
- Specifications use exactly `title` and `summary` frontmatter fields, with the filename stem as the title. `Introduction` and `Core Content` are the only top-level content sections; preserve rationale as declarative content within the affected specification. Keep prose paragraphs unwrapped.
- The established DS000–DS006 filenames are a compatibility boundary for source, tests, and external references. Preserve their identifiers and the specs-only documentation layout.
- The specification matrix has `Name` and `Description` columns and links directly to sibling Markdown specifications. This specs-only repository has no HTML specification loader; do not introduce loader URLs into its matrix.
- **Git policy**: No `codex/` branch prefixes. No Codex, Claude, or other AI assistant co-author trailers. Keep commit metadata human-authored unless the user explicitly asks for different attribution.
- **Instruction file**: this `AGENTS.md` is the single instruction file for every coding agent; do not add a `CLAUDE.md`.
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
