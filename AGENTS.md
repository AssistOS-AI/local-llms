# AGENTS.md

## Scope

This repository provides one Ploinky agent, `local-llm/`, which is the Ploinky glue and all of the hardware-limits code (cgroup memory and CPU budgets, the MPS GPU share, vLLM MPS qualification and calibration) around an engine that lives in the Ploinky-Worker repository (pworker, `lib/local-llm`). The agent depends on that engine at one full commit SHA pinned in `local-llm/package.json`; the engine's specifications at that SHA are the source of truth for engine behavior. The agent runs open-weight LLMs on NVIDIA through CUDA, on discrete AMD Radeon through Vulkan RADV, on experimental AMD integrated GPUs behind `LOCAL_LLM_AMD_APU=experimental`, on Apple GPUs through Vulkan Venus in a libkrun Podman machine, and on the CPU otherwise. Runners are chosen per model (llama.cpp, its Vulkan build, ik_llama.cpp, Ollama, vLLM, TabbyAPI, and LM Studio for internal use only). Vulkan profiles offer only llama.cpp's Vulkan build; AMD support is unverified on hardware, APU uses one shared RAM pool and refuses finite or unknown budgets, and Venus uses a provisional memory cap. Its image is built in `container-image-builds/images/local-llm`, with preserved amd64 and arm64 CUDA parents plus a Vulkan layer, published as one multi-arch index. The twelve shared-image agents the repository held before were retired; `local-llm/docs/specs/DS000-vision-and-scope.md` says what went with them.

## Mandatory Reading Order

1. `local-llm/docs/specs/DS006-coding-style.md` — Coding style, source layout, language rules, and test organization.
2. `local-llm/docs/specs/matrix.md` — Full specification index (DS000–DS006).

## Repository Rules

- **DS specifications are the source of truth.** When source code changes, the DS specifications under `local-llm/docs/specs/` must be updated in the same change. There is no separate HTML documentation.
- **Spec ownership.** DS001 owns the manifest, the tools, the chat endpoint, the drain and the binding to the engine (`src/agentHost.mjs`, `src/ploinkyMessages.mjs`, `src/engineOptions.mjs`, the pin and the internal import list). DS002 to DS005 own the agent's data and paths, the GPU grant, the hardware limits (`src/hardwareLimits.mjs` and the budget and qualification modules), the agent's runner lock and its CI gate, the operator switches and the dashboard; they point at the engine's specifications at the pinned commit for engine behavior (model lifecycle, snapshot, admission, runners, profiles) instead of repeating it. Do not describe moved engine internals here, and do not edit the engine from this repository.
- **The engine pin.** Change `pworker` only to a full 40-hex commit, in `local-llm/package.json` and `local-llm/package-lock.json` together, with the lock's `resolved` kept in `git+https` form. A bump re-runs the whole agent suite, the overview goldens, the Ploinky wording test (`ploinky-wording.test.mjs`) and the `runner-lock-check` workflow (DS001).
- **Coding style authority** is `local-llm/docs/specs/DS006-coding-style.md`. The agent is Node.js ESM; tests use `node:test`; all JSON uses 2-space indentation.
- **All documentation, specifications, and comments must be written in English.**
- **DS numbering must remain gap-free.** The sequence runs DS000 through DS006.
- Specifications use exactly `title` and `summary` frontmatter fields, with the filename stem as the title. `Introduction` and `Core Content` are the only top-level content sections; preserve rationale as declarative content within the affected specification. Keep prose paragraphs unwrapped.
- The established DS000–DS006 filenames are a compatibility boundary for source, tests, and external references. Preserve their identifiers and the specs-only documentation layout.
- The specification matrix has `Name` and `Description` columns and links directly to sibling Markdown specifications. This specs-only repository has no HTML specification loader; do not introduce loader URLs into its matrix.
- **Git policy**: No `codex/` branch prefixes. No Codex, Claude, or other AI assistant co-author trailers. Keep commit metadata human-authored unless the user explicitly asks for different attribution.
- **Instruction file**: this `AGENTS.md` is the single instruction file for every coding agent; do not add a `CLAUDE.md`.
- **Test command**, from the repository root: `npm --prefix local-llm ci`, then `node --test 'local-llm/tests/*.test.mjs'` (or `npm --prefix local-llm test` after the install). The install fetches `pworker` from GitHub, so it needs the network and `git`; the tests after it need no Docker, GPU or network. A bare directory argument does not work on Node 24.

## Key Paths

- `local-llm/manifest.json` — the agent's manifest (image pin, GPU declaration, chat endpoint).
- `local-llm/package.json`, `package-lock.json`, `.npmrc` — the pworker pin (a full-SHA `git+https` dependency) and the install settings (`bin-links=false`, `ignore-scripts=true`).
- `local-llm/src/` — the agent's glue: `main.mjs`, `controlSocket.mjs`, `controlHandlers.mjs`, `chatResponder.mjs`, `testPrompt.mjs`, `promptStream.mjs`, `drainBudget.mjs`; the binding to the engine: `agentHost.mjs`, `ploinkyMessages.mjs`, `engineOptions.mjs`; the hardware limits: `hardwareLimits.mjs` and `controller/ploinkyBudget.mjs`, `controller/vllmMpsQualification.mjs`. There is no controller or runner code here: no `src/runners/` and no other `src/controller/` file; the engine is in `local-llm/node_modules/pworker` after the install.
- `local-llm/catalog/models.json` — seed model catalog (schema `local-llm.catalog/v4`; the schema ships in the engine package). The agent passes this file and the lock files to the engine explicitly.
- `local-llm/catalog/runners.lock.linux-{amd64,arm64}.json` — the agent's own runner lock per platform (DS004); `.github/workflows/runner-lock-check.yml` installs its entries in CI.
- `local-llm/IDE-plugins/` — the Local LLMs dashboard and Settings.
- `local-llm/tools/` — the MCP tool entry point, the CI install check (a thin entry over pworker's) and the vLLM MPS calibration.
- `local-llm/tests/` — tests; they need the installed engine package.
- `local-llm/docs/specs/` — specifications.
- `docs/handoff-local-llms.md` — short handoff for a new session.
