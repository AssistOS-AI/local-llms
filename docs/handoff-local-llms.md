# local-llms Handoff

This is the compact handoff for a new session working in this repository.

## Repository policy

Read `AGENTS.md` first. Do not create branches with a `codex/` prefix; do not add AI assistant co-author trailers; keep commit metadata human-authored.

## What is here

One Ploinky agent, `local-llm/`: the Ploinky glue and all of the hardware-limits code (cgroup memory and CPU budgets, the MPS GPU share, vLLM MPS qualification and calibration) around the engine that lives in the Ploinky-Worker repository (pworker, `lib/local-llm`). The agent pins the engine at one full commit SHA in `local-llm/package.json` (`git+https`, `local-llm/package-lock.json` and `local-llm/.npmrc` beside it); a cold agent start needs GitHub to fetch it. The engine's own specifications at the pinned SHA are linked from DS000 to DS006. Layout: `src/` holds the glue (`main`, `controlSocket`, `controlHandlers`, `chatResponder`, `testPrompt`, `promptStream`, `drainBudget`), the binding (`agentHost`, `ploinkyMessages`, `engineOptions`) and the limits (`hardwareLimits`, `controller/ploinkyBudget`, `controller/vllmMpsQualification`); there is no controller or runner code in the tree. Its specifications (`local-llm/docs/specs/`, DS000–DS006) are the source of truth and change in the same commit as the code:

| Spec | Read it for |
| --- | --- |
| DS000 | scope, the split between this agent and the pworker engine, runners, the retired shared-image agents and the services that went with them |
| DS001 | manifest, the engine binding (host, wording, the pin, the internal import list), the cold-start dependency on GitHub, processes, tools, the chat endpoint, drain, LM Studio's switch |
| DS002 | the agent's seed catalog, paths and overrides over the engine's model lifecycle (catalog, registry, downloads, the state machine) |
| DS003 | GPU access, the container's hardware limits and the limits plug-in, runner isolation |
| DS004 | on-demand runners from the image's runner lock and the agent's own CI-gated lock |
| DS005 | hardware profiles and what the container's limits change in them, the memory guard, the GPU share and vLLM qualification, the dashboard |
| DS006 | coding style |

## Working rules

- Tests: from the repository root, `npm --prefix local-llm ci` (it fetches `pworker` from GitHub, so it needs the network and `git`), then `node --test 'local-llm/tests/*.test.mjs'`; the tests need no Docker, GPU or network. A bump of the pworker pin re-runs the whole suite, the overview goldens, the Ploinky wording test and `runner-lock-check`.
- The image is built in `container-image-builds` (`images/local-llm`); the manifest pins it by digest (a multi-arch index once published).
- Nothing unpinned is downloaded at run time; runners listen only on loopback with a per-start key and get a minimal environment; no secrets and no proprietary files go into images.

## History

The repository held twelve CPU agents in the shared `assistos/local-llms` image until 2026-09-28; their code, specifications, HTML documentation and this handoff's earlier text remain at commit `03697765`.
