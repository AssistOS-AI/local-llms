# local-llms Handoff

This is the compact handoff for a new session working in this repository.

## Repository policy

Read `CLAUDE.md` first, then `AGENTS.md`. Do not create branches with a `codex/` prefix; do not add AI assistant co-author trailers; keep commit metadata human-authored.

## What is here

One Ploinky agent, `local-llm/`. Its specifications (`local-llm/docs/specs/`, DS000–DS006) are the source of truth and change in the same commit as the code:

| Spec | Read it for |
| --- | --- |
| DS000 | scope, runners, the retired shared-image agents and the services that went with them |
| DS001 | manifest, processes, tools, the chat endpoint, drain, LM Studio's switch |
| DS002 | catalog v3, registry, downloads (split GGUF included), the deployment state machine |
| DS003 | GPU access, dedicated-profile admission, runner isolation |
| DS004 | on-demand runners from the image's runner lock |
| DS005 | hardware profiles: dedicated, unified (DGX Spark) and cpu, envelopes, the memory guard |
| DS006 | coding style |

## Working rules

- Tests: `node --test 'local-llm/tests/*.test.mjs'` from the repository root; they need no Docker, GPU or network.
- The image is built in `container-image-builds` (`images/local-llm`); the manifest pins it by digest (a multi-arch index once published).
- Nothing unpinned is downloaded at run time; runners listen only on loopback with a per-start key and get a minimal environment; no secrets and no proprietary files go into images.

## History

The repository held twelve CPU agents in the shared `assistos/local-llms` image until 2026-09-28; their code, specifications, HTML documentation and this handoff's earlier text remain at commit `03697765`.
