---
title: DS002-model-lifecycle
summary: Catalog, registry, on-demand download, the deployment state machine and restart behaviour as the agent hosts them: the engine's specification at the pinned pworker commit, and the agent's seed catalog, paths, overrides and tools.
---

# DS002 Model Lifecycle

## Introduction

This specification defines how a model moves from a catalog entry to a running deployment and back, and what survives a restart, for the `local-llm` agent. The behavior itself belongs to the engine, which lives in the Ploinky-Worker repository (pworker) under `lib/local-llm` and is pinned by this repository (DS001): its catalog and registry, its downloads and the reuse of files already in the workspace, split GGUF, the deployment state machine and restart behaviour are specified in [pworker DS002](https://github.com/AssistOS-AI/Ploinky-Worker/blob/4668a3f0e2cae0c85463945def068741dc8c2b68/docs/specs/DS002-model-lifecycle.md) at commit `4668a3f0e2cae0c85463945def068741dc8c2b68`. That text is the one this specification used to hold, carried with the agent's names replaced by the engine's host fields. This specification records what the agent supplies to it: the seed catalog, the container paths and their overrides, and the tools that reach the controller. It reuses the catalog-plus-registry split of the retired repository DS008 ([at `03697765`](https://github.com/AssistOS-AI/local-llms/blob/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs/DS008-model-lifecycle.md)) and replaces its download and start rules.

## Core Content

### What the agent supplies

The engine reads no path from the environment and has no default location for a catalog or a lock (pworker DS007). `src/agentHost.mjs` fills the engine's host with the container's values, and `src/engineOptions.mjs` composes the controller from them (DS001). The engine specification's host fields are these values in this agent:

| Engine term | In this agent |
| --- | --- |
| `dataDir` | `/data`, on the host `<workspace>/.data/local-llm`; `LOCAL_LLM_DATA_DIR` overrides it. State, weights, the runners' cache, logs and the runners' `HOME` live under it. |
| `runRoot` | `/opt/runners`, in the container's own filesystem; `LOCAL_LLM_RUN_ROOT` overrides it (DS004). |
| `sharedModelsRoot` | `/shared/models`, on the host `<workspace>/.data/shared/models`; `LOCAL_LLM_SHARED_MODELS` overrides it for tests. |
| `seedCatalogFile` | `catalog/models.json` in this repository, from `agentSeedCatalogFile()`; `LOCAL_LLM_CATALOG_FILE` replaces it with the operator's catalog. |
| `hfBaseUrl` | `https://huggingface.co`; `LOCAL_LLM_HF_BASE_URL` overrides it for tests. |
| State file | `/data/state/controller.json`, written atomically. |
| `HF_TOKEN` | Set in the agent profile (`ploinky var HF_TOKEN <token>`, then a restart); the engine sends it as a Bearer header to the Hugging Face origin only and never logs it. |
| Controller operations | The tools of DS001: `local_llm_run` is `run`, `local_llm_status` is `status`, `local_llm_overview` is `overview`, `local_llm_stop` is `stop`, `local_llm_download_cancel` is `cancelDownload`, `local_llm_weights_delete` is `deleteWeights`, `local_llm_model_add`, `_update` and `_remove` are `addModel`, `updateModel` and `removeModel`, `local_llm_model_lookup` is `lookupModel`, and `local_llm_runner_install` and `_uninstall` are `installRunner` and `uninstallRunner`. |

The overrides keep the semantics they had before the engine moved (DS001): an empty value takes the default, and a relative value is resolved against the working directory.

### The seed catalog

The seed catalog is the agent's own data: `catalog/models.json`, schema `local-llm.catalog/v4`, validated by the engine's `validateModel` and its schema, which ship in the pinned package (`catalog/schema.json` below `lib/local-llm`; `v3` remains readable under its original rules). It is read-only and ships with the agent. The user registry is `state.registry` in the state file, edited through `local_llm_model_add`, `_update` and `_remove`. `unified.envelope` and the `validated` labels are certifications that come only from a trusted seed: the catalog shipped with the agent, or the operator's `LOCAL_LLM_CATALOG_FILE`, whose provenance is controlled outside the model-management API (DS005). Seed entries offered in the `cpu` profile are small GGUF models that fit a machine with a few GiB of free RAM (Qwen2.5 0.5B and 1.5B, Qwen3 4B Instruct 2507, all Q4_K_M and Apache-2.0), and gpt-oss-20b, which fits only a large host; `recommended.cpu` gives their llama.cpp defaults. The small ones are offered on every profile, so a GPU host runs them with the runner's own defaults for its profile. The seed export below is the operator's way to produce a catalog that older agents can read.

### Files already in the workspace

`/shared` is mounted read-write into every agent of the workspace, so the engine treats `sharedModelsRoot` as untrusted input and uses a candidate only through a verified private copy (pworker DS002). Two paths are specific to this agent's layout. An operator who wants the zero-copy adoption hard-links a retained file on the host into the canonical path of the agent's store: for GGUF, and for each shard of a split GGUF, `<workspace>/.data/local-llm/models/gguf/<owner>/<repo>/<commit>/<file>` (`<file>` is the file's whole path in the repository), and for a Hugging Face snapshot file `<workspace>/.data/local-llm/models/hf/<owner>/<repo>/<commit>/<file>`. The operator must not link a file that also stays under `.data/shared`: every agent could then write to it. The deployment log warns when an adopted file has other links.

### Catalog export for older agents

Catalog v4 separates recommendation and measurement keys from eligibility, and old agents reject v4 seed envelopes, so a downgrade needs their matching shipped v3 seed or a reviewed v3 export. The export helper is the engine's `exportModelMetadata(models, { seed, targetSchema, dropVenusMetadata })` (pworker DS002); in this repository it is reached through the unstable `internal` subpath (DS001). For an offline seed export, run from the `local-llm` directory, where the package is installed (`npm --prefix local-llm ci`):

```sh
cd local-llm
node --input-type=module -e 'import fs from "node:fs"; import "pworker/local-llm"; import { exportModelMetadata } from "pworker/local-llm/internal/controller/venusMetadata.mjs"; const input=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); const {schema,models,dropped}=exportModelMetadata(input.models,{seed:true,targetSchema:"local-llm.catalog/v3",dropVenusMetadata:true}); process.stdout.write(JSON.stringify({schema,models},null,2)+"\n"); process.stderr.write(JSON.stringify({dropped})+"\n");' ../catalog-input.json > ../catalog-v3.json 2> ../catalog-losses.json
```

The bare `import "pworker/local-llm"` comes first on purpose: the engine's catalog and Venus-metadata modules import each other, and loading the Venus-metadata module first fails with `ReferenceError: Cannot access 'CATALOG_V4' before initialization`. The same failure occurs in the command this specification held before the engine moved. For a registry export, pass `seed: false` and an array of entries. Applying the returned models to backed-up state is a separate operator action.

### Reuse from repository DS008

The retired repository specification is kept at commit [`03697765`](https://github.com/AssistOS-AI/local-llms/blob/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs/DS008-model-lifecycle.md); the table records what local-llm took from it.

| DS008 rule | local-llm |
| --- | --- |
| Catalog is read-only; registry is operator-created and persists on the volume | Kept |
| Registration records metadata only | Kept: `model_add` stores metadata and never downloads |
| Registry writes are atomic | Kept (write to a temporary file, then rename) |
| Download is built into start; llama.cpp models must be pre-placed | Replaced: explicit Run downloads, resumes and verifies |
| Switching models needs an agent restart | Replaced: Run with `replace: true` stops the current runner and starts the new one |

### Rationale

#### Explicit download resumption

A restart can be the operator's way of stopping a large download, and an automatic resume would consume bandwidth and disk without a request. The state records `paused` with the byte count, and the next Run continues from there. Weights move onto the disk only after an explicit Run, are verified against a pinned identity, survive restarts as resumable partials, and are removed only by an explicit delete.

#### Catalog data stays with the agent

The engine ships the schema and the validator but no catalog data and no default path to one, so a consumer of the package cannot silently read the engine's files instead of its own. Ploinky's hardware-limits harness reads the agent's catalog at `local-llm/catalog/models.json`, which is why that path does not change.
