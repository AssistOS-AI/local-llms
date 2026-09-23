---
id: DS002
title: Model Lifecycle
status: accepted
owner: local-llm
summary: Catalog, registry, on-demand download, the deployment state machine, and restart behaviour.
---

# DS002 Model Lifecycle

## Introduction

This specification defines how a model moves from a catalog entry to a running deployment and back, and what survives a restart. It reuses the catalog-plus-registry split of repository DS008 and replaces its download and start rules.

## Core Content

### Catalog and registry

| Source | Where | Editable |
| --- | --- | --- |
| Seed catalog | `catalog/models.json`, schema `local-llm.catalog/v1` (`catalog/schema.json`) | No; ships with the agent |
| User registry | `state.registry` in `/data/state/controller.json` | Yes, through `local_llm_model_add`, `_update`, `_remove` |

Every entry is validated by `validateModel` (`src/controller/catalog.mjs`). Ids are 2–64 lowercase characters. A Hugging Face source names one `.gguf` file in an `owner/name` repository. An Ollama source names a library tag. Seed Hugging Face sources must pin `commit`, `size` and `sha256`. User sources may name a branch; `model_add` resolves it to a commit, size and LFS sha256 through the Hugging Face API before the entry is stored. A user entry cannot reuse a seed id, and an invalid stored entry is skipped rather than breaking the catalog.

### Weights on demand

Nothing is downloaded at enable, at restart, when Settings is opened, or when a model is added. Weights are fetched only by `local_llm_run`.

| Runner | Download |
| --- | --- |
| llama.cpp | `src/controller/downloader.mjs` fetches `/<repo>/resolve/<commit>/<file>` into `/data/models/gguf/<repo>/<commit>/<file>.partial`, with an identity sidecar `<file>.partial.json`. It resumes with `Range`, restarts on a 200 reply, discards the partial on an invalid `Content-Range` or a changed identity, re-resolves expired redirects, refuses to start when free space is below the remaining bytes plus 5 %, pauses on `ENOSPC`, and renames the file into place only after the sha256 matches. |
| Ollama | The controller starts `ollama serve` with `OLLAMA_MODELS=/data/models/ollama` and streams `/api/pull`. When the catalog pins `manifestDigest`, the stored manifest must hash to it. |

`HF_TOKEN`, when set in the agent profile, is sent as a Bearer header to the Hugging Face base URL and is never logged. Node's `fetch` drops the header when a redirect leaves that origin, so the signed CDN URL never receives it.

### Deployment state machine

```
idle ──run──▶ downloading ──▶ verifying ──▶ starting ──▶ loading ──▶ ready
  ▲              │ cancel / drain                                  │ stop
  │              ▼                                                  ▼
  └──────────  paused  ◀── restart ──                           stopping ──▶ idle
any phase ── failure ──▶ error (the next run or stop clears it)
```

One deployment exists at a time. A second `run` while one is active is rejected with `busy` unless `replace: true`. A repeated `run` with the same `requestId` is a no-op, so a client retry cannot start a second job. Before the download and again before the runner starts, admission (DS003) is evaluated against the current hardware snapshot.

The job takes an immutable copy of the model source when it starts. A registry edit during a download cannot change which file is fetched. Deleting the weights of, updating, or removing a model in use is rejected with `in_use`.

### Restart

State is written atomically to `/data/state/controller.json`. After a restart, `downloading`, `verifying` and `pulling` become `paused`, and `starting`, `loading`, `ready` and `stopping` become `idle`, because the runner process did not survive. Nothing resumes automatically; the next explicit Run resumes a paused download with `Range`.

### Reuse from repository DS008

| DS008 rule | local-llm |
| --- | --- |
| Catalog is read-only; registry is operator-created and persists on the volume | Kept |
| Registration records metadata only | Kept: `model_add` stores metadata and never downloads |
| Registry writes are atomic | Kept (write to a temporary file, then rename) |
| Download is built into start; llama.cpp models must be pre-placed | Replaced: explicit Run downloads, resumes and verifies |
| Switching models needs an agent restart | Replaced: Run with `replace: true` stops the current runner and starts the new one |

## Decisions & Questions

### Question #1: Why is a paused download not resumed automatically after a restart?

Response: A restart can be the operator's way of stopping a large download, and an automatic resume would consume bandwidth and disk without a request. The state records `paused` with the byte count, and the next Run continues from there.

## Conclusion

Weights move onto the disk only after an explicit Run, are verified against a pinned identity, survive restarts as resumable partials, and are removed only by an explicit delete.
