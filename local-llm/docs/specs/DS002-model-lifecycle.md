---
id: DS002
title: Model Lifecycle
status: accepted
owner: local-llm
summary: Catalog, registry, on-demand download, the deployment state machine, and restart behaviour.
---

# DS002 Model Lifecycle

## Introduction

This specification defines how a model moves from a catalog entry to a running deployment and back, and what survives a restart. It reuses the catalog-plus-registry split of the retired repository DS008 ([at `03697765`](https://github.com/AssistOS-AI/local-llms/blob/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs/DS008-model-lifecycle.md)) and replaces its download and start rules.

## Core Content

### Catalog and registry

| Source | Where | Editable |
| --- | --- | --- |
| Seed catalog | `catalog/models.json`, schema `local-llm.catalog/v3` (`catalog/schema.json`) | No; ships with the agent |
| User registry | `state.registry` in `/data/state/controller.json` | Yes, through `local_llm_model_add`, `_update`, `_remove` |

Every entry is validated by `validateModel` (`src/controller/catalog.mjs`). Ids are 2–64 lowercase characters. Sources are keyed by weight format, not by runner: `gguf` is a Hugging Face source naming one `.gguf` file in an `owner/name` repository, or a split GGUF (below), `ollama` is an Ollama library tag, and `hf` (safetensors, read by vLLM) and `exl3` (EXL3, read by TabbyAPI) are Hugging Face snapshots (below). Every runner reads exactly one format (its adapter's `weightFormat`), so runners that read the same format share one download and one Delete weights. `recommended` and `validated` are keyed by hardware profile (`dedicated`, `unified`; DS005) and then by runner, because parameters and measurements are per runner and per kind of memory. `profiles` lists where a model is offered (both by default): the overview and Run see only the models of the agent's profile. `unified.envelope` is the unified profile's measured envelope (DS005); a model without one is refused on unified memory. An Ollama source names a library tag. Seed Hugging Face sources must pin `commit`, `size` and `sha256`. User sources may name a branch; `model_add` resolves it to a commit, size and LFS sha256 through the Hugging Face API before the entry is stored. A user entry cannot reuse the id of any seed, whatever the profiles it is offered in, or of any stored entry, and an invalid stored entry is skipped rather than breaking the catalog. Delete weights finds any model's weights, including a model not offered in the current profile.

Earlier schemas are not migrated (catalog v3 dropped the v1 and v2 migration on purpose), and state this controller cannot read is reported, never misread. A registry entry that is not valid v3 stays in the state file, is not offered, is listed with its reason in the overview's `unsupportedModels` and logged once at start, and can be removed with `local_llm_model_remove` (its downloaded weights, if any, stay on disk). Parameters saved for a model that no longer validate (another profile, a removed parameter) give way to the defaults, with one log line; a recorded deployment whose parameters no longer validate shows no context instead of an error. The state file stays at version 1; a state file of any other version is kept aside as `controller.json.unsupported-<time>` instead of being overwritten.

### Reusing a model file already in the workspace

**Trust model.** The agent's store (`/data`, on the host `<workspace>/.data/local-llm`) is written only by the agent and by the operator (the host user); its files and verification records are trusted at that level, and the records are a cache, not authentication. `/shared` is mounted read-write into every agent of the workspace, so everything in it is untrusted input: nothing read from it is used until a private copy of it has been verified. Hashing proves the bytes that were actually read. The stat identity recorded afterwards (device, inode, size, mtime, ctime) only detects accidental change; timestamps do not guarantee immutability, and a writer with operator access can still change a file after the last check, as it can a downloaded file.

**Where the agent looks, in order.** Before a controller download (a GGUF file, each shard of a split GGUF, each file of a Hugging Face snapshot):

1. **Its own store.** A file whose verification record matches the pinned identity is used as it is while its recorded stat is unchanged; a changed one is hashed again, kept if it still matches, otherwise removed and fetched again. A record without a stat is hashed once.
2. **In-place adoption.** A regular file already at the file's canonical store path, of the pinned size and with no valid record, is hashed and adopted where it is. This is the zero-copy path for a retained or separately downloaded file: the operator hard-links it on the host into the canonical path. The canonical paths are:
   - GGUF, and each shard of a split GGUF: `<workspace>/.data/local-llm/models/gguf/<owner>/<repo>/<commit>/<file>` (`<file>` is the file's whole path in the repository);
   - a Hugging Face snapshot file: `<workspace>/.data/local-llm/models/hf/<owner>/<repo>/<commit>/<file>`.

   The operator must not link a file that also stays under `.data/shared`: every agent could then write to it. The deployment log warns when an adopted file has other links.
3. **`/shared/models`** (on the host `<workspace>/.data/shared/models`; `LOCAL_LLM_SHARED_MODELS` overrides it for tests). A candidate is used only through a **private copy** into the store, hashed as it is written; it is never hard-linked (a link would share the inode with a path every agent can write, and inside the container `/shared` and `/data` are separate mounts, where `link()` fails with EXDEV anyway).
4. **The download** from Hugging Face, or its resume.

Only what the agent can see is searched: nothing outside the workspace, and Ollama tags keep Ollama's own store.

**Matching.** Only the pinned identity decides: the exact size and the pinned digest, the sha256 or, for a small file kept in git, the pinned git blob id. This holds for every shard and every snapshot file. Names only order the candidates.

**Binding verification to the bytes that were hashed.** A candidate or store file is opened without following a final symbolic link (`O_NOFOLLOW`) and must be a regular file; `/proc/self/fd` must then show that the opened object is exactly the expected path under its root, which also defeats a symbolic link in, or a swap of, any directory on the way (if that cannot be read, the agent fails closed). The bytes are hashed through that descriptor, with an identical `fstat` before and after; a copy is hashed as it is written, and its source must not change while it is read. The record keeps the identity of the object that was hashed: after the agent's own rename, the final path must hold the same device, inode, size and mtime (only ctime, which the rename changes, may differ). A symbolic link anywhere in the store is treated as absent and logged: at a file's path it is replaced by the fetched file (its target is never touched); in a directory on the way it refuses the Run with `UNSAFE_PATH`.

**Staging and resume.** Each candidate is copied to its own staging file (`<file>.partial.copy-<random>`, next to the partial). An existing `.partial` and its identity record stay untouched until a verified replacement is published; after a mismatch, an error or a Stop only the staging file is removed, and a later download resumes from the earlier offset. Staging files left by a crash are removed at the next attempt.

**Copies follow the download's rules.** Free space for the copy plus the 5 % reserve is checked before copying (`INSUFFICIENT_SPACE`). The copy runs in abortable chunks (8 MiB), so Stop, Cancel and drain settle within one chunk. Progress reports the `copying` phase, and hashing bytes already on disk reports the `verifying` phase. A full disk pauses the deployment (`PAUSED_ENOSPC`); other read and write errors fail it (`COPY_FAILED`); none is ever treated as "no candidate".

**Bounded, cancellable lookup.** One streaming walk of `/shared/models` per artifact (the overview uses one walk for every listed model) collects every size the artifact needs. It follows no directory link, stops at 8 levels and 20,000 entries, and checks for Stop at every entry. At most 3 same-size candidates are hashed per file, those with the file's name first.

**Last check before loading.** Just before the runner starts, after any runner preparation, every file of the artifact is checked again: an unchanged stat, or its bytes are hashed again. A file that no longer matches fails the start (`CHANGED_AFTER_VERIFY`). This detects accidental change; it is not a lock.

**Provenance.** Every file and shard carries one shape, `{ file, source, method, bytes }`: `method` is `in-place`, `copy` or `download`; `source` is the store path, the `/shared` path or the download URL (never a token); `bytes` is what that way moved (0 in place, the size for a copy, the bytes transferred by that download). It is recorded with the file, returned by the fetch, stored as the deployment's `provenance`, and written to the deployment log, one line per file; a file that was already verified is logged as such with its recorded provenance. The Run's plan (`acquisition`: each file's method and the disk it needs) is stored with the deployment when it is accepted.

**Delete weights** removes the agent's own files. `freedBytes` counts the logical size of each distinct inode (device and inode) whose last link the deletion removed; an inode that keeps a link outside the deletion set counts 0.

### Split GGUF

A large model can be published as a split GGUF: `<prefix>-00001-of-0000N.gguf` … `<prefix>-0000N-of-0000N.gguf` in one directory of the repository (Qwen3.5-122B-A10B MXFP4_MOE is three shards). It is still the `gguf` weight format, read by the same runners. The source's `file` is the first shard, which llama.cpp opens (it finds the others next to it), and `shards` lists every shard in canonical order, each pinned by `size` and `sha256`; `size` is their sum and there is no top-level `sha256`. A seed entry must pin every shard; a user entry may name only the first shard, and `local_llm_model_add` resolves the commit from it and every shard at that commit. Naming any other shard, or a split of more than 64 shards, is refused before anything is resolved.

The set is one artifact: one identity (commit, first shard, shard count) for in-use checks; its download state is complete only when every shard is, with bytes counted over the set; the download goes shard by shard into the one directory, each verified by its own sha256, with progress over the whole set; Delete weights removes every shard; and after the verified download every shard is dropped from the page cache.

### Weights on demand

Nothing is downloaded at enable, at restart, when the Local LLMs dashboard is opened, or when a model is added. Weights are fetched only by `local_llm_run`.

Each source type has a weight store (`src/controller/weightStores.mjs`) that pins, inspects, fetches and deletes it and names its artifact identity; runners that read the same file share that identity, so no runner may delete weights another runner is using.

| Format (source type) | Download |
| --- | --- |
| `gguf` (`huggingface`), fetched by the controller before the runner starts | `src/controller/downloader.mjs` fetches `/<repo>/resolve/<commit>/<file>` into `/data/models/gguf/<repo>/<commit>/<file>.partial` (the file's whole relative path, so same-named files in different folders never share a location), with an identity sidecar `<file>.partial.json`. It resumes with `Range`, restarts on a 200 reply, discards the partial on an invalid `Content-Range` or a changed identity, re-resolves expired redirects, refuses to start when free space is below the remaining bytes plus 5 %, pauses on `ENOSPC`, and renames the file into place only after the sha256 matches. |
| `ollama` (`ollama`), fetched by the runner | The Ollama adapter starts `ollama serve` with `OLLAMA_MODELS=/data/models/ollama` and streams `/api/pull`. When the catalog pins `manifestDigest`, the stored manifest must hash to it. |
| `hf` and `exl3` (`hf-snapshot`), fetched by the controller before the runner starts | A snapshot is every top-level model file of a repository at one commit: names ending in `.json`, `.safetensors`, `.txt`, `.model`, `.tiktoken` or `.jinja`. It takes no subdirectories (gpt-oss-20b keeps a second copy of its weights in `original/`), no Python (remote code is never run), and no other weight formats. Each file is pinned by size and either its LFS sha256 or, for a small file kept in git, its git blob oid (sha1 over `blob <size>\0` and the bytes). `size` is the sum of the files. Seed entries pin every file; `model_add` pins a user entry's revision through the Hugging Face tree API. The files download one by one with the same downloader (resume, free-space check, verification before the rename) into `/data/models/hf/<repo>/<commit>/`. The partial files and identity records live in `/data/models/hf/.state/<repo>/<commit>/`, so the directory a runner loads holds only verified files. A snapshot is ready only when every file has verified. Delete weights removes the snapshot and its records together. |

`HF_TOKEN`, when set in the agent profile, is sent as a Bearer header to the Hugging Face base URL and is never logged. Node's `fetch` drops the header when a redirect leaves that origin, so the signed CDN URL never receives it.

### Deployment state machine

```
idle ──run──▶ downloading ──▶ verifying ──▶ starting ──▶ loading ──▶ ready
  ▲              │ cancel / drain                                  │ stop
  │              ▼                                                  ▼
  └──────────  paused  ◀── restart ──                           stopping ──▶ idle
any phase ── failure ──▶ error (the next run or stop clears it; stop also clears paused)
```

One deployment exists at a time. A second `run` while one is active is rejected with `busy` unless `replace: true`. A repeated `run` with the same `requestId` is a no-op, so a client retry cannot start a second job. Before the download and again before the runner starts, admission (DS003) is evaluated against the current hardware snapshot.

The job takes an immutable copy of the model source when it starts. A registry edit during a download cannot change which file is fetched. Deleting the weights of, updating, or removing a model in use is rejected with `in_use`. Ollama tags can share blobs. Deleting one tag's weights never removes a partial blob that another tag's recorded pull claims, and it is rejected with `in_use` while another tag's running pull has reported a blob this deletion would touch. While any other Ollama pull runs, partials that no tag claims are kept, because that pull may not have reported them yet.

### Restart

State is written atomically to `/data/state/controller.json`. After a restart, `downloading`, `verifying` and `pulling` become `paused`, and `starting`, `loading`, `ready` and `stopping` become `idle`, because the runner process did not survive. Nothing resumes automatically; the next explicit Run resumes a paused download with `Range`.

### Reuse from repository DS008

The retired repository specification is kept at commit [`03697765`](https://github.com/AssistOS-AI/local-llms/blob/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs/DS008-model-lifecycle.md); the table records what local-llm took from it.

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
