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

Every entry is validated by `validateModel` (`src/controller/catalog.mjs`). Ids are 2–64 lowercase characters. Sources are keyed by weight format, not by runner: `gguf` is a Hugging Face source naming one `.gguf` file in an `owner/name` repository, or a split GGUF (below), `ollama` is an Ollama library tag, and `hf` (safetensors, read by vLLM) and `exl3` (EXL3, read by TabbyAPI) are Hugging Face snapshots (below). Every runner reads exactly one format (its adapter's `weightFormat`), so runners that read the same format share one download and one Delete weights. `recommended` and `validated` are keyed by hardware profile (`dedicated`, `unified`, `cpu`; DS005) and then by runner, because parameters and measurements are per runner and per kind of memory. `profiles` lists where a model is offered (every profile by default): the overview and Run see only the models of the agent's profile. `unified.envelope` is the unified profile's measured envelope (DS005); a model without one, bundled or added at run time, is sized by a labelled estimate there, not refused. The estimate reads `contextLength`, `memory.kvBytesPerToken` and `memory.layers` when the entry has them (the Add model form offers them, optionally) and names the ones it had to default. `mtp: true` declares that the weights carry a multi-token-prediction head, which MTP needs (DS005); it is a capability, not a measurement, so user entries may declare it, and a `recommended` MTP default requires it. A nonempty `unified.envelope` and nonempty `validated` labels are certifications: they come only from trusted inputs, the seed catalog or the operator's catalog, whose provenance is controlled outside the model-management API. `local_llm_model_add` and `_update` cannot self-certify an entry with them. The empty canonical forms, `unified: null` and `validated: {}`, certify nothing. Every trusted measured rectangle names its `loadMode` explicitly, and a mode it does not name is not covered. This specification approves no envelope values; the rules and the measurements are in [DS005](DS005-hardware-profiles.md). An Ollama source names a library tag. Seed Hugging Face sources must pin `commit`, `size` and `sha256`. User sources may name a branch; `model_add` resolves it to a commit, size and LFS sha256 through the Hugging Face API before the entry is stored. A user entry cannot reuse the id of any seed, whatever the profiles it is offered in, or of any stored entry, and an invalid stored entry is skipped rather than breaking the catalog. Delete weights finds any model's weights, including a model not offered in the current profile. Seed entries offered in the `cpu` profile are small GGUF models that fit a machine with a few GiB of free RAM (Qwen2.5 0.5B and 1.5B, Qwen3 4B Instruct 2507, all Q4_K_M and Apache-2.0), and gpt-oss-20b, which fits only a large host; `recommended.cpu` gives their llama.cpp defaults. The small ones are offered on every profile, so a GPU host runs them with the runner's own defaults for its profile. A stored user entry whose id equals a seed's is listed in `unsupportedModels` instead of being hidden silently: the seed takes the id, so the entry is not offered.

Earlier schemas are not migrated (catalog v3 dropped the v1 and v2 migration on purpose), and state this controller cannot read is reported, never misread. One change within v3 is applied once when the `cpu` profile arrives. A stored user entry that lists both GPU profiles (the old default, "every profile") also gets `cpu`. The state records `cpuProfileMigration`, so the change is not repeated. An entry that lists a single profile keeps it. Each changed entry is logged. A registry entry that is not valid v3 stays in the state file, is not offered, is listed with its reason in the overview's `unsupportedModels` and logged once at start, and can be removed with `local_llm_model_remove` (its downloaded weights, if any, stay on disk). Parameters saved for a model that no longer validate (another profile, a removed parameter) give way to the defaults, with one log line; a recorded deployment whose parameters no longer validate shows no context instead of an error. The state file stays at version 1; a state file of any other version is kept aside as `controller.json.unsupported-<time>` instead of being overwritten.

### Looking up a model before adding it

`local_llm_model_lookup` reads Hugging Face metadata for a repository at a revision:

- the commit;
- whether the repository is gated, and its licence;
- every file of the requested format, with its size and digest.

For GGUF, split sets are grouped under their first shard. For `hf` and `exl3`, the snapshot's top-level files are listed, and sizing comes from the snapshot's `config.json` at the commit the lookup resolved, which is checked against the digest the repository listing gives for that file before it is read. Given one GGUF file, the lookup also reads that file's header with a ranged request. The request starts at the Hugging Face origin and follows redirects, and it stops once the header is parsed (at most 32 MiB). From the header it derives `contextLength`, `memory.layers`, `memory.kvBytesPerToken` (f16) and `architecture`.

The architecture name is the prefix of the header's own keys (`qwen2.block_count`, `gpt-oss.attention.head_count_kv`), so it may use lowercase letters, digits, underscore and hyphen (llama.cpp names include `gpt-oss`, `command-r` and `falcon-h1`), up to 40 characters, and no dot. A header that is well-formed but names no architecture, or one that cannot be used as a key prefix, is not an error: the lookup returns null sizing (`contextLength`, `memory.layers` and `memory.kvBytesPerToken` null, `architecture` `dense`) with a note, as it does for a hybrid, recurrent or latent-attention model whose KV size it cannot compute, and the estimate uses its defaults and says so. `invalid_gguf` is for a header that is malformed as GGUF.

It downloads no weights (tensor data) and writes nothing under /data. `HF_TOKEN` is sent only to the Hugging Face origin: every redirect is followed by the agent, which drops the token from a hop on another origin (the file host's CDN), and a next tree page named by a `Link` header on another origin is not followed. The Add model form uses the lookup so an admin picks a file instead of typing its name. The entry is then added with `local_llm_model_add` as before, recording `sizingSource`. While the form still names the repository, revision and file the lookup read, the entry's revision is the 40-hex commit the lookup resolved, so Add pins exactly the files whose sizing was shown even if the branch has moved since; a form edited to another repository, revision or file pins the revision as typed.

Values read from a header are unverified until the download is. After the controller's verified download, the header is read again from the verified file. The admissions that follow use those values, and any difference is logged. That applies to a GGUF entry whose `sizingSource` is `gguf-header`; sizing typed by hand (`manual`, or no `sizingSource`), a snapshot's `config.json` sizing and an Ollama tag are not read again, and neither is a catalog entry. A downloaded file whose header is malformed fails the Run with `invalid_gguf`, and one that cannot be read at all (an I/O error) with `sizing_unreadable`. A well-formed header whose architecture cannot be used (above) checks nothing: the stored, or defaulted, values stand, the log says so (and so does the admission warning on the `cpu` and `unified` profiles, which carry sizing warnings; the `dedicated` admission has none), and the Run goes on.

When the verified file changes the model's training context, the Run's parameters are worked out again from what the request asked for (and what an earlier Run had saved), not from the result already stored: a default `ctxSize` (the catalog's or the schema's) is capped at the verified context, so it is lowered when it was above it and no longer held down by a smaller stored context, and an explicit `ctxSize` above it fails with `invalid_params` after the download and before any runner starts. The estimate, the recorded parameters and the launch all carry that one value. Installing a model from the UI means adding it, then pressing Run, which downloads it with progress and a Cancel.

The lookup reads at most 50 tree pages, one level of subdirectories (32 of them) and 500 files, and says `truncated` when it stopped early. The header reader (`src/controller/ggufHeader.mjs`) treats the bytes as untrusted: version 2 or 3 only, at most 4,096 key-value pairs, 256-byte keys, 4 MiB strings, 4,000,000-entry arrays, nesting depth 2 and 32 MiB in all, with nothing allocated for a claimed length; numeric arrays of up to 4,096 entries are kept, at most 65,536 numbers across all of them, and longer or later arrays are skipped with their length; any violation, or a header that ends early, is `invalid_gguf`. A failed lookup is `invalid_request` (refused before any request), `not_found`, `access_denied` (401 or 403), `lookup_failed` or `invalid_gguf`. Hugging Face answers 401, not 404, for a repository that does not exist when no token is sent, so without a token the message says the repository was not found or needs access, and with one it says access was refused; both name `ploinky var HF_TOKEN <token>` and a restart. A model-info, revision or tree answer is taken from the Hugging Face origin only: a redirect of one of those requests to another origin is refused (`lookup_failed`), as a `Link` page on another origin is not followed; only a file read may be sent on to the file host's CDN. Each of those answers is read up to 8 MiB, and a larger one is `lookup_failed`. At most two lookups run at once, and a third is `busy`. A drain ends a lookup in flight with `shutting_down`.

A user Ollama tag is pinned when it is added. The agent reads the tag's manifest from `registry.ollama.ai` and records `manifestDigest` (the sha256 of the manifest bytes) and `size` (config plus layers, the Ollama store's own formula). The pull is then verified like a seed's, and the tag can be sized on the CPU. Only the manifest is fetched (at most 1 MiB, no credential, a redirect only within the registry's own origin); an entry that already has both values keeps them, one that has one of them gets the other from the registry, which must agree, and a tag that moved since it was pinned is refused with `identity_changed` instead of being pinned again. A registry that cannot be reached, answers anything but a usable version 2 manifest, or has no such tag refuses the Add, so a tag is never stored unpinned by accident. An Update that names a bare tag pins it afresh, which is how an entry stored before tags were pinned, or a tag that moved, is accepted. The registry's exact headers, and whether Ollama stores the manifest bytes it was sent (so that the stored file hashes to the pinned digest), are assumptions that a live pull confirms.

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

**Binding verification to the bytes that were hashed.**
- **Anchored roots.** Every lookup root is anchored once per operation: its real path and directory identity (device and inode), never resolved again. The untrusted `/shared/models` root must itself be a directory, not a symbolic link, and must keep that identity at every candidate open: the root is opened as a directory (`O_DIRECTORY | O_NOFOLLOW`), its device and inode are checked on that descriptor, and the candidate is then looked up from that descriptor one directory at a time (through `/proc/self/fd/<root>/…`, as `openat` would), never through a link. A root substituted after its identity check therefore cannot supply the file; a root that moves while a candidate is opened makes that candidate unusable. Matching bytes do not waive this. The agent's own store root is trusted and may be a link the operator configured.
- **Opening.** A candidate or store file is opened with `O_NOFOLLOW | O_NONBLOCK`, so neither a final symbolic link nor a FIFO or device node can hold the open, and cancellation is checked around it. `fstat` must show a regular file before anything is read, and `/proc/self/fd` must show that the opened object is exactly the expected path under the anchored root, which also defeats a symbolic link in, or a swap of, any directory on the way (if that cannot be read, the agent fails closed).
- **Hashing.** The bytes are hashed through that descriptor, with an identical `fstat` before and after. A copy is hashed as it is written, and its source must not change while it is read. That digest proves only the bytes read from the source, not the bytes now in the staging file: once the copy is written and synced, the staging file is hashed again through its own descriptor, with an identical `fstat` before and after (the `verifying` phase). A staging file that does not match is `SHA256_MISMATCH`, one that changes while it is read back is `CHANGED_WHILE_VERIFYING`; either way the staging file is removed, any partial is kept, nothing is recorded, and no other candidate is tried.
- **Downloads and resumed prefixes.** A download's streamed digest only proves the bytes received. Before publication the finished partial, including a resumed prefix, is hashed again through its own descriptor with an identical `fstat` before and after; a change is `SHA256_MISMATCH` (the partial is removed) or, while it is being hashed, `CHANGED_WHILE_VERIFYING`. Nothing that changed during transfer or verification is recorded as verified.
- **Records.** The record keeps the identity of the object that was hashed: after the agent's own rename, the final path must hold the same device, inode, size and mtime (only ctime, which the rename changes, may differ). A copy is published while the descriptor it was read back through is still open: its full `fstat`, ctime included, must still be the verified one before the record is written and the file renamed (`CHANGED_WHILE_VERIFYING` otherwise).
- **Links in the store.** A symbolic link anywhere in the store is treated as absent and logged: at a file's path it is replaced by the fetched file (its target is never touched); in a directory on the way it refuses the Run with `UNSAFE_PATH`.
- **Directories.** The final, staging and record (`.state`) directories are created under the store root, refusing linked ancestors, before anything is placed or recorded, including the in-place adoption of a canonical snapshot that has no `.state` yet.

**Staging and resume.** Each candidate is copied to its own staging file (`<file>.partial.copy-<random>`, next to the partial). An existing `.partial` and its identity record stay untouched until a verified replacement is published; after a mismatch, an error or a Stop only the staging file is removed, and a later download resumes from the earlier offset. Staging files left by a crash are removed at the next attempt.

**A candidate never makes a resume impossible.** A `/shared` candidate is matched by size alone until it is hashed, so any file of the pinned size (even a sparse one that takes no disk) counts as one; its allocated blocks are not trusted to say otherwise. A partial counts when it is a regular file whose identity record matches and that is not longer than the pinned size; the overview, the plan and the transfer use this one rule. Beside such a partial holding at least one byte, the plan's method is `copy-or-resume` and the disk it needs is the resume's (`copyBytesNeeded` is the copy's): the transfer makes the copy only when the whole copy and its reserve fit, and otherwise leaves the candidate unread and resumes the partial in place, with one line in the deployment log. One free-space measurement per candidate, taken just before its copy would start, makes that choice and gives any refusal its figures, so two readings can never disagree. A copy that fitted by its measurement and then meets a full disk still pauses (`PAUSED_ENOSPC`, partial kept, below), and the next Run chooses again. Without a partial the plan is `copy` and needs the whole size.

**Copies follow the download's rules.** Free space for the copy plus the 5 % reserve is checked before copying (`INSUFFICIENT_SPACE` when there is no partial to resume instead). The copy runs in abortable chunks (8 MiB), so Stop, Cancel and drain settle within one chunk. Progress reports the `copying` phase, and hashing bytes already on disk reports the `verifying` phase. A full disk pauses the deployment (`PAUSED_ENOSPC`); other read and write errors fail it (`COPY_FAILED`); none is ever treated as "no candidate".

**Bounded, cancellable lookup.** One streaming walk of `/shared/models` per artifact (the overview uses one walk for every listed model) collects every size the artifact needs. It follows no directory link, stops at 8 levels and 20,000 entries, and checks for Stop before every directory it opens and at every entry. The Run's acquisition planning is cancellable too: Stop, Cancel and drain abort it before they queue behind the Run, so the walk ends once the pending read returns and the Run fails with `cancelled`. That holds whatever the walk found, including an empty or missing root: an aborted Run never reaches admission or creates a job. A Cancel that stopped the planning succeeds (the deployment is unchanged); a Cancel with no planning and no transfer is still `not_downloading`. At most 3 same-size candidates are hashed per file, those with the file's name first.

**Last check before loading.** Just before the runner starts, after any runner preparation, every file of the artifact is checked again: an unchanged stat, or its bytes are hashed again, shown as the `verifying` phase. A file that no longer matches fails the start (`CHANGED_AFTER_VERIFY`) before any runner is launched. This detects accidental change; it is not a lock.

**Provenance.** Every file and shard carries one shape, `{ file, source, method, bytes }`: `method` is `in-place`, `copy` or `download`; `source` is the store path, the `/shared` path or the download URL (never a token); `bytes` is what that way moved (0 in place, the size for a copy, the bytes transferred by that download). It is recorded with the file, returned by the fetch, stored as the deployment's `provenance`, and written to the deployment log, one line per file; a file that was already verified is logged as such with its recorded provenance. The Run's plan (`acquisition`: each file's method, `owned`, `in-place`, `copy`, `copy-or-resume` or `download`, and the disk it needs) is stored with the deployment when it is accepted.

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
| `ollama` (`ollama`), fetched by the runner | The Ollama adapter starts `ollama serve` with `OLLAMA_MODELS=/data/models/ollama` and streams `/api/pull`. When the entry pins `manifestDigest` (every seed, and every tag added through `local_llm_model_add`), the stored manifest must hash to it. |
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
