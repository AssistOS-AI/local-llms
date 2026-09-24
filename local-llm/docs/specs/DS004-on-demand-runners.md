---
id: DS004
title: On-demand Runners
status: accepted
owner: local-llm
summary: Runners installed after the image is built, from a lock shipped in the image, a verified cache in /data and a runnable copy rebuilt in the container.
---

# DS004 On-demand Runners

## Introduction

llama.cpp, ik_llama.cpp and Ollama are part of the agent image. Heavier runners, the PyTorch-based vLLM and ExLlamaV3 + TabbyAPI, would add several gigabytes to every Explorer deployment, so an administrator installs them only where they are wanted (runners plan, decision R2 = c). This specification defines what may be installed, where the files go, how they are checked, and when the runnable copy is rebuilt.

## Core Content

### What may be installed

The image ships `/opt/local-llm/runners.lock.json` (schema `local-llm.runners-lock/v1`, `src/controller/runnerLock.mjs`). For each installable runner it names the version, the kind (`python` or `archive`), the licence, every file with its URL, size and sha256, and the checks the CI install job runs. Administrators choose a runner id; they never supply a URL. The lock accepts only plain https URLs on `files.pythonhosted.org`, `github.com`, `codeload.github.com`, `download.pytorch.org` and `openaipublic.blob.core.windows.net`, and each file is pinned by size and sha256 as well.

A `python` runner's files are wheels, source archives with `extract` (the directory they are unpacked into), and data files with `into` (the directory of the runnable copy they are copied into). A data file is something the runner would otherwise download when it runs. vLLM's `o200k_base.tiktoken`, the tokenizer vocabulary gpt-oss's chat format needs, is pinned this way into `tiktoken/`, and the vLLM adapter points `TIKTOKEN_ENCODINGS_BASE` there. openai_harmony then reads the file and checks its sha256 instead of fetching it from openaipublic (runners plan §6: nothing unpinned is downloaded at run time). vLLM loads harmony only on the first gpt-oss chat request, so a gpt-oss snapshot (`model_type` `gpt_oss`) is refused at start with `runner_incomplete` when the runnable copy lacks the file, for example an install from an older lock.

### Where the files go

| Part | Path | Lifetime |
| --- | --- | --- |
| Cache | `/data/runners/<id>/<version>/files/<name>`, and `installed.json`, the install record | Persists: the only part that survives the container |
| Runnable copy | `/opt/runners/<id>/<version>/` in the container's own filesystem. The image creates `/opt/runners` owned by uid 1000. | The container's lifetime |

### Install

| Step | Rule |
| --- | --- |
| Start | `local_llm_runner_install { runnerId, acceptLicence }`, admin only. Nothing installs automatically. One install runs at a time. |
| Licence | A runner whose lock entry sets `requiresAcceptance` is refused with `licence_required` unless `acceptLicence` is true. The acceptance is recorded in the install state and in `installed.json`, with who accepted (the verified caller from the Router-signed invocation, never tool input) and when. The dashboard shows the licence, its source link and the notice before the Install is sent. |
| Download | Each file goes through the downloader's resume, size and free-disk checks and is verified against its sha256 before it is used (`downloadFile`). The whole install first needs free space for the remaining bytes plus 5 %. A Stop, a drain or a network failure leaves the install `paused`; the next Install resumes the partial files. |
| Record | `installed.json` is written only after every file verified. |
| Runnable copy | Built right after the download (below), so the install also measures the rebuild time. |
| Update | A new lock version is installed beside the old one. The old version's cache and copy are removed only after the new version is in place. |

### Runnable copy

Before the first launch in a container, and at the end of an install, `ensureRunnable` checks the copy's `.ready.json` marker. If the marker names this exact lock entry, the copy is reused, so a process restart in the same container does not rebuild. Otherwise:

1. every cached file is copied into container-local staging (`/opt/runners/.stage-<id>`) and hashed as it is copied; any difference from the lock blocks the rebuild with `cache_changed`. The steps below read only the staged copies, so a file replaced in `/data` after the check cannot reach the runnable copy;
2. the copy's directory is emptied;
3. a `python` runner gets a virtual environment from `/usr/bin/python3` (`uv venv`), and `uv pip install --offline --no-index --find-links <staging> --no-deps --require-hashes --no-cache` installs exactly the locked wheels; source archives are unpacked where the lock says, and data files are copied read-only (0444) into their `into` directory;
4. an `archive` runner's archives are unpacked;
5. the staging is removed and the marker is written last.

Only one build of a runner's copy runs at a time: a caller that arrives while it is being built (an Install finishing while a Run starts) shares that build. A Run on a runner whose install is in progress is refused with `busy`.

Every step runs in its own process group, so a Stop or a drain ends it. `uv` gets a minimal environment (`UV_OFFLINE`, `UV_PYTHON_DOWNLOADS=never`, no agent secrets). Code changed in the runnable copy lasts only as long as the container, like code baked into the image; `ploinky restart` creates a new container, which rebuilds the copy from the verified cache.

### Uninstall

`local_llm_runner_uninstall { runnerId }` deletes the runner's cache and runnable copy, clears its install state and re-runs runner detection. It is refused while the runner runs a model, and it cancels an install in progress.

### Detection

A runner is installable when it is in the image's lock and this release has an adapter for it; a lock entry without an adapter (a runner a later release will run) is installed only by the CI install check. An installable runner is installed when its cache is complete and its record names the lock entry. Detection reads files only; it never imports Python packages. It runs again after every install and uninstall, so the dashboard and `local_llm_overview` follow without an agent restart. A Run on an installable runner that is not installed is refused with `runner_not_installed`.

### CI install check

The image's publish workflow installs every lock entry inside the published image with `tools/runner_install_check.mjs`, one entry at a time. It checks each pinned distribution's version, imports the modules that load without a GPU, lists the ones that need the driver, and runs `ldd` on every shared object in the runnable copy. A library `ldd` cannot find passes only if the Box GPU grant supplies it (`libcuda.so.1`), if another file in the same environment has that name (wheels load each other's libraries at run time, as torch loads its CUDA libraries), or if the lock entry lists it in `check.optionalLibraries` with the reason the runner never needs it (for example MPI and RDMA libraries for multi-node transports). Any other missing library fails the check, and the report lists all three groups. Every data file (`into`) must also be in the runnable copy with the lock's sha256.

## Decisions & Questions

### Question #1: Why rebuild the runnable copy instead of running from /data?

Response: `/data` is a workspace bind that survives containers and that other processes with workspace access can write. Rebuilding into the container's own filesystem from a cache that is re-verified against the image's lock means that code changed on disk never outlives the container, the same guarantee code baked into the image has. The check and the build use the same bytes: each file is hashed while it is copied into container-local staging, and the build reads only that copy, so a writer that swaps a cached file after the check gains nothing.

### Question #2: Why `--no-deps` with the full closure?

Response: The lock already lists the resolved closure for this image's Python, so the install needs no resolver: every wheel is named with its hash, and nothing else can be installed.
