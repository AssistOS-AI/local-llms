---
id: DS000
title: Vision and Scope
status: accepted
owner: local-llm
summary: What local-llm is, how it differs from the legacy agents, and what it deliberately leaves out.
---

# DS000 Vision and Scope

## Introduction

`local-llm` runs open-weight language models on the workspace's own NVIDIA GPU. A workspace administrator picks a model and a runner in Explorer's Settings, presses Run, and the agent downloads the weights, starts the runner on the GPU, and exposes the running model to other agents through the workspace-local Soul Gateway.

## Core Content

### Relation to the legacy agents

The repository's other twelve agents share the CPU image `assistos/local-llms`, contain no application source and start one fixed model each (repository DS001, DS002). `local-llm` is a separate agent with its own image (`container-image-builds/images/local-llm`), its own Node.js source under `local-llm/src/`, and its own tests under `local-llm/tests/`. The legacy agents, their image and their dispatcher are unchanged. The repository-level statement that agent directories hold no source code describes the legacy agents only.

### Goals

| Goal | Where |
| --- | --- |
| Weights are downloaded only after an explicit Run, never at enable, restart or when a model is added. | DS002 |
| The GPU is used only through the named-agent Box GPU grant; the agent asks for no privilege, capability or host device of its own. | DS003 |
| Every control operation is admin-only and goes through one serialized controller. | DS001 |
| Other agents use the running model through AchillesAgentLib and the local Soul Gateway, never through a runner port. | DS001 |
| Admission explains, before anything is downloaded, whether a model and its parameters fit this machine. | DS003 |

### Runners

| Runner | State in this release |
| --- | --- |
| llama.cpp `b11125` (CUDA 12.8 build) | Supported. GGUF from Hugging Face, pinned by commit, size and sha256. |
| Ollama `0.34.3` | Supported. Library tags pulled by the Ollama daemon, verified by manifest digest when the catalog pins one. |
| vLLM | Listed, not supported: it cannot load the MXFP4 GGUF the seed model ships as. |
| LM Studio | Listed, not supported: it needs a licence acceptance and an engine download that this release does not perform. |

### Out of scope

A CPU fallback, a shared host model cache, installing runners on demand, publishing the image, and adding the agent to Explorer's default manifest. Each needs its own decision.

## Decisions & Questions

### Question #1: Why a separate agent instead of extending the shared image?

Response: The shared image is CPU-only and arm64-first, starts one fixed model per agent, and its dispatcher is shell. GPU runners, on-demand weights, admission and a Settings UI need a long-running controller with state; keeping the legacy agents unchanged avoids regressions for their users.

## Conclusion

`local-llm` is an opt-in, per-workspace, GPU-backed model host that downloads nothing until an administrator runs a model, and serves other agents only through the local Soul Gateway.
