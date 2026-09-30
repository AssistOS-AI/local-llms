# local-llm Specification Matrix

The `local-llm` agent is the repository's only agent, and this is the repository's specification set. The retired shared-image agents' specifications (repository `docs/specs` DS000–DS011) are kept at commit [`03697765`](https://github.com/AssistOS-AI/local-llms/tree/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs); DS000 says what was retired.

| Spec | Title | Status | Summary |
| --- | --- | --- | --- |
| [DS000](DS000-vision-and-scope.md) | Vision and Scope | accepted | What local-llm is (GPU when present, CPU otherwise), the retired shared-image agents and the services that went with them, its runners (LM Studio for internal use only), and what it deliberately leaves out. |
| [DS001](DS001-agent-contract.md) | Agent Contract | accepted | Manifest, process model, tools, authorization, the chat endpoint, drain, and LM Studio's operator switch and limits. |
| [DS002](DS002-model-lifecycle.md) | Model Lifecycle | accepted | Catalog, registry, looking up and adding models at run time, on-demand download, the deployment state machine, and restart behaviour. Records what is reused from the retired repository DS008. |
| [DS003](DS003-gpu-and-resources.md) | GPU and Resources | accepted | The Box GPU grant, running without it on the CPU, admission, runner isolation, and memory estimates. Records what is reused from the retired repository DS010. |
| [DS004](DS004-on-demand-runners.md) | On-demand Runners | accepted | Runners installed after the image is built: the lock in the image and the agent's own CI-gated lock (Phase 3), the verified cache in /data, the runnable copy rebuilt in the container, and the CI install check, which never downloads a proprietary entry. |
| [DS005](DS005-hardware-profiles.md) | Hardware Profiles | accepted | The dedicated, unified and cpu hardware profiles, how the GPU's memory model and its absence are detected, what changes on a GPU that shares system memory (NVIDIA GB10 in DGX Spark), and how models run on the CPU when no NVIDIA GPU is usable. |
| [DS006](DS006-coding-style.md) | Coding Style | accepted | Coding conventions, source layout, language rules and test organization; carries forward the repository coding-style rules. |

Tests, from the repository root: `node --test 'local-llm/tests/*.test.mjs'` or `npm --prefix local-llm test` (no Docker, GPU or network required). A bare directory argument does not work on Node 24.
