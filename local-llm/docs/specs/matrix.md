# Specification Matrix

| Name | Description |
| --- | --- |
| [DS000-vision-and-scope](DS000-vision-and-scope.md) | What local-llm is, how its Ploinky glue and hardware limits split from the pworker engine, the retired shared-image agents and the services that went with them, its runners (LM Studio for internal use only), and what it deliberately leaves out. |
| [DS001-agent-contract](DS001-agent-contract.md) | Manifest, the binding to the pworker engine and its cold-start dependency, process model, tools, authorization, the chat endpoint, drain, and LM Studio's operator switch and limits. |
| [DS002-model-lifecycle](DS002-model-lifecycle.md) | Catalog, registry, on-demand download, the deployment state machine and restart behaviour as the agent hosts them: the engine's specification at the pinned pworker commit, and the agent's seed catalog, paths, overrides and tools. |
| [DS003-gpu-and-resources](DS003-gpu-and-resources.md) | Manifest-declared GPU access and the operator override, the container's hardware limits (memory, CPU quota, GPU share) and the plug-in that applies them to the engine's admission, and runner isolation; the engine's snapshot, admission and estimates are in the pinned pworker specification. |
| [DS004-on-demand-runners](DS004-on-demand-runners.md) | Runners installed after the image is built, from a lock shipped in the image or the agent's own CI-gated lock (Phase 3), a verified cache in /data and a runnable copy rebuilt in the container; a proprietary entry is never downloaded in CI. |
| [DS005-hardware-profiles](DS005-hardware-profiles.md) | The hardware profiles the agent runs (NVIDIA dedicated and unified, CPU, discrete AMD Vulkan, experimental shared-memory APU, Apple Venus), how the container's limits narrow them, the memory guard, the GPU share and its vLLM qualification, and the dashboard; the engine's detection, admission and estimates are in the pinned pworker specification. |
| [DS006-coding-style](DS006-coding-style.md) | Coding conventions, source layout, language rules and test organization for local-llm, the repository's only agent; carries forward the repository coding-style rules of the retired shared-image agents. |
