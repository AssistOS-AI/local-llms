# Specification Matrix

| Name | Description |
| --- | --- |
| [DS000-vision-and-scope](DS000-vision-and-scope.md) | What local-llm is, the retired shared-image agents and the services that went with them, its runners (LM Studio for internal use only), and what it deliberately leaves out. |
| [DS001-agent-contract](DS001-agent-contract.md) | Manifest, process model, tools, authorization, the chat endpoint, drain, and LM Studio's operator switch and limits. |
| [DS002-model-lifecycle](DS002-model-lifecycle.md) | Catalog, registry, on-demand download, the deployment state machine, and restart behaviour. |
| [DS003-gpu-and-resources](DS003-gpu-and-resources.md) | Manifest-declared GPU access and the operator override, admission, runner isolation, and memory estimates. |
| [DS004-on-demand-runners](DS004-on-demand-runners.md) | Runners installed after the image is built, from a lock shipped in the image or the agent's own CI-gated lock (Phase 3), a verified cache in /data and a runnable copy rebuilt in the container; a proprietary entry is never downloaded in CI. |
| [DS005-hardware-profiles](DS005-hardware-profiles.md) | NVIDIA dedicated and unified memory, CPU fallback, discrete AMD Vulkan and Apple Venus profiles, their detection, admission and memory guards. |
| [DS006-coding-style](DS006-coding-style.md) | Coding conventions, source layout, language rules and test organization for local-llm, the repository's only agent; carries forward the repository coding-style rules of the retired shared-image agents. |
