# local-llm Specification Matrix

The `local-llm` agent keeps its own specification set. Its numbering is local to this directory and is separate from the repository-level sequence under `docs/specs/`, which describes the legacy shared-image agents.

| Spec | Title | Status | Summary |
| --- | --- | --- | --- |
| [DS000](DS000-vision-and-scope.md) | Vision and Scope | accepted | What local-llm is, how it differs from the legacy agents, and what it deliberately leaves out. |
| [DS001](DS001-agent-contract.md) | Agent Contract | accepted | Manifest, process model, tools, authorization, the chat endpoint, and drain. |
| [DS002](DS002-model-lifecycle.md) | Model Lifecycle | accepted | Catalog, registry, on-demand download, the deployment state machine, and restart behaviour. Records what is reused from repository DS008. |
| [DS003](DS003-gpu-and-resources.md) | GPU and Resources | accepted | The Box GPU grant, admission, runner isolation, and memory estimates. Records what is reused from repository DS010. |

Tests: `node --test local-llm/tests/` (no Docker, GPU or network required).
