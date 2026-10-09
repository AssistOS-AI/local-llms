# Hardware budget dependencies

The hardware budget implementation adds no third-party dependency. It uses Node.js 24 ES modules, native cryptographic hashing, and existing local controller/runner modules. Its tests use node:test and injected hardware, runner processes and reviewed-data fixtures. No GPU, container, network download or package installation is needed for those tests.

Ploinky continues to supply the shared AchillesAgentLib runtime. Existing GPU queries, runner wheels, binary pins, licenses and installation checks remain governed by local-llm/docs/specs/DS003-gpu-and-resources.md and DS004-on-demand-runners.md. No alternate AgentLib source or new Python code is introduced.
