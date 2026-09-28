# local-llms

Ploinky repository with one agent, [`local-llm`](local-llm/), which runs open-weight large language models on the machine's NVIDIA GPU and serves them to the workspace's other agents through the local Soul Gateway (`soul_gateway/local-llms/local-llm/default`).

- **x86 hosts with a GPU of their own:** llama.cpp, ik_llama.cpp and Ollama in the image; vLLM and TabbyAPI installed on demand; LM Studio for internal use only, behind an operator switch.
- **NVIDIA DGX Spark** (arm64, GB10, memory shared with the system): llama.cpp, for models measured there, sized against the shared memory pool.

Admins pick a model and a runner in Settings > Agents > Local LLMs. Weights download only when Run is pressed, pinned by commit, size and sha256. Admission says before anything downloads whether a model and its parameters fit.

The image is built in [`container-image-builds/images/local-llm`](https://github.com/AssistOS-AI/container-image-builds/tree/main/images/local-llm) as one multi-arch index (amd64 and arm64).

## Documentation

The specifications are the source of truth: [`local-llm/docs/specs/matrix.md`](local-llm/docs/specs/matrix.md) (DS000–DS006). Coding style: [`DS006`](local-llm/docs/specs/DS006-coding-style.md).

## Tests

```sh
node --test 'local-llm/tests/*.test.mjs'
```

No Docker, GPU or network is required.

## Retired agents

Until 2026-09-28 this repository also held twelve CPU agents in the shared `assistos/local-llms` image (`local-llms-manager` and eleven model agents, including the translation and reranking services). They were retired without a replacement; [DS000](local-llm/docs/specs/DS000-vision-and-scope.md) lists what went with them. Their last revision is commit [`03697765`](https://github.com/AssistOS-AI/local-llms/tree/03697765a35d92adf4c7db1be9616c0b5a64cb63). Removing them deleted no stored models or workspace data.
