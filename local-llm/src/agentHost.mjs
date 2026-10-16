// The container's host for the pworker engine (DS002, DS003, DS004): where the
// image keeps things, with the same environment overrides the agent has always
// honored. The engine reads no environment variable for a path; this file is
// the one place that does. `||` means an empty value takes the default, as it
// did before the engine moved.
//
// `defineHost` refuses a path that is relative, not normalized or holds ':'.
// A relative override is therefore resolved against the working directory,
// which is what a read of the relative path would have done. A bare
// LOCAL_LLM_NVIDIA_SMI (a command, not a path) is looked up in the environment's
// PATH, as execFile did. A value that holds ':' is refused by `defineHost`.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineHost, linuxPlatform } from 'pworker/local-llm';

const CATALOG_DIR = fileURLToPath(new URL('../catalog/', import.meta.url));
const AGENT_LOCK_PLATFORMS = Object.freeze({ arm64: 'arm64', x64: 'amd64' });

/** The agent's seed model catalog. */
export function agentSeedCatalogFile() {
    return path.join(CATALOG_DIR, 'models.json');
}

/** The agent's runner lock for a CPU architecture (Node's `process.arch`), or null for an architecture that has none. */
export function agentRunnerLockFile(arch = process.arch) {
    if (!Object.hasOwn(AGENT_LOCK_PLATFORMS, arch)) return null;
    return path.join(CATALOG_DIR, `runners.lock.linux-${AGENT_LOCK_PLATFORMS[arch]}.json`);
}

// A path override as the file system would have read it: relative to the working directory.
function overridePath(value) {
    return path.resolve(value);
}

// A command that names no directory is found in PATH, like execFile does.
function commandPath(value, env) {
    if (value.includes('/')) return path.resolve(value);
    for (const directory of String(env.PATH || '').split(path.delimiter)) {
        if (!directory) continue;
        const candidate = path.resolve(directory, value);
        try {
            fs.accessSync(candidate, fs.constants.X_OK);
            if (fs.statSync(candidate).isFile()) return candidate;
        } catch {}
    }
    return path.resolve(value);
}

/** The host of the local-llm container, with the environment's overrides. */
export function agentHost(env = process.env) {
    return defineHost({
        dataDir: env.LOCAL_LLM_DATA_DIR ? overridePath(env.LOCAL_LLM_DATA_DIR) : '/data',
        runRoot: env.LOCAL_LLM_RUN_ROOT ? overridePath(env.LOCAL_LLM_RUN_ROOT) : '/opt/runners',
        sharedModelsRoot: env.LOCAL_LLM_SHARED_MODELS ? overridePath(env.LOCAL_LLM_SHARED_MODELS) : '/shared/models',
        imageContractFile: '/opt/local-llm/source.contract',
        imageRunnerLockFile: env.LOCAL_LLM_RUNNER_LOCK ? overridePath(env.LOCAL_LLM_RUNNER_LOCK) : '/opt/local-llm/runners.lock.json',
        agentRunnerLockFile,
        seedCatalogFile: env.LOCAL_LLM_CATALOG_FILE ? overridePath(env.LOCAL_LLM_CATALOG_FILE) : agentSeedCatalogFile(),
        hfBaseUrl: env.LOCAL_LLM_HF_BASE_URL || 'https://huggingface.co',
        nvidiaSmi: env.LOCAL_LLM_NVIDIA_SMI ? commandPath(env.LOCAL_LLM_NVIDIA_SMI, env) : '/usr/local/nvidia/bin/nvidia-smi',
        nvidiaLibDir: '/usr/local/nvidia/lib64',
        executables: {
            'llama.cpp': '/opt/llama.cpp/llama-server',
            'ik_llama.cpp': '/opt/ik_llama.cpp/llama-server',
            'llama.cpp-vulkan': '/opt/llama.cpp-vulkan/llama-server',
            ollama: '/opt/ollama/bin/ollama',
        },
        vulkan: {
            serverPath: '/opt/llama.cpp-vulkan/llama-server',
            vulkaninfo: '/usr/bin/vulkaninfo',
            icdDir: '/opt/local-llm/vulkan/icd.d',
        },
        lmStudioSdkDir: '/opt/local-llm/lmstudio-sdk',
        uv: '/usr/local/bin/uv',
        python: '/usr/bin/python3',
        shmDir: '/dev/shm',
        toolPath: '/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin',
        gpuAccess: {
            report: (processEnv) => ({ unavailable: processEnv.PLOINKY_GPU_STATUS === 'unavailable', reason: processEnv.PLOINKY_GPU_REASON }),
            source: 'ploinky-environment',
        },
        platform: linuxPlatform(),
    });
}
