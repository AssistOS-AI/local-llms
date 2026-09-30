// The dedicated and unified overviews the GPU-profile regression test compares (Phase 3, DS004).
// Not a test file itself (the gate runs tests/*.test.mjs only). It depends on nothing Phase 3 added, so the
// same scenarios ran against the tree at a84617d to produce overview-golden-a84617d.json, which holds what
// the dedicated and unified profiles showed before llama.cpp's CPU build and the agent's lock existed.
//
// To regenerate: extract that commit (`git archive a84617d local-llm | tar -x -C <dir>`), then in <dir>/local-llm
// call collectOverviews with that tree's createController, createStateStore and validateModel and a makeInstaller
// that builds its installer from the image lock alone (loadRunnerLock), and write the result.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SHA = 'a'.repeat(64);

// The image's runner lock, reduced: one python and one proprietary archive runner, as the amd64 image has.
const IMAGE_LOCK = {
    schema: 'local-llm.runners-lock/v1',
    runners: {
        vllm: {
            version: '0.30.0', kind: 'python',
            licence: { name: 'Apache-2.0', url: 'https://github.com/vllm-project/vllm/blob/v0.30.0/LICENSE' },
            files: [{ name: 'vllm-0.30.0-py3-none-any.whl', url: 'https://files.pythonhosted.org/packages/vllm-0.30.0-py3-none-any.whl', size: 1000, sha256: SHA }],
        },
        lmstudio: {
            version: '0.0.25-1', kind: 'archive',
            licence: { name: 'LM Studio Terms', url: 'https://lmstudio.ai/app-terms', requiresAcceptance: true, proprietary: true },
            files: [{ name: 'llmster.tar.gz', url: 'https://llmster.lmstudio.ai/download/llmster.tar.gz', size: 2000, sha256: 'b'.repeat(64), strip: 0 }],
        },
    },
};

const GGUF = { type: 'huggingface', repo: 'acme/models', file: 'm.gguf', revision: 'main', commit: 'a'.repeat(40), size: 400 * MIB, sha256: 'b'.repeat(64) };

const MODELS = [
    { id: 'both-formats', displayName: 'Both formats', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 12288 },
        sources: { gguf: GGUF, ollama: { type: 'ollama', tag: 'both:1b', manifestDigest: `sha256:${'c'.repeat(64)}`, size: 400 * MIB } } },
    { id: 'gguf-only', displayName: 'GGUF only', architecture: 'moe', contextLength: 131072, sources: { gguf: { ...GGUF, file: 'n.gguf', size: 9 * GIB } } },
    { id: 'tag-only', displayName: 'Tag only', sources: { ollama: { type: 'ollama', tag: 'tag:1b' } } },
];

const x86 = {
    available: true, name: 'NVIDIA GeForce RTX 3060 Laptop GPU', driverVersion: '595.91.07', memoryModel: 'dedicated',
    totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [],
    device: { pciDeviceId: '0x252010DE', computeCapability: '8.6', addressingMode: 'None' },
};
const gb10 = {
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified', totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
};

export const SCENARIOS = Object.freeze({
    'dedicated, amd64 image': {
        gpu: x86, hostArch: 'x64', memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
        imageContract: { architecture: 'amd64', llama_cpp: 'b11159', ik_llama_cpp: '20f7a72' },
        present: ['/opt/llama.cpp/llama-server', '/opt/ik_llama.cpp/llama-server', '/opt/ollama/bin/ollama'],
    },
    'dedicated, no image contract': {
        gpu: x86, hostArch: 'x64', memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB }, imageContract: null, present: null,
    },
    'unified, arm64 image': {
        gpu: gb10, hostArch: 'arm64', memory: { totalBytes: 125442396 * 1024, availableBytes: 108 * GIB },
        imageContract: { architecture: 'arm64', llama_cpp: 'b11159', gpu_compute_capabilities: '12.1' },
        present: ['/opt/llama.cpp/llama-server'],
    },
    'unified, no image contract': {
        gpu: gb10, hostArch: 'arm64', memory: { totalBytes: 125442396 * 1024, availableBytes: 108 * GIB }, imageContract: null, present: null,
    },
});

/**
 * The overview of every scenario, as plain JSON: runners (with their install state) and models (with their per-runner
 * rows), the hardware and the profile. `makeInstaller({ imageLockFile, dataDir })` builds the installer, so a caller
 * chooses which locks it reads. Only the decision's timestamp is left out.
 */
export async function collectOverviews({ createController, createStateStore, validateModel, makeInstaller }) {
    const overviews = {};
    for (const [name, scenario] of Object.entries(SCENARIOS)) {
        const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-overview-'));
        let controller = null;
        try {
            const imageLockFile = path.join(dataDir, 'image.lock.json');
            fs.writeFileSync(imageLockFile, JSON.stringify(IMAGE_LOCK));
            controller = createController({
                dataDir,
                env: { PATH: '/usr/bin' },
                seedCatalog: MODELS.map((model) => validateModel(model, { seed: true })),
                stateStore: createStateStore({ dataDir }),
                snapshot: async () => ({ gpu: structuredClone(scenario.gpu), memory: { ...scenario.memory }, disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB }, cpus: 20 }),
                installer: makeInstaller({ imageLockFile, dataDir }),
                detectRunner: (runner) => ({ installed: ['llama.cpp', 'ik_llama.cpp', 'ollama'].includes(runner.id), version: runner.pinnedVersion, reason: null }),
                imageContract: scenario.imageContract,
                fileExists: scenario.present ? (file) => scenario.present.includes(file) : () => true,
                hostArch: scenario.hostArch,
                sharedModelsRoot: null,
            });
            const overview = JSON.parse(JSON.stringify(await controller.overview()));
            if (overview.profileDecision) delete overview.profileDecision.decidedAt;
            overviews[name] = overview;
        } finally {
            await controller?.drain();
            fs.rmSync(dataDir, { recursive: true, force: true });
        }
    }
    return overviews;
}
