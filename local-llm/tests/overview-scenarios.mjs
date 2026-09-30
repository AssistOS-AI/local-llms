// What the GPU-profile regression test compares (Phase 3, DS004): the dedicated and unified overviews, and
// adding and running an Ollama tag on dedicated hosts. Not a test file itself (the gate runs tests/*.test.mjs
// only). It depends on nothing Phase 3 added, so the same scenarios ran against the tree at a84617d to produce
// overview-golden-a84617d.json, which holds what those profiles did before llama.cpp's CPU build, the agent's
// lock and the Ollama pin existed.
//
// To regenerate: extract that commit (`git archive a84617d local-llm | tar -x -C <dir>`), then in <dir>/local-llm
// call collectOverviews and collectOllamaRuns with that tree's createController, createStateStore and
// validateModel and a makeInstaller that builds its installer from the image lock alone (loadRunnerLock), and
// write { scenarios, ollamaRuns } beside a _meta note.
import crypto from 'node:crypto';
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

// The runner's manifest as a finished pull leaves it, and where Ollama's store keeps it.
const FAKE_MANIFEST = JSON.stringify({
    schemaVersion: 2, layers: [{ digest: `sha256:${'b'.repeat(64)}`, size: 16, mediaType: 'application/vnd.ollama.image.model' }],
});

function writeOllamaStore(dataDir, tag) {
    const modelsDir = path.join(dataDir, 'models', 'ollama');
    fs.mkdirSync(path.join(modelsDir, 'blobs'), { recursive: true });
    fs.writeFileSync(path.join(modelsDir, 'blobs', `sha256-${'b'.repeat(64)}`), Buffer.alloc(16));
    const [name, version = 'latest'] = tag.split(':');
    const target = path.join(modelsDir, 'manifests', 'registry.ollama.ai', 'library', name, version);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, FAKE_MANIFEST);
}

export const OLLAMA_RUN_SCENARIOS = Object.freeze({
    'dedicated, amd64 image with Ollama in it': {
        imageContract: { architecture: 'amd64', llama_cpp: 'b11159', ik_llama_cpp: '20f7a72' },
        present: ['/opt/llama.cpp/llama-server', '/opt/ik_llama.cpp/llama-server', '/opt/ollama/bin/ollama'],
    },
    'dedicated, no image contract': { imageContract: null, present: null },
    'dedicated, arm64 image without Ollama': {
        imageContract: { architecture: 'arm64', llama_cpp: 'b11159', gpu_compute_capabilities: '8.6' }, present: ['/opt/llama.cpp/llama-server'],
    },
});

/**
 * On a dedicated host: add an Ollama library tag (what the dashboard's form stores), run it, and record what came
 * of each step as plain JSON: the stored entry, the process the controller launched, every request it made to
 * Ollama, the chat target, and the deployment's facts; or the code and message of a refusal. `resolveOllama` is
 * handed to the controller as it is, for a caller that wants to know whether the registry was asked. Paths under
 * the data directory are written as <data>.
 */
export async function collectOllamaRuns({ createController, createStateStore, validateModel, makeInstaller, resolveOllama = undefined }) {
    const runs = {};
    for (const [name, scenario] of Object.entries(OLLAMA_RUN_SCENARIOS)) {
        const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-ollama-run-'));
        let controller = null;
        try {
            const imageLockFile = path.join(dataDir, 'image.lock.json');
            fs.writeFileSync(imageLockFile, JSON.stringify(IMAGE_LOCK));
            const started = [];
            const calls = [];
            controller = createController({
                dataDir,
                env: { PATH: '/usr/bin' },
                seedCatalog: [validateModel(MODELS[0], { seed: true })],
                stateStore: createStateStore({ dataDir }),
                snapshot: async () => ({ gpu: structuredClone(x86), memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB }, disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB }, cpus: 20 }),
                installer: makeInstaller({ imageLockFile, dataDir }),
                imageContract: scenario.imageContract,
                fileExists: scenario.present ? (file) => scenario.present.includes(file) : () => true,
                hostArch: 'x64',
                sharedModelsRoot: null,
                download: async () => ({ status: 'complete', path: '/data/models/gguf/x.gguf', bytesTransferred: 0 }),
                inspect: async () => ({ state: 'absent', bytes: 0 }),
                remove: async () => 0,
                startRunner({ command, args, env }) {
                    let running = true;
                    let resolveExit;
                    const handle = {
                        pid: 5000 + started.length, command, args, env,
                        exited: new Promise((resolve) => { resolveExit = resolve; }),
                        get running() { return running; },
                        async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
                    };
                    started.push(handle);
                    return handle;
                },
                fetchImpl: async (url, options = {}) => {
                    const pathname = new URL(String(url)).pathname;
                    calls.push([String(url), options.method || 'GET', options.body ? JSON.parse(options.body) : null]);
                    if (pathname === '/api/pull') {
                        writeOllamaStore(dataDir, 'gpt-oss:20b');
                        return { ok: true, status: 200, body: [new TextEncoder().encode(`${JSON.stringify({ status: 'success' })}\n`)] };
                    }
                    if (pathname === '/api/ps') {
                        return { ok: true, status: 200, json: async () => ({ models: [{ name: 'gpt-oss:20b', size: 16, size_vram: 8, context_length: 4096 }] }) };
                    }
                    return { ok: true, status: 200, json: async () => ({}) };
                },
                apiKeyFactory: () => 'k'.repeat(43),
                detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
                pollMs: 2,
                stopGraceMs: 50,
                readMemory: () => ({ totalBytes: 31 * GIB, availableBytes: 24 * GIB }),
                readPressure: () => 0,
                ...(resolveOllama ? { resolveOllama } : {}),
            });
            await controller.overview();
            const added = await controller.addModel({ id: 'olla-user', sources: { ollama: { type: 'ollama', tag: 'gpt-oss:20b' } } });
            const record = { added: added.model.sources, stored: structuredClone(controller.state.registry.map((entry) => entry.sources)) };
            try {
                const accepted = await controller.run({ modelId: 'olla-user', runnerId: 'ollama', requestId: 'request-0001' });
                record.accepted = accepted.accepted;
                const deadline = Date.now() + 3000;
                while (!['ready', 'error'].includes(controller.state.deployment?.phase) && Date.now() < deadline) await new Promise((resolve) => setTimeout(resolve, 5));
                const deployment = controller.state.deployment;
                record.phase = deployment.phase;
                record.error = deployment.error;
                record.profile = deployment.profile;
                record.params = deployment.params;
                record.admission = { status: deployment.admission.status, reason: deployment.admission.reason };
                record.artifact = deployment.artifact;
                record.resolved = deployment.resolved ?? null;
                record.runner = deployment.runner ? { ollama: deployment.runner.ollama, port: deployment.runner.port } : null;
                record.process = started[0] ? { command: started[0].command, args: started[0].args, env: started[0].env } : null;
                record.calls = calls;
                record.chatTarget = deployment.phase === 'ready' ? controller.chatTarget() : null;
                await controller.stop();
            } catch (error) {
                record.refused = { code: error.code, message: error.message };
            }
            runs[name] = JSON.parse(JSON.stringify(record).split(dataDir).join('<data>'));
        } finally {
            await controller?.drain();
            fs.rmSync(dataDir, { recursive: true, force: true });
        }
    }
    return runs;
}
