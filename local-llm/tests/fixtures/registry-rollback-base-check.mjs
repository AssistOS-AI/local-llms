// Rollback rehearsal (A33, DS002). Runs over the local-llms base 7fdbff5 (`git archive`, then `node <this file>`)
// with only this file added, so it imports nothing the base does not have. A newer agent ran an OCI model last, so
// its controller.json holds an idle deployment whose artifact is an `oci` source, which the base has no store for.
// The base's Delete weights of a supported model then fails with invalid_model (the documented hazard); the operator
// step, a Run of any huggingface.co model, replaces that deployment, and Delete weights works again.
// Not a test file: it prints one line per step and exits 0 only when every step did what DS002 says.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { loadSeedCatalog } from '../../src/controller/catalog.mjs';
import { createController } from '../../src/controller/deployments.mjs';
import { createStateStore } from '../../src/controller/stateStore.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const RUN = Object.freeze({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' });
// Docker Hub's ai/smollm2:135m-q4_K_M, pinned as the newer agent pins it (CNCF layout, the layer digest is the file's sha256).
const OCI_SOURCE = Object.freeze({
    type: 'oci',
    registry: 'dockerhub',
    repository: 'ai/smollm2',
    tag: '135m-q4_K_M',
    manifestDigest: 'sha256:eb3483481647229668c7625b080c2d6a632bf49535e0714198de5c026d4f41a5',
    layout: 'cncf',
    file: 'SmolLM2-135M-Instruct-Q4_K_M.gguf',
    size: 105454144,
    sha256: 'ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747',
});

function snapshot() {
    return {
        gpu: { available: true, name: 'Test GPU', driverVersion: '595.91.07', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
        disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
        cpus: 20,
    };
}

function startRunner({ command, args, log }) {
    let running = true;
    let finish;
    const exited = new Promise((resolve) => { finish = resolve; });
    log.append('stdout', 'load_tensors: offloaded 25/25 layers to GPU');
    return {
        pid: 4242, command, args, exited,
        get running() { return running; },
        async stop() {
            if (running) { running = false; finish({ code: 0, signal: 'SIGTERM', error: null }); }
            return exited;
        },
    };
}

async function until(predicate, timeoutMs = 10_000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (predicate()) return true;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    return false;
}

const outcome = async (promise) => {
    try {
        await promise;
        return 'ok';
    } catch (error) {
        return error?.code ?? `error ${error?.message}`;
    }
};

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-a33-'));
let failed = false;
let controller = null;
try {
    // controller.json as the newer agent leaves it after an OCI Run was stopped: the deployment is idle.
    const at = new Date().toISOString();
    fs.mkdirSync(path.join(dataDir, 'state'), { recursive: true, mode: 0o700 });
    fs.writeFileSync(path.join(dataDir, 'state', 'controller.json'), `${JSON.stringify({
        version: 1,
        deployment: {
            id: 'oci-deployment', requestId: 'request-oci-0001', modelId: 'smollm2-oci', runnerId: 'llama.cpp', profile: 'dedicated', params: {},
            artifact: OCI_SOURCE, phase: 'idle', admission: null, acquisition: [],
            download: { bytes: OCI_SOURCE.size, total: OCI_SOURCE.size, rate: 0, etaSeconds: null, transferred: OCI_SOURCE.size },
            error: null, pausedReason: null, runner: null, logSeqStart: 0, createdAt: at, updatedAt: at,
        },
        params: {},
        requests: {},
        registry: [{ id: 'smollm2-oci', displayName: 'SmolLM2 135M (OCI)', contextLength: 8192, sources: { gguf: OCI_SOURCE } }],
    }, null, 2)}\n`, { mode: 0o600 });
    controller = createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: loadSeedCatalog(),
        stateStore: createStateStore({ dataDir }),
        snapshot: async () => snapshot(),
        download: async () => ({ status: 'complete', path: path.join(dataDir, 'gpt-oss-20b.gguf'), bytesTransferred: 0 }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        startRunner,
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        pollMs: 2,
        stopGraceMs: 50,
    });
    const before = await outcome(controller.deleteWeights(RUN));
    console.log(`deleteWeights-before: ${before}`);
    if (before !== 'invalid_model') failed = true;
    // The operator step: Run any huggingface.co model once (its download is stubbed here).
    await controller.run({ ...RUN, requestId: 'request-hf-0001' });
    const ready = await until(() => ['ready', 'error'].includes(controller.state.deployment?.phase));
    console.log(`run: ${controller.state.deployment?.phase}${controller.state.deployment?.error ? ` (${controller.state.deployment.error})` : ''}`);
    if (!ready || controller.state.deployment?.phase !== 'ready' || controller.state.deployment.artifact?.type !== 'huggingface') failed = true;
    await controller.stop();
    const after = await outcome(controller.deleteWeights(RUN));
    console.log(`deleteWeights-after: ${after}`);
    if (after !== 'ok') failed = true;
} catch (error) {
    console.log(`unexpected: ${error?.code ?? ''} ${error?.message}`);
    failed = true;
} finally {
    try { await controller?.drain(); } catch { /* the outcome is already printed */ }
    fs.rmSync(dataDir, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
