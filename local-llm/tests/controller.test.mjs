import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { DownloadError } from '../src/controller/downloader.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog();
const GPT = SEED[0];
const RUN = { modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' };

function snapshot({ gpuFree = 6000 * MIB, available = 24 * GIB, processes = [] } = {}) {
    return {
        gpu: {
            available: true, name: 'Test GPU', driverVersion: '595.91.07',
            totalBytes: 6144 * MIB, usedBytes: 6144 * MIB - gpuFree, freeBytes: gpuFree, processes,
        },
        memory: { totalBytes: 31 * GIB, availableBytes: available },
        disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
        cpus: 20,
    };
}

function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
    return { promise, resolve, reject };
}

function fakeRunnerFactory({ quietAfterFirst = false } = {}) {
    const started = [];
    return {
        started,
        startRunner({ command, args, env, log }) {
            const exit = deferred();
            let running = true;
            const handle = {
                pid: 4242 + started.length,
                command,
                args,
                env,
                exited: exit.promise,
                get running() { return running; },
                async stop() {
                    if (running) {
                        running = false;
                        handle.stopped = true;
                        exit.resolve({ code: 0, signal: 'SIGTERM', error: null });
                    }
                    return exit.promise;
                },
                crash(code = 1) {
                    running = false;
                    exit.resolve({ code, signal: null, error: null });
                },
            };
            if (!(quietAfterFirst && started.length > 0)) {
                log.append('stdout', 'load_tensors: offloaded 25/25 layers to GPU');
                log.append('stdout', 'load_tensors:        CUDA0 model buffer size =  4073.34 MiB');
            }
            started.push(handle);
            return handle;
        },
    };
}

function harness(t, {
    stateFile = null,
    snap = snapshot(),
    downloads = [],
    initialState = null,
    extraSeed = [],
    quietAfterFirst = false,
    resolveHf = null,
    fetchImpl = async () => ({ ok: true, status: 200, json: async () => ({}) }),
} = {}) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-controller-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const stateStore = createStateStore({ dataDir });
    if (initialState) stateStore.save(initialState);
    const runners = fakeRunnerFactory({ quietAfterFirst });
    const calls = { download: [], remove: [] };
    let inspectImpl = async () => ({ state: 'absent', bytes: 0 });
    const controller = createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: [...SEED, ...extraSeed],
        stateStore,
        snapshot: async () => (typeof snap === 'function' ? snap() : snap),
        download: ({ artifact, signal, onProgress }) => {
            const entry = deferred();
            calls.download.push({ artifact, signal, entry, onProgress });
            signal?.addEventListener('abort', () => {
                entry.reject(new DownloadError('ABORTED', 'aborted'));
            }, { once: true });
            const planned = downloads.shift();
            if (planned === 'complete') entry.resolve({ status: 'complete', path: '/data/models/x.gguf', bytesTransferred: 0 });
            return entry.promise;
        },
        inspect: (options) => inspectImpl(options),
        remove: async (options) => { calls.remove.push(options); return 123; },
        startRunner: runners.startRunner,
        fetchImpl,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        pollMs: 2,
        stopGraceMs: 50,
        ...(resolveHf ? { resolveHf } : {}),
    });
    return { controller, runners, calls, dataDir, stateStore, setInspect: (next) => { inspectImpl = next; } };
}

async function until(predicate, timeoutMs = 2000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

const phase = (h) => h.controller.state.deployment?.phase;

test('run downloads, verifies admission again, starts the runner and becomes ready', async (t) => {
    const h = harness(t);
    const accepted = await h.controller.run({ ...RUN, requestId: 'request-0001' });
    assert.equal(accepted.accepted, true);
    assert.equal(accepted.deployment.phase, 'downloading');
    assert.equal(accepted.deployment.params.nCpuMoe, 17);
    assert.equal(h.calls.download.length, 1);
    assert.equal(h.calls.download[0].artifact.sha256, GPT.sources['llama.cpp'].sha256);
    h.calls.download[0].entry.resolve({ status: 'complete', path: '/data/models/gpt.gguf', bytesTransferred: 12 });
    await until(() => phase(h) === 'ready');
    const [process] = h.runners.started;
    assert.equal(process.command, '/opt/llama.cpp/llama-server');
    assert.ok(process.args.includes('--api-key'));
    assert.equal(process.args[process.args.indexOf('--host') + 1], '127.0.0.1');
    assert.equal(process.env.HF_TOKEN, undefined);
    const status = await h.controller.status({ sinceSeq: 0 });
    assert.equal(status.phase, 'ready');
    assert.equal(status.runnerReport.modelMiB, 4073.34);
    assert.deepEqual(status.runnerReport.offloaded, { layers: 25, of: 25 });
    // The per-start key never leaves the controller except to the responder.
    assert.equal(JSON.stringify(status).includes(process.args[process.args.indexOf('--api-key') + 1]), false);
    const target = h.controller.chatTarget();
    assert.equal(target.apiKey, process.args[process.args.indexOf('--api-key') + 1]);
});

test('a repeated requestId is a no-op and a second run while busy needs replace', async (t) => {
    const h = harness(t, { downloads: ['complete'] });
    await h.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => phase(h) === 'ready');
    const again = await h.controller.run({ ...RUN, requestId: 'request-0001' });
    assert.equal(again.duplicate, true);
    assert.equal(h.calls.download.length, 1);
    await assert.rejects(() => h.controller.run({ ...RUN, requestId: 'request-0002' }), { code: 'busy' });
    await h.controller.run({ ...RUN, requestId: 'request-0003', replace: true, params: { ctxSize: 8192 } });
    assert.equal(h.runners.started[0].stopped, true);
    assert.equal(h.controller.state.deployment.params.ctxSize, 8192);
});

test('cancel keeps a paused, resumable download; stop ends idle', async (t) => {
    const h = harness(t);
    await h.controller.run({ ...RUN, requestId: 'request-0001' });
    const result = await h.controller.cancelDownload();
    assert.equal(result.deployment.phase, 'paused');
    assert.equal(h.calls.download[0].signal.aborted, true);
    await assert.rejects(() => h.controller.cancelDownload(), { code: 'not_downloading' });
    await h.controller.run({ ...RUN, requestId: 'request-0002' });
    assert.equal(h.calls.download.length, 2);
    await h.controller.stop();
    assert.equal(phase(h), 'idle');
});

test('commands are serialized and weights or entries in use cannot change', async (t) => {
    const user = {
        id: 'tiny-user',
        displayName: 'Tiny',
        sources: {
            'llama.cpp': {
                type: 'huggingface', repo: 'test/tiny', file: 'tiny.gguf', revision: 'main',
                commit: 'a'.repeat(40), size: 1000, sha256: 'b'.repeat(64),
            },
        },
    };
    const h = harness(t, { initialState: { version: 1, deployment: null, params: {}, requests: {}, registry: [user] } });
    const order = [];
    const run = h.controller.run({ modelId: 'tiny-user', runnerId: 'llama.cpp', requestId: 'request-0001' })
        .then(() => order.push('run'));
    const remove = h.controller.deleteWeights({ modelId: 'tiny-user', runnerId: 'llama.cpp' })
        .then(() => order.push('delete-ok'), (error) => order.push(`delete-${error.code}`));
    const update = h.controller.updateModel({ ...user, displayName: 'Renamed' })
        .then(() => order.push('update-ok'), (error) => order.push(`update-${error.code}`));
    await Promise.all([run, remove, update]);
    assert.deepEqual(order, ['run', 'delete-in_use', 'update-in_use']);
    // The job holds its own copy: an edit of the stored entry cannot redirect it.
    h.controller.state.registry[0].sources['llama.cpp'].file = 'other.gguf';
    assert.equal(h.controller.state.deployment.artifact.file, 'tiny.gguf');
    assert.equal(h.calls.download[0].artifact.file, 'tiny.gguf');
    await h.controller.cancelDownload();
    const deleted = await h.controller.deleteWeights({ modelId: 'tiny-user', runnerId: 'llama.cpp' });
    assert.equal(deleted.freedBytes, 123);
    await assert.rejects(() => h.controller.removeModel({ modelId: 'gpt-oss-20b' }), { code: 'read_only' });
});

test('after a restart a download is paused without resuming and a ready model is idle', async (t) => {
    for (const [before, after] of [['downloading', 'paused'], ['verifying', 'paused'], ['ready', 'idle'], ['starting', 'idle']]) {
        const h = harness(t, {
            initialState: {
                version: 1,
                deployment: { id: 'd1', requestId: 'r1', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', phase: before,
                    params: {}, artifact: GPT.sources['llama.cpp'], runner: { pid: 1 } },
                params: {},
                requests: {},
                registry: [],
            },
        });
        assert.equal(phase(h), after, before);
        assert.equal(h.calls.download.length, 0, 'nothing resumes until the next explicit Run');
        assert.equal(JSON.parse(fs.readFileSync(h.stateStore.file, 'utf8')).deployment.phase, after);
    }
});

test('a runner crash puts the deployment in error', async (t) => {
    const h = harness(t, { downloads: ['complete'] });
    await h.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => phase(h) === 'ready');
    h.runners.started[0].crash(139);
    await until(() => phase(h) === 'error');
    assert.match(h.controller.state.deployment.error, /exited unexpectedly \(code 139/);
    assert.throws(() => h.controller.chatTarget(), { code: 'not_ready' });
});

test('drain checkpoints a download as paused and reaps a running runner', async (t) => {
    const downloading = harness(t);
    await downloading.controller.run({ ...RUN, requestId: 'request-0001' });
    await downloading.controller.drain();
    assert.equal(downloading.calls.download[0].signal.aborted, true);
    assert.equal(JSON.parse(fs.readFileSync(downloading.stateStore.file, 'utf8')).deployment.phase, 'paused');
    await assert.rejects(() => downloading.controller.run({ ...RUN, requestId: 'request-0002' }), { code: 'shutting_down' });

    const ready = harness(t, { downloads: ['complete'] });
    await ready.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => phase(ready) === 'ready');
    await ready.controller.drain();
    assert.equal(ready.runners.started[0].stopped, true);
    assert.equal(JSON.parse(fs.readFileSync(ready.stateStore.file, 'utf8')).deployment.phase, 'idle');
});

test('admission refuses before anything is downloaded, and re-checks before launch', async (t) => {
    const h = harness(t);
    await assert.rejects(
        () => h.controller.run({ ...RUN, requestId: 'request-0001', params: { ctxSize: 131072, nCpuMoe: 0 } }),
        (error) => error.code === 'admission_incompatible' && /GPU memory/.test(error.message),
    );
    await assert.rejects(
        () => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'vllm', requestId: 'request-0002' }),
        (error) => error.code === 'runner_unsupported' && /vLLM cannot load MXFP4 GGUF; not supported in this release/.test(error.message),
    );
    assert.equal(h.calls.download.length, 0);

    let busy = false;
    const late = harness(t, { snap: () => (busy ? snapshot({ gpuFree: 1000 * MIB }) : snapshot()) });
    await late.controller.run({ ...RUN, requestId: 'request-0003' });
    busy = true;
    late.calls.download[0].entry.resolve({ status: 'complete', path: '/data/models/gpt.gguf', bytesTransferred: 0 });
    await until(() => phase(late) === 'error');
    assert.match(late.controller.state.deployment.error, /free now/);
    assert.equal(late.runners.started.length, 0);
});

test('overview lists runners, per-runner sizes, download state and admission', async (t) => {
    const h = harness(t);
    const overview = await h.controller.overview();
    assert.deepEqual(overview.runners.map((runner) => runner.id), ['llama.cpp', 'ollama', 'vllm', 'lmstudio']);
    const gpt = overview.models.find((model) => model.id === 'gpt-oss-20b');
    assert.equal(gpt.runners['llama.cpp'].size, 12109566624);
    assert.equal(gpt.runners.ollama.size, 13793441244);
    assert.equal(gpt.runners['llama.cpp'].admission.status, 'ok');
    assert.equal(gpt.runners['llama.cpp'].admission.estimate.isEstimate, true);
    assert.equal(gpt.runners.vllm.admission.status, 'incompatible');
    assert.equal(gpt.runners.lmstudio.admission.status, 'incompatible');
    assert.equal(gpt.runners['llama.cpp'].context.totalContext, 16384);
});

test('completion speed is recorded only while a model is ready and is shown in status', async (t) => {
    const h = harness(t, { downloads: ['complete'] });
    assert.deepEqual(h.controller.recordCompletion({ generationTokensPerSecond: 50 }), { recorded: false });
    await h.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => phase(h) === 'ready');
    const stats = { promptTokens: 12, completionTokens: 34, promptTokensPerSecond: 400.5, generationTokensPerSecond: 58.2, source: 'runner timings' };
    assert.deepEqual(h.controller.recordCompletion({ ...stats, apiKey: 'ignored', promptTokens: -1 }), { recorded: true });
    const { lastCompletion, logs } = await h.controller.status({ sinceSeq: 0 });
    assert.equal(lastCompletion.modelId, 'gpt-oss-20b');
    assert.equal(lastCompletion.runnerId, 'llama.cpp');
    assert.equal(lastCompletion.promptTokens, null);
    assert.equal(lastCompletion.generationTokensPerSecond, 58.2);
    assert.equal(lastCompletion.apiKey, undefined);
    assert.ok(logs.some((entry) => /completion served: 34 tokens at 58\.2 tokens\/s/.test(entry.line)));
});

test('overview previews admission for Run form values without saving or downloading', async (t) => {
    const h = harness(t);
    const plain = await h.controller.overview();
    assert.equal(plain.preview, undefined);
    assert.equal(plain.gatewayModel, 'soul_gateway/local-llms/local-llm/default');
    const fits = (await h.controller.overview({ preview: { modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', params: { ctxSize: 16384, nCpuMoe: 17 } } })).preview;
    assert.equal(fits.admission.status, 'ok');
    assert.equal(fits.params.nCpuMoe, 17);
    assert.equal(fits.context.totalContext, 16384);
    const tooBig = (await h.controller.overview({ preview: { modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', params: { ctxSize: 131072, nCpuMoe: 0 } } })).preview;
    assert.equal(tooBig.admission.status, 'incompatible');
    const invalid = (await h.controller.overview({ preview: { modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', params: { ctxSize: 'x' } } })).preview;
    assert.match(invalid.error, /ctxSize/);
    const vllm = (await h.controller.overview({ preview: { modelId: 'gpt-oss-20b', runnerId: 'vllm' } })).preview;
    assert.equal(vllm.admission.status, 'incompatible');
    await assert.rejects(() => h.controller.overview({ preview: { modelId: 'nope-model', runnerId: 'llama.cpp' } }), { code: 'unknown_model' });
    assert.equal(h.calls.download.length, 0);
    assert.deepEqual(h.controller.state.params, {});
});

test('stop clears a failed deployment', async (t) => {
    const h = harness(t, { downloads: ['complete'] });
    await h.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => phase(h) === 'ready');
    h.runners.started[0].crash(1);
    await until(() => phase(h) === 'error');
    await h.controller.stop();
    assert.equal(phase(h), 'idle');
    assert.equal(h.controller.state.deployment.error, null);
});

test('a drain waits for a Run already in the queue, which then refuses to start a job', async (t) => {
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    let snapshots = 0;
    const h = harness(t, {
        downloads: ['complete'],
        snap: async () => {
            snapshots += 1;
            if (snapshots === 1) await gate;
            return snapshot();
        },
    });
    const running = h.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => snapshots === 1);
    const drained = h.controller.drain();
    release();
    await assert.rejects(running, { code: 'shutting_down' });
    await drained;
    assert.equal(h.calls.download.length, 0);
    assert.equal(h.controller.state.deployment, null);
    await assert.rejects(() => h.controller.run({ ...RUN, requestId: 'request-0002' }), { code: 'shutting_down' });
});

test('the runner report and speed describe only the current runner start', async (t) => {
    const h = harness(t, { downloads: ['complete', 'complete'], quietAfterFirst: true });
    await h.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => phase(h) === 'ready');
    h.controller.recordCompletion({ generationTokensPerSecond: 36, completionTokens: 10, source: 'runner timings' });
    let status = await h.controller.status({ sinceSeq: 0 });
    assert.deepEqual(status.runnerReport.offloaded, { layers: 25, of: 25 });
    assert.equal(status.lastCompletion.generationTokensPerSecond, 36);
    await h.controller.run({ ...RUN, requestId: 'request-0002', replace: true });
    await until(() => phase(h) === 'ready' && h.runners.started.length === 2);
    status = await h.controller.status({ sinceSeq: 0 });
    assert.equal(status.runnerReport.offloaded, null);
    assert.equal(status.runnerReport.modelMiB, null);
    assert.equal(status.lastCompletion, null);
});

test('a hanging Hugging Face lookup for Add model does not hold up Stop', async (t) => {
    const h = harness(t, { resolveHf: () => new Promise(() => {}) });
    const adding = h.controller.addModel({
        id: 'user-qwen',
        sources: { 'llama.cpp': { type: 'huggingface', repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', revision: 'main' } },
    });
    adding.catch(() => {});
    const stopped = await Promise.race([
        h.controller.stop().then(() => 'stopped'),
        new Promise((resolve) => setTimeout(() => resolve('still waiting'), 500)),
    ]);
    assert.equal(stopped, 'stopped');
    // A request that comes back after a drain started is refused, not stored.
    assert.equal(h.controller.state.registry.length, 0);
});

const PARTIAL_D = `sha256-${'d'.repeat(64)}-partial`;
const PARTIAL_E = `sha256-${'e'.repeat(64)}-partial`;

function writePartial(h, name, bytes) {
    const blobs = path.join(h.dataDir, 'models', 'ollama', 'blobs');
    fs.mkdirSync(blobs, { recursive: true });
    fs.writeFileSync(path.join(blobs, name), Buffer.alloc(bytes));
    return path.join(blobs, name);
}

test('another tag\'s partial Ollama pull neither counts for nor blocks removing a user Ollama model', async (t) => {
    const user = { id: 'user-olla', sources: { ollama: { type: 'ollama', tag: 'qwen3:0.6b' } } };
    const h = harness(t, { initialState: {
        version: 1, deployment: null, params: {}, requests: {}, registry: [user],
        ollamaPulls: { 'gpt-oss:20b': [`sha256:${'d'.repeat(64)}`] },
    } });
    writePartial(h, PARTIAL_D, 4096);
    const overview = await h.controller.overview();
    assert.equal(overview.models.find((model) => model.id === 'user-olla').runners.ollama.download.state, 'absent');
    assert.deepEqual(overview.models.find((model) => model.id === 'gpt-oss-20b').runners.ollama.download,
        { state: 'partial', bytes: 4096, total: GPT.sources.ollama.size });
    await h.controller.removeModel({ modelId: 'user-olla' });
    assert.equal(h.controller.state.registry.length, 0);
});

test('deleting a tag\'s Ollama weights frees its paused partial and orphans, and leaves other tags\' partials', async (t) => {
    const h = harness(t, { initialState: {
        version: 1, deployment: null, params: {}, requests: {}, registry: [],
        ollamaPulls: { 'gpt-oss:20b': [`sha256:${'e'.repeat(64)}`], 'granite4:tiny-h': [`sha256:${'d'.repeat(64)}`] },
    } });
    const own = writePartial(h, PARTIAL_E, 4096);
    const ownParts = writePartial(h, `${PARTIAL_E}-0`, 10);
    const other = writePartial(h, PARTIAL_D, 2048);
    const orphan = writePartial(h, `sha256-${'f'.repeat(64)}-partial`, 1024);
    const deleted = await h.controller.deleteWeights({ modelId: 'gpt-oss-20b', runnerId: 'ollama' });
    assert.equal(deleted.freedBytes, 4096 + 10 + 1024);
    assert.equal(fs.existsSync(own) || fs.existsSync(ownParts) || fs.existsSync(orphan), false);
    assert.equal(fs.existsSync(other), true);
    assert.deepEqual(Object.keys(h.controller.state.ollamaPulls), ['granite4:tiny-h']);
});

test('an Ollama pull records the blob digests it touches', async (t) => {
    const digest = `sha256:${'a'.repeat(64)}`;
    const events = [
        { status: 'pulling manifest' },
        { status: `pulling ${digest.slice(7, 19)}`, digest, total: 1000, completed: 100 },
        { status: 'pulling x', digest: 'sha256:../../etc/passwd', total: 1, completed: 0 },
        { error: 'connection reset' },
    ];
    const body = [new TextEncoder().encode(events.map((event) => JSON.stringify(event)).join('\n') + '\n')];
    const h = harness(t, {
        fetchImpl: async (url) => (String(url).endsWith('/api/pull')
            ? { ok: true, status: 200, body }
            : { ok: true, status: 200, json: async () => ({}) }),
    });
    await h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'ollama', requestId: 'request-0001' });
    await until(() => phase(h) === 'error');
    assert.deepEqual(h.controller.state.ollamaPulls, { 'gpt-oss:20b': [digest] });
});

test('updating a user model keeps its pin, and a re-pin that would orphan downloaded weights is refused', async (t) => {
    const commits = ['a'.repeat(40), 'b'.repeat(40)];
    let resolves = 0;
    const h = harness(t, {
        resolveHf: async ({ revision }) => {
            resolves += 1;
            return { commit: revision === 'v2' ? commits[1] : commits[0], size: 4, sha256: 'c'.repeat(64) };
        },
    });
    const source = { type: 'huggingface', repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', revision: 'main' };
    await h.controller.addModel({ id: 'user-qwen', displayName: 'Q', sources: { 'llama.cpp': source } });
    assert.equal(resolves, 1);
    // A metadata edit keeps the pinned commit instead of resolving main again.
    await h.controller.updateModel({ id: 'user-qwen', displayName: 'Q2', sources: { 'llama.cpp': source } });
    assert.equal(resolves, 1);
    assert.equal(h.controller.state.registry[0].sources['llama.cpp'].commit, commits[0]);
    assert.equal(h.controller.state.registry[0].displayName, 'Q2');
    // With the pinned weights on disk, a changed source would orphan them.
    h.setInspect(async ({ artifact }) => (artifact.commit === commits[0] ? { state: 'complete', bytes: 4 } : { state: 'absent', bytes: 0 }));
    await assert.rejects(
        () => h.controller.updateModel({ id: 'user-qwen', sources: { 'llama.cpp': { ...source, revision: 'v2' } } }),
        { code: 'weights_present' },
    );
    assert.equal(h.controller.state.registry[0].sources['llama.cpp'].commit, commits[0]);
    h.setInspect(async () => ({ state: 'absent', bytes: 0 }));
    await h.controller.updateModel({ id: 'user-qwen', sources: { 'llama.cpp': { ...source, revision: 'v2' } } });
    assert.equal(h.controller.state.registry[0].sources['llama.cpp'].commit, commits[1]);
});
