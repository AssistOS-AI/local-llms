// ExLlamaV3 + TabbyAPI (runners plan §5.6, Phase R6): installed on demand
// after an admin accepts TabbyAPI's AGPL notice (DS004), reads an EXL3
// snapshot, runs from its own source directory in the runnable copy on
// loopback, and requires its per-start key.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { RUNNERS, defaultPorts } from '../src/runners/index.mjs';
import { ParamError } from '../src/runners/params.mjs';
import { tabbyApiRunner } from '../src/runners/tabbyApi.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
const SEED = loadSeedCatalog();
const QWEN8 = SEED.find((model) => model.id === 'qwen3-8b-exl3');
const RUN_DIR = '/opt/runners/tabbyapi/f07131cd8fe34e449fe87cdd3a066b52b96d3cac';
const SNAPSHOT = '/data/models/hf/turboderp/Qwen3-8B-exl3/1fd66d10f8fbdf071a0ff35842a2d1bf0df94b45';

function snapshot({ freeMiB = 6000 } = {}) {
    return {
        gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: (6144 - freeMiB) * MIB, freeBytes: freeMiB * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
        disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
    };
}

const launch = (params = {}, extra = {}) => tabbyApiRunner.buildLaunch({
    runnerDir: RUN_DIR, artifactPath: SNAPSHOT, params, port: 18083, apiKey: KEY, model: QWEN8, cacheDir: '/opt/runners/.cache/tabbyapi', ...extra,
});
const valueOf = (args, flag) => args[args.indexOf(flag) + 1];

test('the seed catalog pins Qwen3-8B EXL3 4.0 bpw as an EXL3 snapshot', () => {
    assert.ok(QWEN8);
    const source = QWEN8.sources.exl3;
    assert.equal(source.repo, 'turboderp/Qwen3-8B-exl3');
    assert.equal(source.revision, '4.0bpw');
    assert.equal(source.commit, '1fd66d10f8fbdf071a0ff35842a2d1bf0df94b45');
    assert.equal(source.files.length, 8);
    assert.equal(source.size, 5206949815);
    assert.equal(QWEN8.recommended.tabbyapi.cacheMode, 'Q4');
});

test('TabbyAPI is a supported runner that reads EXL3 snapshots on its own loopback port', () => {
    assert.equal(RUNNERS.tabbyapi, tabbyApiRunner);
    assert.equal(tabbyApiRunner.supported, true);
    assert.equal(tabbyApiRunner.weightFormat, 'exl3');
    assert.equal(tabbyApiRunner.pinnedVersion, 'f07131c');
    assert.equal(defaultPorts().tabbyapi, 18083);
    assert.equal(new Set(Object.values(defaultPorts())).size, Object.keys(defaultPorts()).length);
});

test('TabbyAPI runs from its source directory with the snapshot, loopback only, and no key in its arguments or environment', () => {
    const { command, args, env, cwd } = launch();
    assert.equal(command, `${RUN_DIR}/venv/bin/python`);
    assert.equal(args[0], `${RUN_DIR}/tabbyAPI/main.py`);
    // It resolves templates/, config.yml and api_tokens.yml from its working directory.
    assert.equal(cwd, `${RUN_DIR}/tabbyAPI`);
    assert.ok(args.every((arg) => typeof arg === 'string'));
    assert.equal(valueOf(args, '--host'), '127.0.0.1');
    assert.equal(valueOf(args, '--port'), '18083');
    assert.equal(valueOf(args, '--model-dir'), path.dirname(SNAPSHOT));
    assert.equal(valueOf(args, '--model-name'), path.basename(SNAPSHOT));
    assert.equal(valueOf(args, '--backend'), 'exllamav3');
    assert.equal(valueOf(args, '--max-seq-len'), '2048');
    assert.equal(valueOf(args, '--cache-size'), '2048');
    assert.equal(valueOf(args, '--cache-mode'), 'Q4');
    assert.equal(valueOf(args, '--chunk-size'), '512');
    assert.equal(args.join(' ').includes(KEY), false);
    assert.equal(Object.values(env).some((value) => String(value).includes(KEY)), false);
    assert.equal(env.LD_LIBRARY_PATH, '/usr/local/nvidia/lib64');
    // TabbyAPI's Triton (3.5) links -lcuda: a directory with libcuda.so pointing at the granted driver library.
    assert.equal(env.TRITON_LIBCUDA_PATH, '/opt/runners/.cache/tabbyapi/libcuda');
    assert.ok(env.TMPDIR.startsWith('/dev/shm/'));
    assert.equal(env.HF_HUB_OFFLINE, '1');
    // The key lives in api_tokens.yml, written per start; the admin key is the same per-start key.
    const tokens = tabbyApiRunner.authFile(KEY);
    assert.match(tokens, new RegExp(`^api_key: ${KEY}$`, 'm'));
    assert.match(tokens, new RegExp(`^admin_key: ${KEY}$`, 'm'));
    assert.throws(() => tabbyApiRunner.authFile('short'));
});

test('TabbyAPI parameters are validated; a cache smaller than the context is refused', () => {
    const { args } = launch({ maxSeqLen: 4096, cacheSize: 8192, cacheMode: 'Q8', chunkSize: 1024 });
    assert.equal(valueOf(args, '--max-seq-len'), '4096');
    assert.equal(valueOf(args, '--cache-size'), '8192');
    assert.equal(valueOf(args, '--cache-mode'), 'Q8');
    assert.equal(valueOf(args, '--chunk-size'), '1024');
    assert.throws(() => launch({ maxSeqLen: 4096, cacheSize: 2048 }), ParamError);
    assert.throws(() => launch({ cacheMode: 'Q2' }), ParamError);
    assert.throws(() => launch({ modelDir: '/etc' }), ParamError);
    assert.throws(() => launch({}, { runnerDir: 'relative' }));
});

test('TabbyAPI refuses to start when its port is taken, instead of moving to the next port', async (t) => {
    const blocker = net.createServer();
    await new Promise((resolve) => blocker.listen(0, '127.0.0.1', resolve));
    t.after(() => blocker.close());
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-tabby-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    // The layout of /opt/runners/<id>/<version>.
    const runnerDir = path.join(root, 'runners', 'tabbyapi', 'f07131c');
    fs.mkdirSync(path.join(runnerDir, 'tabbyAPI'), { recursive: true });
    let launched = false;
    await assert.rejects(() => tabbyApiRunner.start({
        runner: tabbyApiRunner, runnerDir, weights: { path: SNAPSHOT }, params: {}, port: blocker.address().port, apiKey: KEY, model: QWEN8, shmDir: path.join(root, 'shm'),
        launch: () => { launched = true; }, waitForHttp: async () => {},
    }), { code: 'port_busy' });
    assert.equal(launched, false);
    assert.equal(fs.existsSync(path.join(runnerDir, 'tabbyAPI', 'api_tokens.yml')), false);
});

test('TabbyAPI writes its per-start key file privately and waits until the model is loaded', async (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-tabby-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    // The layout of /opt/runners/<id>/<version>.
    const runnerDir = path.join(root, 'runners', 'tabbyapi', 'f07131c');
    fs.mkdirSync(path.join(runnerDir, 'tabbyAPI'), { recursive: true });
    const probes = [];
    const port = await new Promise((resolve) => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); }); });
    await tabbyApiRunner.start({
        runner: tabbyApiRunner, runnerDir, weights: { path: SNAPSHOT }, params: {}, port, apiKey: KEY, model: QWEN8, shmDir: path.join(root, 'shm'),
        launch: () => ({ pid: 1 }), waitForHttp: async (url, options) => { probes.push([url, options?.headers?.authorization ?? null]); },
    });
    const file = path.join(runnerDir, 'tabbyAPI', 'api_tokens.yml');
    assert.equal(fs.statSync(file).mode & 0o777, 0o600);
    assert.equal(fs.readFileSync(file, 'utf8'), tabbyApiRunner.authFile(KEY));
    const libcuda = path.join(root, 'runners', '.cache', 'tabbyapi', 'libcuda');
    for (const name of ['libcuda.so', 'libcuda.so.1']) {
        assert.equal(fs.readlinkSync(path.join(libcuda, name)), '/usr/local/nvidia/lib64/libcuda.so.1');
    }
    assert.deepEqual(probes, [[`http://127.0.0.1:${port}/health`, null], [`http://127.0.0.1:${port}/v1/model`, `Bearer ${KEY}`]]);
});

test('admission: Qwen3-8B EXL3 at 2k context with a Q4 cache fits a 6 GB GPU; a long FP16 cache does not', () => {
    const params = tabbyApiRunner.normalizeParams({}, { model: QWEN8 });
    const ok = admit({ runner: tabbyApiRunner, model: QWEN8, source: QWEN8.sources.exl3, params, snapshot: snapshot() });
    assert.equal(ok.status, 'ok', ok.reason);
    assert.equal(ok.estimate.weightsBytes, 5206949815);
    assert.equal(ok.estimate.kvBytes, QWEN8.memory.kvBytesPerToken * 2048 * 0.25);
    // ExLlamaV3 keeps the input embedding in system RAM (measured: 4,141 MiB on the GPU for this model at 2k Q4).
    assert.equal(ok.estimate.gpuWeightsBytes, 5206949815 - QWEN8.memory.embeddingBytes);
    assert.ok(ok.estimate.gpuBytes < 4.6 * GIB && ok.estimate.gpuBytes > 4.0 * GIB, String(ok.estimate.gpuBytes / GIB));
    const long = admit({ runner: tabbyApiRunner, model: QWEN8, source: QWEN8.sources.exl3,
        params: tabbyApiRunner.normalizeParams({ maxSeqLen: 32768, cacheMode: 'FP16' }, { model: QWEN8 }), snapshot: snapshot() });
    assert.equal(long.status, 'incompatible');
    assert.match(long.reason, /cacheMode|maxSeqLen/);
    const busy = admit({ runner: tabbyApiRunner, model: QWEN8, source: QWEN8.sources.exl3, params, snapshot: snapshot({ freeMiB: 3000 }) });
    assert.equal(busy.status, 'insufficient-now');
});

test('admission: TabbyAPI checks system RAM, embedding included, like the other runners', () => {
    const params = tabbyApiRunner.normalizeParams({}, { model: QWEN8 });
    const lowRam = (totalGiB, availableGiB) => ({ ...snapshot(), memory: { totalBytes: totalGiB * GIB, availableBytes: availableGiB * GIB } });
    const never = admit({ runner: tabbyApiRunner, model: QWEN8, source: QWEN8.sources.exl3, params, snapshot: lowRam(2, 1) });
    assert.equal(never.status, 'incompatible');
    assert.match(never.reason, /RAM/);
    const busy = admit({ runner: tabbyApiRunner, model: QWEN8, source: QWEN8.sources.exl3, params, snapshot: lowRam(31, 2) });
    assert.equal(busy.status, 'insufficient-now');
    assert.match(busy.reason, /RAM/);
    assert.equal(busy.estimate.ramBytes, 3 * GIB + QWEN8.memory.embeddingBytes);
});
