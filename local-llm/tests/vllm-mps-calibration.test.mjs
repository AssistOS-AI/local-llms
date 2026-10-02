// LIVE-L3 for vLLM under a Ploinky MPS GPU share (Ploinky hardware-limits plan
// section 12.2): step 0, the prerequisite check, and stage 1, the calibration,
// of tools/vllm_mps_calibration.mjs. Every GPU, process and file here is a fake:
// nothing starts a process, opens a device or touches the network. What the
// tests pin is that the tuple, the intended utilization and the final argv come
// from production's own functions, that a missing prerequisite is a blocker and
// never a pass, that the reviewed entry rendered from stage-1 evidence is
// accepted by production for exactly its tuple, and that stage 1 changes nothing.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';

import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { createVllmMpsQualificationResolver, resolveVllmMpsQualification, vllmRunnerLockDigest } from '../src/controller/vllmMpsQualification.mjs';
import { vllmMpsTuple, vllmRunner } from '../src/runners/vllm.mjs';
import {
    CALIBRATION_SCHEMA, CTYPES_QUERY, DIST_QUERY, TORCH_QUERY, TUPLE_FIELDS, applyQualificationEntry, calibrationReport, canonicalJson, classifyDenominator, compareIdentities, evidenceDigest,
    REVIEWED_SIZING, main, prerequisiteReport, renderQualificationEntry, scanSizingSource, sizingVerdict, wheelPythonTag,
} from '../tools/vllm_mps_calibration.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SHA = (letter) => letter.repeat(64);
const url = (name) => `https://files.pythonhosted.org/packages/${name}`;
const WHEELS = [
    ['vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl', 314_883_777, '1'],
    ['torch-2.13.0-cp313-cp313-manylinux_2_28_x86_64.whl', 526_602_329, '2'],
    ['nvidia_cuda_nvrtc-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl', 90_215_200, '3'],
    ['nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl', 423_138_758, '4'],
    ['triton-3.7.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl', 197_729_241, '5'],
    ['tokenspeed_triton-3.8.10.post20260920-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl', 99_900_581, '6'],
    ['six-1.17.0-py2.py3-none-any.whl', 11_000, '7'],
];
function lockDocument({ wheels = WHEELS, kind = 'python', requiresAcceptance = false } = {}) {
    return {
        schema: 'local-llm.runners-lock/v1',
        runners: {
            vllm: {
                version: '0.30.0', kind,
                licence: { name: 'Apache-2.0', url: 'https://github.com/vllm-project/vllm/blob/v0.30.0/LICENSE', requiresAcceptance },
                files: wheels.map(([name, size, letter]) => ({ name, url: url(name), size, sha256: SHA(letter) })),
                check: { distributions: { vllm: '0.30.0', torch: '2.13.0', triton: '3.7.1' }, imports: ['torch'] },
            },
        },
    };
}
const LOCK = validateRunnerLock(lockDocument());
const ENTRY = LOCK.runners.vllm;
const PINS = Object.freeze({ version: '0.30.0', runnerLockDigest: vllmRunnerLockDigest(ENTRY), files: ENTRY.files.length, downloadBytes: ENTRY.totalBytes });

const RTX = Object.freeze({
    available: true, name: 'NVIDIA GeForce RTX 3060 Laptop GPU', driverVersion: '595.91.07', memoryModel: 'dedicated',
    totalBytes: 6144 * MIB, usedBytes: 13 * MIB, freeBytes: 6131 * MIB, processes: [],
    device: { pciDeviceId: '0x252010DE', computeCapability: '8.6', addressingMode: 'None' },
});
const MPS_ENV = Object.freeze({ CUDA_MPS_PIPE_DIRECTORY: '/run/ploinky-mps-pipe', CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: '100', CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '0=5529M' });
const USABLE = Math.floor(6144 * MIB * 0.94);

function scratch(t) {
    const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-cal-')));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    return root;
}

// An installed vLLM under <root>/vllm/0.30.0 with a venv, its package source and its ready marker.
// `sizing`: true writes the reviewed statements, false writes no sizing file, or an object { worker, utils } gives the two files' text.
function installVllm(root, { sizing = true, version = '0.30.0', ready = { digest: ENTRY.digest }, extraFiles = 0 } = {}) {
    const runner = path.join(root, 'vllm', '0.30.0');
    const site = path.join(runner, 'venv', 'lib', 'python3.13', 'site-packages');
    fs.mkdirSync(path.join(runner, 'venv', 'bin'), { recursive: true });
    fs.writeFileSync(path.join(runner, 'venv', 'bin', 'python'), '#!/bin/sh\n', { mode: 0o755 });
    fs.mkdirSync(path.join(site, 'vllm', 'v1', 'worker'), { recursive: true });
    fs.mkdirSync(path.join(site, 'vllm', 'utils'), { recursive: true });
    fs.mkdirSync(path.join(site, `vllm-${version}.dist-info`), { recursive: true });
    if (version !== null) fs.writeFileSync(path.join(site, `vllm-${version}.dist-info`, 'METADATA'), `Metadata-Version: 2.4\nName: vllm\nVersion: ${version}\n`);
    if (sizing) {
        const text = typeof sizing === 'object' ? sizing : {
            worker: 'def request_memory(snapshot, cache_config):\n    requested_memory = snapshot.total_memory * cache_config.gpu_memory_utilization\n    return requested_memory\n',
            utils: 'class MemorySnapshot:\n    def measure(self):\n        self.free_memory, self.total_memory = torch.cuda.mem_get_info()\n',
        };
        fs.writeFileSync(path.join(site, 'vllm', 'v1', 'worker', 'gpu_worker.py'), text.worker);
        fs.writeFileSync(path.join(site, 'vllm', 'utils', 'mem_utils.py'), text.utils);
    }
    for (let index = 0; index < extraFiles; index += 1) fs.writeFileSync(path.join(site, 'vllm', 'utils', `zz_extra_${String(index).padStart(3, '0')}.py`), '# nothing\n');
    if (ready !== null) fs.writeFileSync(path.join(runner, '.ready.json'), JSON.stringify(ready));
    return runner;
}

// The in-client queries, scripted: torch and ctypes see `views(env)`; a different pinned limit is a different env.
const LOCKED_DISTRIBUTIONS = Object.freeze({ vllm: '0.30.0', torch: '2.13.0', triton: '3.7.1' });
function fakeRun({ views, torch = {}, fail = null, calls = [], distributions = LOCKED_DISTRIBUTIONS }) {
    return async (file, args, options = {}) => {
        calls.push({ file, args, env: options.env, timeoutMs: options.timeoutMs });
        const script = args[1];
        const limit = options.env?.CUDA_MPS_PINNED_DEVICE_MEM_LIMIT;
        if (fail === script) return { ok: false, status: 1, stdout: '', stderr: 'Traceback: no CUDA', error: '1' };
        const total = views(limit);
        if (script === TORCH_QUERY) {
            return {
                ok: true, status: 0, stderr: '', stdout: `${JSON.stringify({
                    python: '3.13.5', torch: '2.13.0', torchCuda: '13.0', cudaAvailable: true, memGetInfo: { free: total - 200 * MIB, total }, totalMemory: total, name: RTX.name,
                    capability: [8, 6], multiProcessorCount: 30, archList: ['sm_75', 'sm_80', 'sm_86', 'sm_90'], mps: { CUDA_MPS_PIPE_DIRECTORY: options.env.CUDA_MPS_PIPE_DIRECTORY, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: options.env.CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: limit },
                    ...torch,
                })}\n`,
            };
        }
        if (script === CTYPES_QUERY) return { ok: true, status: 0, stderr: '', stdout: `${JSON.stringify({ free: total - 150 * MIB, total })}\n` };
        if (script === DIST_QUERY) return { ok: true, status: 0, stderr: '', stdout: `${JSON.stringify(Object.fromEntries(args.slice(2).map((name) => [name, distributions[name] ?? null])))}\n` };
        return { ok: false, status: 1, stdout: '', stderr: 'unexpected query', error: '1' };
    };
}
const physicalViews = () => USABLE;
const shareViews = (limit) => Number(/^0=(\d+)M$/.exec(limit)[1]) * MIB;

async function calibrate(t, options = {}) {
    const root = scratch(t);
    installVllm(root, options.install);
    const calls = [];
    const report = await calibrationReport({
        env: { ...MPS_ENV, ...(options.env || {}) }, runRoot: root, readGpuImpl: async () => structuredClone(options.gpu || RTX), readMemoryImpl: () => ({ totalBytes: 31 * GIB, availableBytes: 24 * GIB }),
        locksImpl: () => LOCK, run: fakeRun({ views: options.views || physicalViews, torch: options.torch, fail: options.fail, calls, ...(options.distributions ? { distributions: options.distributions } : {}) }),
        hostNvmlBytes: options.hostNvmlBytes ?? 6144 * MIB, now: () => new Date('2026-10-03T10:00:00Z'), ...(options.fsApi ? { fsApi: options.fsApi } : {}),
    });
    return { report, root, calls };
}

// --- Step 0 -----------------------------------------------------------------------------------
function prerequisiteWorld(t, { lock = lockDocument(), python = '3 13 /usr/include/python3.13', tools = true, free = 400 * GIB, dev = '1', gpu = RTX, pins = PINS, arch = 'x64', gpuThrows = false } = {}) {
    const root = scratch(t);
    const lockFile = path.join(root, 'runners.lock.json');
    fs.writeFileSync(lockFile, JSON.stringify(lock));
    const real = fs;
    const fsApi = {
        ...real,
        accessSync: (file, mode) => { if (/\/uv$|\/gcc$|Python\.h$/.test(String(file))) { if (tools) return undefined; throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' }); } return real.accessSync(file, mode); },
        existsSync: (file) => (String(file) === '/opt/runners' ? true : real.existsSync(file)),
        statSync: (file) => (/^\/(?:data|opt\/runners)$/.test(String(file)) ? { dev: typeof dev === 'function' ? dev(file) : dev } : real.statSync(file)),
    };
    const statfs = (target) => ({ bavail: Math.floor((typeof free === 'function' ? free(target) : free) / 4096), bsize: 4096, blocks: Math.floor(1000 * GIB / 4096) });
    const run = async (file, args) => (python === null ? { ok: false, status: 1, stdout: '', stderr: 'not found', error: 'ENOENT' } : { ok: true, status: 0, stdout: `${python}\n`, stderr: '', error: null });
    return {
        report: () => prerequisiteReport({
            pins, arch, fsApi, statfs, run, imageLockFile: lockFile, agentLockFile: null, readGpuImpl: async () => { if (gpuThrows) throw new Error('nvidia-smi failed'); return structuredClone(gpu); },
            catalog: loadSeedCatalog(),
        }),
    };
}
const codes = (report) => report.blockers.map((entry) => entry.code).sort();

test('CAL.prerequisites-pass-and-report-the-entry-wheels-and-download-size', async (t) => {
    const report = await prerequisiteWorld(t).report();
    assert.equal(report.ok, true, JSON.stringify(report.blockers));
    const { vllm } = report.facts.lock;
    assert.equal(vllm.version, '0.30.0'); assert.equal(vllm.kind, 'python');
    assert.equal(vllm.files, WHEELS.length); assert.equal(vllm.downloadBytes, WHEELS.reduce((sum, wheel) => sum + wheel[1], 0));
    assert.equal(vllm.runnerLockDigest, vllmRunnerLockDigest(ENTRY), 'the digest is production\'s');
    assert.deepEqual(vllm.distributions, { vllm: '0.30.0', torch: '2.13.0', triton: '3.7.1' });
    assert.equal(vllm.torchWheel.name, WHEELS[1][0]); assert.deepEqual(vllm.cudaRuntime, { major: 13, minor: 0, wheel: WHEELS[2][0] });
    assert.equal(vllm.largest[0].name, WHEELS[1][0]); assert.match(vllm.fileListDigest, /^[0-9a-f]{64}$/);
    assert.deepEqual(vllm.hosts, ['files.pythonhosted.org']);
    assert.equal(report.facts.driverVersion, '595.91.07'); assert.deepEqual([report.facts.python.major, report.facts.python.minor], [3, 13]);
    assert.equal(report.facts.disk.sameFilesystem, true); assert.ok(report.facts.disk.need.dataBytes > vllm.downloadBytes);
    assert.match(report.facts.lock.sha256, /^[0-9a-f]{64}$/);
});

test('CAL.prerequisites-block-without-a-vllm-entry-and-never-pass', async (t) => {
    const lock = lockDocument(); delete lock.runners.vllm;
    lock.runners.tabbyapi = { ...lockDocument().runners.vllm };
    const report = await prerequisiteWorld(t, { lock }).report();
    assert.equal(report.ok, false);
    assert.ok(codes(report).includes('vllm_entry_missing'), JSON.stringify(report.blockers));
    const missing = report.blockers.find((entry) => entry.code === 'vllm_entry_missing');
    assert.deepEqual(missing.evidence.runners, ['tabbyapi']); assert.match(missing.message, /no vLLM entry/);
    // An image with no lock at all, or an invalid one, blocks too.
    const empty = await prerequisiteWorld(t, { lock: { schema: 'local-llm.runners-lock/v1', runners: {} } }).report();
    assert.equal(empty.ok, false); assert.ok(codes(empty).includes('vllm_entry_missing'));
    const invalid = await prerequisiteWorld(t, { lock: { schema: 'other' } }).report();
    assert.equal(invalid.ok, false); assert.ok(codes(invalid).includes('lock_unreadable'));
    const archive = await prerequisiteWorld(t, { lock: (() => { const doc = lockDocument({ kind: 'archive', wheels: [] }); doc.runners.vllm.files = [{ name: 'x.tar.gz', url: url('x.tar.gz'), size: 5, sha256: SHA('9') }]; return doc; })() }).report();
    assert.ok(codes(archive).includes('vllm_entry_not_python'));
});

test('CAL.prerequisites-block-each-missing-requirement-with-its-evidence', async (t) => {
    const cases = [
        ['an arm64 container', { arch: 'arm64' }, 'unsupported_platform'],
        ['a wheel for another platform', { lock: lockDocument({ wheels: [...WHEELS, ['numpy-2.0.0-cp313-cp313-manylinux_2_28_aarch64.whl', 5000, '8']] }) }, 'wheel_platform'],
        ['a Python the wheels do not fit', { python: '3 12 /usr/include/python3.12' }, 'python_abi'],
        ['an interpreter that does not answer', { python: null }, 'python_unavailable'],
        ['no CUDA wheels', { lock: lockDocument({ wheels: WHEELS.filter((wheel) => !/^nvidia/.test(wheel[0])) }) }, 'no_cuda_wheels'],
        ['a driver older than the CUDA 13 wheels need', { gpu: { ...RTX, driverVersion: '570.86.10' } }, 'driver_too_old'],
        ['no usable GPU', { gpu: { available: false, state: 'absent', reason: 'No GPU is available to this agent.' } }, 'gpu_unavailable'],
        ['an unreadable GPU', { gpuThrows: true }, 'gpu_unreadable'],
        ['a unified-memory GPU', { gpu: { ...RTX, memoryModel: 'unified' } }, 'memory_model'],
        ['no uv, compiler or Python headers', { tools: false }, 'toolchain_missing'],
        ['too little disk on one filesystem', { free: 5 * GIB }, 'insufficient_disk'],
        ['too little disk on two filesystems', { free: (target) => (target === '/data' ? 400 * GIB : 5 * GIB), dev: (target) => (target === '/data' ? '1' : '2') }, 'insufficient_disk'],
        ['a licence that needs acceptance', { lock: lockDocument({ requiresAcceptance: true }) }, 'licence_acceptance_required'],
        ['no pins at all', { pins: null }, 'pins_missing'],
        ['a different pinned version', { pins: { ...PINS, version: '0.29.0' } }, 'pin_mismatch'],
        ['a different pinned digest', { pins: { ...PINS, runnerLockDigest: SHA('0') } }, 'pin_mismatch'],
        ['a different download size', { pins: { ...PINS, downloadBytes: PINS.downloadBytes + 1 } }, 'pin_mismatch'],
    ];
    for (const [label, options, code] of cases) {
        const report = await prerequisiteWorld(t, options).report();
        assert.equal(report.ok, false, `${label} must not pass`);
        assert.ok(codes(report).includes(code), `${label}: expected ${code}, got ${codes(report)}`);
        assert.ok(report.blockers.every((entry) => typeof entry.message === 'string' && entry.message.length > 10 && typeof entry.evidence === 'object'), label);
    }
    // The disk blocker states the numbers.
    const disk = (await prerequisiteWorld(t, { free: 5 * GIB }).report()).blockers.find((entry) => entry.code === 'insufficient_disk');
    assert.equal(disk.evidence.data.freeBytes, 5 * GIB); assert.ok(disk.evidence.need.dataBytes > 3 * GIB); assert.match(disk.message, /bytes free/);
    // The wheel-name grammar: abi3 fits a newer Python, a CPython tag only its own.
    assert.deepEqual(wheelPythonTag('x-1-cp38-abi3-manylinux_2_28_x86_64.whl'), { kind: 'abi3', minor: 8, platform: 'manylinux_2_28_x86_64' });
    assert.equal(wheelPythonTag('x-1-cp313-cp313-linux_x86_64.whl').kind, 'cpython');
    assert.equal(wheelPythonTag('x-1-py3-none-any.whl').kind, 'any');
});

// --- Stage 1 ------------------------------------------------------------------------------------
test('CAL.tuple-comes-from-the-production-functions-over-the-real-readings', async (t) => {
    const { report } = await calibrate(t);
    assert.equal(report.ok, true, JSON.stringify(report.blockers));
    const { evidence } = report;
    // Exactly what production computes from the same GPU and the same lock entry.
    assert.deepEqual(evidence.tuple, vllmMpsTuple({ gpu: RTX, runnerLockEntry: ENTRY }));
    assert.equal(evidence.tuple.runnerLockDigest, vllmRunnerLockDigest(ENTRY));
    assert.deepEqual(Object.keys(evidence.tuple), [...TUPLE_FIELDS].sort((a, b) => Object.keys(vllmMpsTuple({ gpu: RTX, runnerLockEntry: ENTRY })).indexOf(a) - Object.keys(vllmMpsTuple({ gpu: RTX, runnerLockEntry: ENTRY })).indexOf(b)));
    assert.deepEqual(evidence.tuple, { runnerLockDigest: vllmRunnerLockDigest(ENTRY), driverVersion: '595.91.07', gpuPciDeviceId: '0x252010DE', computeCapability: '8.6', deviceTotalBytes: 6144 * MIB });
    // The production qualification path accepts the rendered entry for this very tuple, and the adapter's own qualifyMps agrees.
    const rendered = renderQualificationEntry(evidence);
    assert.deepEqual(vllmRunner.qualifyMps({ gpu: RTX, runnerLockEntry: ENTRY, qualificationDataProvider: () => [rendered.entry] }),
        { qualified: true, denominator: 'physical-device', evidenceDigest: rendered.digest });
    // Another digest function would not match: the lock digest is not a digest of the file list or of the entry's own `digest` field.
    assert.notEqual(evidence.tuple.runnerLockDigest, ENTRY.digest);
    assert.equal(vllmRunner.qualifyMps({ gpu: RTX, runnerLockEntry: { ...ENTRY, files: ENTRY.files.slice(1) }, qualificationDataProvider: () => [rendered.entry] }).qualified, false);
    assert.equal(resolveVllmMpsQualification(evidence.tuple).qualified, false, 'the packaged data holds no entry yet');
});

test('CAL.calibration-runs-bounded-queries-under-both-limits-compares-with-nvml-and-builds-the-production-argv', async (t) => {
    const { report, root, calls } = await calibrate(t);
    const { evidence } = report;
    // Five bounded queries: torch and the driver, under the saved limit and a tighter one, and the pinned distributions' versions.
    assert.equal(calls.length, 5);
    for (const call of calls) { assert.ok(call.timeoutMs <= 180_000); assert.equal(call.args[0], '-c'); assert.equal(Object.hasOwn(call.env, 'VLLM_API_KEY'), false, 'no key reaches a query'); }
    assert.deepEqual(calls.map((call) => (call.file.startsWith(root) ? path.relative(root, call.file) : call.file)), ['vllm/0.30.0/venv/bin/python', 'vllm/0.30.0/venv/bin/python', '/usr/bin/python3', '/usr/bin/python3', 'vllm/0.30.0/venv/bin/python']);
    assert.deepEqual(calls.map((call) => call.env.CUDA_MPS_PINNED_DEVICE_MEM_LIMIT), ['0=5529M', '0=2048M', '0=5529M', '0=2048M', '0=5529M']);
    assert.deepEqual(calls[4].args.slice(2), ['vllm', 'torch', 'triton'], 'the lock\'s pinned distributions are asked for by name');
    assert.ok(calls.every((call) => call.env.CUDA_MPS_PIPE_DIRECTORY === '/run/ploinky-mps-pipe' && call.env.LD_LIBRARY_PATH === '/usr/local/nvidia/lib64'));
    const m = evidence.measurements;
    assert.equal(m.torchShare.memGetInfo.total, USABLE); assert.equal(m.torchShare.totalMemory, USABLE);
    assert.equal(m.ctypesShare.total, USABLE); assert.equal(m.torchTight.memGetInfo.total, USABLE);
    assert.equal(m.hostNvmlTotalBytes, 6144 * MIB); assert.equal(m.containerNvmlTotalBytes, 6144 * MIB);
    // The intended budget is production's admission over the share: 81 % of the physical device, fitting.
    assert.equal(evidence.admission.status, 'ok'); assert.equal(m.utilization, 0.81);
    assert.equal(m.requestedBytes, Math.round(0.81 * USABLE)); assert.equal(m.intendedBytes, Math.round(0.81 * 6144 * MIB * 0.94));
    // The final argv is the adapter's buildLaunch with the admitted utilization, and no secret.
    const model = loadSeedCatalog().find((entry) => entry.id === 'qwen3-4b-awq');
    const expected = vllmRunner.buildLaunch({
        runnerDir: path.join(root, 'vllm', '0.30.0'), artifactPath: `/data/models/hf/${model.sources.hf.repo}/${model.sources.hf.commit}`, params: vllmRunner.normalizeParams({}, { model, profile: 'dedicated' }),
        port: 18082, apiKey: 'k'.repeat(43), model, gpuMemoryUtilization: 0.81, cacheDir: path.join(root, '.cache', 'vllm'), profile: 'dedicated',
    });
    assert.deepEqual(evidence.argv.args, expected.args); assert.equal(evidence.argv.command, expected.command);
    assert.equal(evidence.argv.args[evidence.argv.args.indexOf('--gpu-memory-utilization') + 1], '0.81');
    assert.equal(JSON.stringify(evidence).includes('k'.repeat(43)), false);
    assert.ok(evidence.argv.envNames.includes('VLLM_API_KEY') && !JSON.stringify(evidence.argv).includes('VLLM_API_KEY='));
    // The sizing expression of the installed version, with file and line, and the installed version.
    assert.equal(evidence.install.packageVersion, '0.30.0');
    assert.ok(evidence.sizing.lines.some((line) => line.file === 'v1/worker/gpu_worker.py' && line.line === 2 && /gpu_memory_utilization/.test(line.text)), JSON.stringify(evidence.sizing));
    assert.ok(evidence.sizing.lines.some((line) => /mem_get_info/.test(line.text)));
    // The reviewed statements of this exact version were found, with file and line: that is the sizing evidence.
    assert.deepEqual(evidence.sizing.rules.request.map((hit) => [hit.file, hit.line]), [['v1/worker/gpu_worker.py', 2]]);
    assert.deepEqual(evidence.sizing.rules.total.map((hit) => [hit.file, hit.line]), [['utils/mem_utils.py', 3]]);
    assert.deepEqual([evidence.sizing.verdict.ok, evidence.sizing.verdict.reviewed, evidence.sizing.verdict.installedVersion, evidence.sizing.verdict.lockVersion], [true, true, '0.30.0', '0.30.0']);
    // The verdict and the digest.
    assert.equal(evidence.denominator.denominator, 'physical-device');
    assert.deepEqual(evidence.verdict, { qualifiable: true, denominator: 'physical-device', checks: evidence.verdict.checks, failed: [] });
    assert.ok(Object.values(evidence.verdict.checks).every(Boolean), JSON.stringify(evidence.verdict.checks));
    assert.equal(evidence.evidenceDigest, evidenceDigest(evidence)); assert.equal(evidence.schema, CALIBRATION_SCHEMA);
    // A qualifiable calibration proposes the reviewed entry for its own evidence, rendered by the same helper.
    assert.deepEqual(report.proposed, { entry: renderQualificationEntry(evidence).entry, digest: evidence.evidenceDigest, source: renderQualificationEntry(evidence).source });
    assert.equal(report.proposed.entry.evidenceDigest, evidence.evidenceDigest);
    assert.ok(Buffer.byteLength(JSON.stringify(report)) < 48 * 1024, 'the document fits the output bound');
});

test('CAL.the-denominator-is-classified-physical-share-or-unknown-from-the-two-limits', () => {
    const base = { nvmlBytes: 6144 * MIB, shareBytes: 5529 * MIB, tightBytes: 2048 * MIB };
    const view = (share, tight, cShare = share, cTight = tight) => ({ ...base, torchShare: share, torchTight: tight, ctypesShare: cShare, ctypesTight: cTight });
    assert.equal(classifyDenominator(view(USABLE, USABLE)).denominator, 'physical-device', 'independent of the limit and the device\'s usable memory');
    assert.equal(classifyDenominator(view(USABLE, USABLE + 10 * MIB)).denominator, 'physical-device', 'a 10 MiB wobble is independence');
    assert.equal(classifyDenominator(view(5529 * MIB, 2048 * MIB)).denominator, 'share', 'the total follows the pinned limit');
    for (const [label, input] of [
        ['torch and the driver disagree', view(USABLE, USABLE, USABLE - 5 * MIB)],
        ['a missing view', view(USABLE, null)],
        ['independent of the limit but not the device (a small total)', view(3000 * MIB, 3000 * MIB)],
        ['bigger than the device', view(7000 * MIB, 7000 * MIB)],
        ['moves with the limit but exceeds it', view(6000 * MIB, 2048 * MIB)],
        ['a malformed view', view('x', USABLE)],
    ]) assert.equal(classifyDenominator(input).denominator, 'unknown', label);
});

test('CAL.a-share-denominator-or-a-failed-check-is-not-qualifiable-and-renders-no-entry', async (t) => {
    const share = (await calibrate(t, { views: shareViews })).report;
    assert.equal(share.ok, true, 'the calibration itself ran');
    assert.equal(share.evidence.denominator.denominator, 'share');
    assert.equal(share.evidence.verdict.qualifiable, false);
    assert.ok(share.evidence.verdict.failed.includes('denominatorIsPhysical') && share.evidence.verdict.failed.includes('matchesIntended'), share.evidence.verdict.failed.join(','));
    assert.throws(() => renderQualificationEntry(share.evidence), /did not establish the physical-device denominator/);
    assert.equal(share.proposed, undefined, 'a share denominator proposes nothing');
    // Every other check fails the verdict by itself.
    for (const [label, options, failed] of [
        ['a device the wheel does not support', { torch: { archList: ['sm_70', 'sm_75'] } }, 'archSupported'],
        ['CUDA unavailable to torch', { torch: { cudaAvailable: false, memGetInfo: undefined, totalMemory: undefined } }, 'cudaAvailable'],
        ['two torch totals that differ', { torch: { totalMemory: USABLE - MIB } }, 'torchViewsAgree'],
        ['a host NVML total that differs from the container\'s', { hostNvmlBytes: 8192 * MIB }, 'hostAgrees'],
        ['a total that is not 94 % of the device', { views: () => 4096 * MIB }, 'matchesIntended'],
    ]) {
        const { report } = await calibrate(t, options);
        assert.equal(report.evidence?.verdict?.qualifiable, false, label);
        assert.ok(report.evidence.verdict.failed.includes(failed), `${label}: ${report.evidence.verdict.failed}`);
        assert.throws(() => renderQualificationEntry(report.evidence), /Cannot render a qualification entry/, label);
    }
    // A share that admission says does not fit (the share is far too small for the model).
    const small = await calibrate(t, { env: { CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '0=1024M' } });
    assert.equal(small.report.evidence.admission.status, 'incompatible'); assert.ok(small.report.evidence.verdict.failed.includes('admissionFits'));
    assert.ok(small.report.evidence.verdict.failed.includes('argvBuilt') && small.report.evidence.argv.error, 'no valid launch exists for an unfit share');
});

test('CAL.stage-one-blocks-without-a-share-a-gpu-an-install-or-a-query-and-never-invents-a-document', async (t) => {
    const noShare = await calibrate(t, { env: { CUDA_MPS_PIPE_DIRECTORY: undefined, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: undefined, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: undefined } });
    assert.equal(noShare.report.ok, false); assert.equal(noShare.report.evidence, null); assert.equal(noShare.report.blockers[0].code, 'no_mps_share');
    const partial = await calibrate(t, { env: { CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: undefined } });
    assert.equal(partial.report.blockers[0].code, 'no_mps_share');
    const noGpu = await calibrate(t, { gpu: { available: false, state: 'absent', reason: 'none' } });
    assert.equal(noGpu.report.blockers[0].code, 'gpu_unusable');
    const unified = await calibrate(t, { gpu: { ...RTX, memoryModel: 'unified' } });
    assert.equal(unified.report.blockers[0].code, 'gpu_unusable');
    const noDevice = await calibrate(t, { gpu: { ...RTX, device: { pciDeviceId: null, computeCapability: null } } });
    assert.equal(noDevice.report.blockers[0].code, 'gpu_unusable');
    // Not installed: the venv python is missing.
    const root = scratch(t);
    const missing = await calibrationReport({ env: { ...MPS_ENV }, runRoot: root, readGpuImpl: async () => RTX, locksImpl: () => LOCK, run: fakeRun({ views: physicalViews }), readMemoryImpl: () => ({}) });
    assert.equal(missing.blockers[0].code, 'vllm_not_installed'); assert.equal(missing.evidence, null);
    // A failed query is a blocker; the document that would carry it is not qualifiable.
    for (const script of [TORCH_QUERY, CTYPES_QUERY]) {
        const failed = await calibrate(t, { fail: script });
        assert.ok(failed.report.blockers.some((entry) => entry.code === 'query_failed'), script.slice(0, 20));
        assert.equal(failed.report.ok, false); assert.equal(failed.report.evidence.verdict.qualifiable, false);
    }
});

test('CAL.the-evidence-digest-is-canonical-and-covers-the-tuple-and-every-measurement', async (t) => {
    const { evidence } = (await calibrate(t)).report;
    const reverse = (value) => (Array.isArray(value) ? value.map(reverse) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).reverse().map((key) => [key, reverse(value[key])])) : value);
    const shuffled = reverse(evidence);
    assert.notEqual(JSON.stringify(shuffled), JSON.stringify(evidence), 'the shuffled document really differs in key order');
    assert.equal(evidenceDigest(shuffled), evidenceDigest(evidence), 'key order is not part of the digest');
    assert.equal(canonicalJson({ b: 1, a: [2, { d: 1, c: 2 }] }), '{"a":[2,{"c":2,"d":1}],"b":1}');
    const flip = (mutate) => { const copy = structuredClone(evidence); mutate(copy); return evidenceDigest(copy); };
    for (const [label, mutate] of [
        ['the lock digest', (doc) => { doc.tuple.runnerLockDigest = SHA('0'); }], ['the driver', (doc) => { doc.tuple.driverVersion = '595.91.08'; }],
        ['the PCI id', (doc) => { doc.tuple.gpuPciDeviceId = '0x252110DE'; }], ['the capability', (doc) => { doc.tuple.computeCapability = '8.9'; }],
        ['the device total', (doc) => { doc.tuple.deviceTotalBytes += 1; }], ['a torch view', (doc) => { doc.measurements.torchShare.memGetInfo.total += 1; }],
        ['the argv', (doc) => { doc.argv.args.push('--x'); }], ['the verdict', (doc) => { doc.verdict.qualifiable = false; }],
    ]) assert.notEqual(flip(mutate), evidence.evidenceDigest, `${label} changes the digest`);
    assert.equal(evidenceDigest({ ...evidence, evidenceDigest: 'ignored' }), evidence.evidenceDigest, 'the digest field is not part of itself');
});

test('CAL.the-rendered-entry-is-accepted-by-production-for-its-tuple-and-refused-after-any-change', async (t) => {
    const { evidence } = (await calibrate(t)).report;
    const rendered = renderQualificationEntry(evidence);
    assert.deepEqual(Object.keys(rendered.entry), [...TUPLE_FIELDS, 'denominator', 'evidenceDigest']);
    assert.equal(rendered.entry.denominator, 'physical-device'); assert.equal(rendered.entry.evidenceDigest, evidence.evidenceDigest);
    const resolve = createVllmMpsQualificationResolver(() => [rendered.entry]);
    assert.deepEqual(resolve(evidence.tuple), { qualified: true, denominator: 'physical-device', evidenceDigest: evidence.evidenceDigest });
    for (const [field, other] of [['runnerLockDigest', SHA('0')], ['driverVersion', '595.91.08'], ['gpuPciDeviceId', '0x252110DE'], ['computeCapability', '8.9'], ['deviceTotalBytes', evidence.tuple.deviceTotalBytes + 1]]) {
        assert.equal(resolve({ ...evidence.tuple, [field]: other }).code, 'vllm_mps_unqualified', `a changed ${field} is unqualified`);
    }
    // A document that was edited after its digest was taken is refused; so is one that is not a calibration.
    const tampered = structuredClone(evidence); tampered.tuple.deviceTotalBytes += 1;
    assert.throws(() => renderQualificationEntry(tampered), /does not match its own digest/);
    assert.throws(() => renderQualificationEntry({ ...evidence, schema: 'other' }), /not a stage-1 calibration/);
    assert.throws(() => renderQualificationEntry(null), /not a stage-1 calibration/);
    // The source text is the entry, ready for REVIEWED_QUALIFICATIONS.
    assert.match(rendered.source, /^    Object\.freeze\(\{\n        runnerLockDigest: "[0-9a-f]{64}",\n/);
    assert.ok(rendered.source.includes(`evidenceDigest: '${evidence.evidenceDigest}'`));
});

test('CAL.the-data-entry-edits-only-the-reviewed-list-and-production-then-qualifies-exactly-that-tuple', async (t) => {
    const { evidence } = (await calibrate(t)).report;
    const rendered = renderQualificationEntry(evidence);
    const source = fs.readFileSync(new URL('../src/controller/vllmMpsQualification.mjs', import.meta.url), 'utf8');
    assert.match(source, /const REVIEWED_QUALIFICATIONS = Object\.freeze\(\[\]\);/);
    const patched = applyQualificationEntry(source, rendered);
    // Only that declaration changed.
    const remove = (text) => text.replace(/const REVIEWED_QUALIFICATIONS = Object\.freeze\(\[[\s\S]*?\]\);/, 'LIST');
    assert.equal(remove(patched), remove(source));
    assert.notEqual(patched, source);
    // The patched module is the production module: it qualifies this tuple and no other.
    const directory = scratch(t);
    const file = path.join(directory, 'vllmMpsQualification.mjs');
    fs.writeFileSync(file, patched);
    const module = await import(`${pathToFileURL(file).href}?patched`);
    assert.deepEqual(module.resolveVllmMpsQualification(evidence.tuple), { qualified: true, denominator: 'physical-device', evidenceDigest: evidence.evidenceDigest });
    assert.equal(module.resolveVllmMpsQualification({ ...evidence.tuple, deviceTotalBytes: evidence.tuple.deviceTotalBytes + 1 }).qualified, false);
    assert.equal(module.vllmRunnerLockDigest(ENTRY), vllmRunnerLockDigest(ENTRY));
    // A second entry is appended; the same evidence or the same tuple twice, and an unknown layout, are refused.
    assert.throws(() => applyQualificationEntry(patched, rendered), /already in REVIEWED_QUALIFICATIONS/);
    const other = renderQualificationEntry({ ...evidence, evidenceDigest: undefined, tuple: { ...evidence.tuple, driverVersion: '595.91.08' } });
    const two = applyQualificationEntry(patched, other);
    fs.writeFileSync(file, two);
    const both = await import(`${pathToFileURL(file).href}?two`);
    assert.equal(both.resolveVllmMpsQualification(evidence.tuple).qualified, true);
    assert.equal(both.resolveVllmMpsQualification({ ...evidence.tuple, driverVersion: '595.91.08' }).qualified, true);
    assert.throws(() => applyQualificationEntry('const x = 1;\n', rendered), /was not found/);
    assert.throws(() => applyQualificationEntry(source, { entry: rendered.entry }), /rendered entry is required/);
});

test('CAL.the-command-line-prints-one-bounded-json-document-and-an-exit-code', async (t) => {
    const root = scratch(t);
    installVllm(root);
    const lines = [];
    const deps = { out: (text) => lines.push(text), env: MPS_ENV, runRoot: root, readGpuImpl: async () => RTX, readMemoryImpl: () => ({ totalBytes: 31 * GIB, availableBytes: 24 * GIB }), locksImpl: () => LOCK, run: fakeRun({ views: physicalViews }) };
    assert.equal(await main(['calibrate', '--host-nvml-bytes', String(6144 * MIB)], deps), 0);
    assert.equal(lines.length, 1); assert.ok(lines[0].endsWith('\n'));
    const document = JSON.parse(lines[0]);
    assert.equal(document.ok, true); assert.equal(document.evidence.verdict.qualifiable, true);
    const file = path.join(root, 'evidence.json'); fs.writeFileSync(file, lines[0]);
    const rendered = [];
    assert.equal(await main(['render', '--evidence', file], { out: (text) => rendered.push(text) }), 0);
    assert.equal(JSON.parse(rendered[0]).digest, document.evidence.evidenceDigest);
    // A bad invocation, an unknown command and a blocked step are exit 3 with a blocker, never a thrown error.
    for (const argv of [['calibrate', '--host-nvml-bytes', 'x'], ['frobnicate'], ['calibrate', 'bad'], ['render', '--evidence', path.join(root, 'absent.json')]]) {
        const out = [];
        assert.equal(await main(argv, { ...deps, out: (text) => out.push(text) }), 3, argv.join(' '));
        assert.equal(JSON.parse(out[0]).blockers[0].code, 'tool_error');
    }
    const blockedOut = [];
    assert.equal(await main(['calibrate'], { ...deps, env: {}, out: (text) => blockedOut.push(text) }), 3);
    assert.equal(JSON.parse(blockedOut[0]).blockers[0].code, 'no_mps_share');
});

test('CAL.stage-one-never-writes-and-the-tool-only-reads-the-installed-package', async (t) => {
    const root = scratch(t);
    installVllm(root);
    const writes = [];
    const spy = new Proxy(fs, { get: (target, name) => (/^(?:write|append|unlink|rm|rename|mkdir|copyFile|chmod|truncate|symlink|link|open)/i.test(String(name)) ? (...args) => { writes.push([name, args[0]]); throw new Error('write'); } : target[name]) });
    const { report } = await calibrate(t, { fsApi: spy, install: undefined });
    void report;
    // The installed tree is read by the scan alone.
    const installed = path.join(root, 'vllm', '0.30.0', 'venv', 'lib', 'python3.13', 'site-packages', 'vllm');
    const before = fs.readFileSync(path.join(installed, 'v1', 'worker', 'gpu_worker.py'), 'utf8');
    scanSizingSource({ fsApi: spy, root: installed });
    assert.deepEqual(writes, []);
    assert.equal(fs.readFileSync(path.join(installed, 'v1', 'worker', 'gpu_worker.py'), 'utf8'), before);
    // Statically: the tool has no write, delete or spawn-shell call outside the bounded runner and `render`.
    const text = fs.readFileSync(new URL('../tools/vllm_mps_calibration.mjs', import.meta.url), 'utf8').replace(/^\s*\/\/.*$/gm, '');
    assert.equal(/writeFile|appendFile|unlinkSync|rmSync|renameSync|mkdirSync|copyFile|child_process'\)\.exec\b|spawn\(|shell: true/.test(text), false);
    assert.equal(/REVIEWED_QUALIFICATIONS\s*=/.test(text.replace(/const empty|const filled|const current/g, '')) && /writeFileSync\(.*vllmMpsQualification/.test(text), false);
});

// --- CAL1: what is installed and queried must be what the selected lock entry pins ---------------------------
test('CAL.identities-are-compared-with-the-selected-lock-entry-and-any-mismatch-or-missing-identity-is-not-qualifiable', async (t) => {
    const control = await calibrate(t);
    assert.equal(control.report.evidence.verdict.qualifiable, true, JSON.stringify(control.report.evidence.verdict));
    assert.equal(control.report.evidence.identity.matches, true); assert.deepEqual(control.report.evidence.identity.mismatches, []);
    assert.deepEqual(control.report.evidence.identity.observed, { vllm: '0.30.0', readyDigest: ENTRY.digest, torch: '2.13.0', torchCuda: '13.0', distributions: LOCKED_DISTRIBUTIONS });
    // A local version label is not part of the pin.
    const labelled = await calibrate(t, { torch: { torch: '2.13.0+cu130' }, distributions: { ...LOCKED_DISTRIBUTIONS, torch: '2.13.0+cu130' } });
    assert.equal(labelled.report.evidence.verdict.qualifiable, true);
    const cases = [
        ['an installed vLLM that is not the locked version', { install: { version: '0.29.0' } }, 'vLLM package version', '0.30.0', '0.29.0'],
        ['a runnable copy marker of another entry', { install: { ready: { digest: SHA('9') } } }, 'runnable copy marker digest', ENTRY.digest, SHA('9')],
        ['no runnable copy marker', { install: { ready: null } }, 'runnable copy marker digest', ENTRY.digest, null],
        ['a queried torch that is not the locked one', { torch: { torch: '2.12.0' } }, 'torch version (saved limit)', '2.13.0', '2.12.0'],
        ['a torch built for another CUDA runtime', { torch: { torchCuda: '12.8' } }, 'CUDA runtime of torch (saved limit)', '13.0', '12.8'],
        ['another pinned distribution', { distributions: { ...LOCKED_DISTRIBUTIONS, triton: '3.6.0' } }, 'distribution triton', '3.7.1', '3.6.0'],
        ['a pinned distribution that is not installed', { distributions: { vllm: '0.30.0', torch: '2.13.0' } }, 'distribution triton', '3.7.1', null],
        ['no readable vLLM version', { install: { version: null } }, null, null, null],
    ];
    for (const [label, options, what, expected, actual] of cases) {
        const { report } = await calibrate(t, options);
        const evidence = report.evidence;
        assert.equal(evidence.verdict.qualifiable, false, `${label} must not be qualifiable`);
        assert.ok(evidence.verdict.failed.includes('identityMatchesLock'), `${label}: ${evidence.verdict.failed}`);
        assert.equal(evidence.verdict.checks.identityMatchesLock, false, label);
        // Independent: nothing else failed because of it, and the real observations stay in the document.
        // A package that is not the locked version also fails the sizing evidence: its source belongs to that other version (CAL2).
        const consequences = what === 'vLLM package version' || what === null ? ['sizingEvidence'] : [];
        assert.deepEqual(evidence.verdict.failed.filter((name) => name !== 'identityMatchesLock' && !consequences.includes(name)), [], `${label}: only the identity check failed`);
        if (what) {
            const found = evidence.identity.mismatches.find((entry) => entry.what === what);
            assert.ok(found, `${label}: ${JSON.stringify(evidence.identity.mismatches)}`);
            assert.deepEqual([found.expected, found.actual], [expected, actual], label); assert.match(found.reason, /differs from the lock|cannot be read/, label);
        } else assert.ok(evidence.identity.mismatches.length > 0, label);
        assert.equal(report.proposed, undefined, `${label}: no entry is proposed`);
        assert.throws(() => renderQualificationEntry(evidence), /Cannot render a qualification entry/, label);
    }
    // The observations are kept whatever they are.
    const stale = await calibrate(t, { install: { version: '0.29.0' }, torch: { torch: '2.12.0' } });
    assert.deepEqual([stale.report.evidence.identity.observed.vllm, stale.report.evidence.identity.observed.torch], ['0.29.0', '2.12.0']);
    assert.equal(stale.report.evidence.install.packageVersion, '0.29.0'); assert.equal(stale.report.evidence.measurements.torchShare.torch, '2.12.0');
    assert.ok(stale.report.evidence.identity.mismatches.length >= 3, 'both the package and torch (twice) are named');
});

test('CAL.the-identity-comparison-is-pure-and-names-a-lock-without-pins-or-cuda', () => {
    const base = { entry: ENTRY, installedVersion: '0.30.0', ready: { digest: ENTRY.digest }, torch: { share: { torch: '2.13.0', torchCuda: '13.0' }, tight: { torch: '2.13.0', torchCuda: '13.0' } }, distributions: LOCKED_DISTRIBUTIONS };
    assert.equal(compareIdentities(base).matches, true);
    // A lock that pins no torch version or no CUDA wheel cannot be compared with, which is a mismatch, not a pass.
    const noTorch = { ...base, entry: { ...ENTRY, check: { ...ENTRY.check, distributions: { vllm: '0.30.0' } } } };
    assert.ok(compareIdentities(noTorch).mismatches.some((entry) => entry.what.startsWith('torch version') && entry.reason === 'the lock pins no such identity'));
    const noCuda = { ...base, entry: { ...ENTRY, files: ENTRY.files.filter((file) => !/^nvidia/.test(file.name)) } };
    assert.ok(compareIdentities(noCuda).mismatches.some((entry) => entry.what.startsWith('CUDA runtime') && entry.reason === 'the lock pins no such identity'));
    // Every identity is its own comparison.
    for (const [label, change, what] of [
        ['package', { installedVersion: '0.31.0' }, 'vLLM package version'], ['marker', { ready: { digest: 'x' } }, 'runnable copy marker digest'], ['marker missing', { ready: null }, 'runnable copy marker digest'],
        ['tight torch', { torch: { ...base.torch, tight: { torch: '2.1.0', torchCuda: '13.0' } } }, 'torch version (tighter limit)'], ['tight cuda', { torch: { ...base.torch, tight: { torch: '2.13.0', torchCuda: null } } }, 'CUDA runtime of torch (tighter limit)'],
        ['no distributions', { distributions: null }, 'distribution vllm'],
    ]) {
        const result = compareIdentities({ ...base, ...change });
        assert.equal(result.matches, false, label); assert.ok(result.mismatches.some((entry) => entry.what === what), `${label}: ${JSON.stringify(result.mismatches)}`);
    }
});

// --- CAL2: the sizing evidence of the exact locked version -------------------------------------------------
test('CAL.qualification-needs-the-reviewed-sizing-statements-of-the-locked-version-and-a-keyword-is-never-proof', async (t) => {
    const control = await calibrate(t);
    assert.equal(control.report.evidence.verdict.qualifiable, true, JSON.stringify(control.report.evidence.verdict));
    assert.equal(control.report.evidence.verdict.checks.sizingEvidence, true);
    const REQUEST = 'requested_memory = snapshot.total_memory * cache_config.gpu_memory_utilization\n';
    const TOTAL = 'self.free_memory, self.total_memory = torch.cuda.mem_get_info()\n';
    const cases = [
        ['no sizing file at all', { install: { sizing: false } }, /no sizing source files were found/],
        ['keyword hits that state nothing', { install: { sizing: { worker: '# gpu_memory_utilization is the fraction of total_memory that is used\n', utils: '# see mem_get_info for the free and total memory\n' } } }, /no statement computes the requested memory/],
        ['the product without the total', { install: { sizing: { worker: REQUEST, utils: '# mem_get_info\nvalue = total_memory\n' } } }, /no statement reads the device total from mem_get_info/],
        ['the total without the product', { install: { sizing: { worker: 'requested = gpu_memory_utilization\n', utils: TOTAL } } }, /no statement computes the requested memory/],
        ['a statement that takes the total from another source', { install: { sizing: { worker: REQUEST, utils: 'self.free_memory, self.total_memory = some_cache.read()\n' } } }, /no statement reads the device total from mem_get_info/],
        ['a scan that was cut short', { install: { extraFiles: 130 } }, /sizing scan was cut short/],
    ];
    for (const [label, options, reason] of cases) {
        const { report } = await calibrate(t, options);
        const evidence = report.evidence;
        assert.equal(evidence.verdict.qualifiable, false, `${label} must not be qualifiable`);
        assert.ok(evidence.verdict.failed.includes('sizingEvidence'), `${label}: ${evidence.verdict.failed}`);
        assert.deepEqual(evidence.verdict.failed.filter((name) => name !== 'sizingEvidence'), [], `${label}: only the sizing check failed`);
        assert.ok(evidence.sizing.verdict.reasons.some((entry) => reason.test(entry)), `${label}: ${JSON.stringify(evidence.sizing.verdict.reasons)}`);
        assert.equal(report.proposed, undefined, label); assert.throws(() => renderQualificationEntry(evidence), /Cannot render a qualification entry/, label);
    }
    // Keyword evidence stays in the document, but it is evidence, not the verdict.
    const keywords = await calibrate(t, { install: { sizing: { worker: '# gpu_memory_utilization total_memory\n', utils: '# mem_get_info\n' } } });
    assert.ok(keywords.report.evidence.sizing.lines.length >= 2); assert.deepEqual(keywords.report.evidence.sizing.rules, { request: [], total: [] });
    // A source of another version than the lock pins, and a version whose sizing was never reviewed.
    const other = await calibrate(t, { install: { version: '0.29.0' } });
    assert.ok(other.report.evidence.sizing.verdict.reasons.some((entry) => /belongs to vLLM 0\.29\.0, but the lock pins 0\.30\.0/.test(entry)));
    assert.ok(other.report.evidence.verdict.failed.includes('sizingEvidence') && other.report.evidence.verdict.failed.includes('identityMatchesLock'));
    assert.equal(sizingVerdict({ installedVersion: '0.31.0', lockVersion: '0.31.0', scan: scanSizingSource({ root: '/nonexistent', version: '0.31.0' }) }).ok, false);
    const unreviewed = sizingVerdict({ installedVersion: '0.31.0', lockVersion: '0.31.0', scan: { filesScanned: 2, truncated: false, rules: { request: [{}], total: [{}] } } });
    assert.equal(unreviewed.ok, false); assert.ok(unreviewed.reasons.some((entry) => /vLLM 0\.31\.0 has not been reviewed/.test(entry)), 'a version without reviewed semantics is never qualifiable, whatever the scan found');
    assert.deepEqual(Object.keys(REVIEWED_SIZING), ['0.30.0']);
    // The helper refuses evidence without the sizing evidence, even when it was re-digested.
    const doctored = structuredClone(control.report.evidence);
    doctored.sizing.verdict.ok = false; doctored.verdict.checks.sizingEvidence = false; delete doctored.evidenceDigest;
    assert.throws(() => renderQualificationEntry(doctored), /holds no reviewed sizing expression/);
});

test('CAL.the-sizing-scan-is-complete-reports-where-each-statement-was-found-and-treats-unreadable-files-as-truncation', (t) => {
    const root = scratch(t);
    installVllm(root);
    const site = path.join(root, 'vllm', '0.30.0', 'venv', 'lib', 'python3.13', 'site-packages', 'vllm');
    const scan = scanSizingSource({ root: site, version: '0.30.0' });
    assert.equal(scan.truncated, false); assert.equal(scan.unreadable, 0); assert.equal(scan.filesScanned, 2);
    assert.deepEqual([scan.rules.request[0].file, scan.rules.total[0].file], ['v1/worker/gpu_worker.py', 'utils/mem_utils.py']);
    // The evidence lines are bounded without making the scan incomplete: the rules come from the whole scan.
    fs.writeFileSync(path.join(site, 'utils', 'aaa_noise.py'), `${'x = total_memory\n'.repeat(200)}`);
    const noisy = scanSizingSource({ root: site, version: '0.30.0' });
    assert.equal(noisy.lines.length, 40); assert.ok(noisy.linesOmitted > 100); assert.equal(noisy.truncated, false); assert.ok(noisy.rules.request.length >= 1 && noisy.rules.total.length >= 1);
    // A file that cannot be read cuts the scan short.
    const spy = new Proxy(fs, { get: (target, name) => (name === 'readFileSync' ? (file, ...rest) => { if (String(file).endsWith('mem_utils.py')) throw Object.assign(new Error('EACCES'), { code: 'EACCES' }); return target.readFileSync(file, ...rest); } : target[name]) });
    const blocked = scanSizingSource({ fsApi: spy, root: site, version: '0.30.0' });
    assert.deepEqual([blocked.truncated, blocked.unreadable], [true, 1]);
    assert.equal(sizingVerdict({ installedVersion: '0.30.0', lockVersion: '0.30.0', scan: blocked }).ok, false);
    // Without a reviewed version no rule is applied at all.
    assert.deepEqual(scanSizingSource({ root: site, version: '9.9.9' }).rules, { request: [], total: [] });
});
