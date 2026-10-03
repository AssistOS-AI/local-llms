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
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { createVllmMpsQualificationResolver, resolveVllmMpsQualification, vllmRunnerLockDigest } from '../src/controller/vllmMpsQualification.mjs';
import { vllmMpsTuple, vllmRunner } from '../src/runners/vllm.mjs';
import {
    CALIBRATION_SCHEMA, CTYPES_QUERY, DIST_QUERY, TORCH_QUERY, TUPLE_FIELDS, applyQualificationEntry, calibrationReport, canonicalJson, classifyDenominator, compareIdentities, evidenceDigest,
    REVIEWED_SIZING, enclosingPythonScope, main, prerequisiteReport, renderQualificationEntry, scanSizingSource, sizingVerdict, stripPythonNonCode, wheelPythonTag,
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
// `sizing`: true writes the reviewed statements, false writes no sizing file, or an object { worker, utils, gpuWorker? } gives the text of
// v1/worker/utils.py (the request), utils/mem_utils.py (the total) and, when given, v1/worker/gpu_worker.py.
function installVllm(root, { sizing = true, version = '0.30.0', ready = { digest: ENTRY.digest }, extraFiles = 0, extraPackageFiles = {} } = {}) {
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
        fs.writeFileSync(path.join(site, 'vllm', 'v1', 'worker', 'utils.py'), text.worker);
        if (text.gpuWorker !== undefined) fs.writeFileSync(path.join(site, 'vllm', 'v1', 'worker', 'gpu_worker.py'), text.gpuWorker);
        fs.writeFileSync(path.join(site, 'vllm', 'utils', 'mem_utils.py'), text.utils);
    }
    for (const [relative, text] of Object.entries(extraPackageFiles)) { fs.mkdirSync(path.dirname(path.join(site, 'vllm', relative)), { recursive: true }); fs.writeFileSync(path.join(site, 'vllm', relative), text); }
    for (let index = 0; index < extraFiles; index += 1) fs.writeFileSync(path.join(site, 'vllm', 'utils', `zz_extra_${String(index).padStart(3, '0')}.py`), '# nothing\n');
    if (ready !== null) fs.writeFileSync(path.join(runner, '.ready.json'), JSON.stringify(ready));
    return runner;
}

// The in-client queries, scripted: torch and ctypes see `views(env)`; a different pinned limit is a different env.
const LOCKED_DISTRIBUTIONS = Object.freeze({ vllm: '0.30.0', torch: '2.13.0', triton: '3.7.1' });
// `torch` and `ctypes` override fields of the torch and the driver documents; a function receives the pinned limit, so one limit can differ.
function fakeRun({ views, torch = {}, ctypes = {}, fail = null, calls = [], distributions = LOCKED_DISTRIBUTIONS }) {
    return async (file, args, options = {}) => {
        calls.push({ file, args, env: options.env, timeoutMs: options.timeoutMs });
        const script = args[1];
        const limit = options.env?.CUDA_MPS_PINNED_DEVICE_MEM_LIMIT;
        if (fail === script) return { ok: false, status: 1, stdout: '', stderr: 'Traceback: no CUDA', error: '1' };
        const total = views(limit);
        if (script === TORCH_QUERY) {
            return {
                ok: true, status: 0, stderr: '', stdout: `${JSON.stringify({
                    python: '3.13.5', torch: '2.13.0', torchCuda: '13.0', cudaAvailable: true, memGetInfo: { free: total - 200 * MIB, total }, totalMemory: total, acceleratorMemoryInfo: { free: total - 200 * MIB, total }, name: RTX.name,
                    capability: [8, 6], multiProcessorCount: 30, archList: ['sm_75', 'sm_80', 'sm_86', 'sm_90'], mps: { CUDA_MPS_PIPE_DIRECTORY: options.env.CUDA_MPS_PIPE_DIRECTORY, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: options.env.CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: limit },
                    ...(typeof torch === 'function' ? torch(limit, total) : torch),
                })}\n`,
            };
        }
        if (script === CTYPES_QUERY) return { ok: true, status: 0, stderr: '', stdout: `${JSON.stringify({ free: total - 150 * MIB, total, ...(typeof ctypes === 'function' ? ctypes(limit, total) : ctypes) })}\n` };
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
        locksImpl: () => LOCK, run: fakeRun({ views: options.views || physicalViews, torch: options.torch, ctypes: options.ctypes, fail: options.fail, calls, ...(options.distributions ? { distributions: options.distributions } : {}) }),
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
    assert.ok(evidence.sizing.lines.some((line) => line.file === 'v1/worker/utils.py' && line.line === 2 && /gpu_memory_utilization/.test(line.text)), JSON.stringify(evidence.sizing));
    assert.ok(evidence.sizing.lines.some((line) => /mem_get_info/.test(line.text)));
    // The reviewed statements of this exact version were found, with file and line: that is the sizing evidence.
    assert.deepEqual(evidence.sizing.rules.request.map((hit) => [hit.file, hit.line]), [['v1/worker/utils.py', 2]]);
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
    const before = fs.readFileSync(path.join(installed, 'v1', 'worker', 'utils.py'), 'utf8');
    scanSizingSource({ fsApi: spy, root: installed });
    assert.deepEqual(writes, []);
    assert.equal(fs.readFileSync(path.join(installed, 'v1', 'worker', 'utils.py'), 'utf8'), before);
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
    assert.deepEqual([scan.rules.request[0].file, scan.rules.total[0].file], ['v1/worker/utils.py', 'utils/mem_utils.py']);
    // The evidence lines are bounded without making the scan incomplete: the rules come from the whole scan.
    fs.writeFileSync(path.join(site, 'utils', 'aaa_noise.py'), `${'x = total_memory\n'.repeat(200)}`);
    const noisy = scanSizingSource({ root: site, version: '0.30.0' });
    // 40 evidence lines, plus the hit of each rule beyond the bound (the request hit is among the first 40; the total hit is listed after them).
    assert.equal(noisy.lines.length, 41); assert.ok(noisy.lines.some((line) => line.file === 'utils/mem_utils.py' && line.line === 3), 'a rule hit is never crowded out of the evidence lines'); assert.ok(noisy.linesOmitted > 100); assert.equal(noisy.truncated, false); assert.ok(noisy.rules.request.length >= 1 && noisy.rules.total.length >= 1);
    // A file that cannot be read cuts the scan short.
    const spy = new Proxy(fs, { get: (target, name) => (name === 'readFileSync' ? (file, ...rest) => { if (String(file).endsWith('mem_utils.py')) throw Object.assign(new Error('EACCES'), { code: 'EACCES' }); return target.readFileSync(file, ...rest); } : target[name]) });
    const blocked = scanSizingSource({ fsApi: spy, root: site, version: '0.30.0' });
    assert.deepEqual([blocked.truncated, blocked.unreadable], [true, 1]);
    assert.equal(sizingVerdict({ installedVersion: '0.30.0', lockVersion: '0.30.0', scan: blocked }).ok, false);
    // Without a reviewed version no rule is applied at all.
    assert.deepEqual(scanSizingSource({ root: site, version: '9.9.9' }).rules, { request: [], total: [] });
});

// --- CAL2b: only executable statements are sizing evidence ---------------------------------------------------
const CODE_REQUEST = 'requested_memory = snapshot.total_memory * cache_config.gpu_memory_utilization\n';
const CODE_TOTAL = 'self.free_memory, self.total_memory = torch.cuda.mem_get_info()\n';
// `extra`: more files under the vllm package (path relative to it -> text), for the cases where the same statement sits elsewhere.
const sizingOutcome = async (t, sizing, extra = undefined) => {
    const { report } = await calibrate(t, { install: { sizing, ...(extra ? { extraPackageFiles: extra } : {}) } });
    return report.evidence;
};
const refusedForSizing = (evidence, label) => {
    assert.equal(evidence.verdict.qualifiable, false, `${label}: not qualifiable`);
    assert.deepEqual(evidence.verdict.failed, ['sizingEvidence'], label);
    assert.deepEqual(evidence.sizing.rules, { request: [], total: [] }, `${label}: no rule is matched`);
    assert.ok(evidence.sizing.verdict.reasons.some((entry) => /no statement computes the requested memory/.test(entry)) && evidence.sizing.verdict.reasons.some((entry) => /no statement reads the device total/.test(entry)), `${label}: ${JSON.stringify(evidence.sizing.verdict.reasons)}`);
};

test('CAL.a-commented-out-expression-is-not-a-statement-even-when-unsupported-statements-follow-it', async (t) => {
    // The shape of the monitor's probe: the old reviewed expressions, commented out, then the actual (unsupported) statements.
    const worker = '# requested_memory = snapshot.total_memory * cache_config.gpu_memory_utilization\nrequested_memory = snapshot.free_memory - configured_cache_limit\n';
    const utils = '#   self.free_memory, self.total_memory = torch.cuda.mem_get_info()\nself.free_memory, self.total_memory = configured_cache_limit, configured_cache_limit\n';
    const evidence = await sizingOutcome(t, { worker, utils });
    refusedForSizing(evidence, 'commented-out expressions');
    // The comment is still listed as keyword evidence, and no proposal is made.
    assert.ok(evidence.sizing.lines.some((line) => line.line === 1 && /gpu_memory_utilization/.test(line.text)));
    const { report } = await calibrate(t, { install: { sizing: { worker, utils } } });
    assert.equal(report.proposed, undefined); assert.throws(() => renderQualificationEntry(report.evidence), /Cannot render a qualification entry/);
    // A comment that only ends a code line behaves the same way.
    refusedForSizing(await sizingOutcome(t, { worker: `x = 1  # ${CODE_REQUEST}`, utils: `y = 2  # ${CODE_TOTAL}` }), 'trailing comments');
});

test('CAL.a-quoted-expression-is-not-a-statement-docstrings-and-triple-quoted-blocks-included', async (t) => {
    const cases = [
        ['a docstring', { worker: `def f():\n    """\n    ${CODE_REQUEST}    """\n    return None\n`, utils: `def g():\n    '''\n    ${CODE_TOTAL}    '''\n    return None\n` }],
        ['single-line strings', { worker: 'NOTE = "requested_memory = snapshot.total_memory * cache_config.gpu_memory_utilization"\n', utils: "NOTE = 'self.free_memory, self.total_memory = torch.cuda.mem_get_info()'\n" }],
        ['strings with escaped quotes', { worker: 'NOTE = "say \\"hi\\"; requested = total_memory * gpu_memory_utilization"\n', utils: "NOTE = 'it\\'s total = torch.cuda.mem_get_info()'\n" }],
        ['f-strings and prefixed strings', { worker: 'NOTE = f"{x} requested = total_memory * gpu_memory_utilization"\n', utils: "NOTE = rb'total = torch.cuda.mem_get_info()'\n" }],
        ['a comment inside a string and a string inside a comment', { worker: 'NOTE = "# requested = total_memory * gpu_memory_utilization"\n', utils: "# NOTE = 'total = torch.cuda.mem_get_info()'\n" }],
    ];
    for (const [label, sizing] of cases) refusedForSizing(await sizingOutcome(t, sizing), label);
});

test('CAL.a-real-statement-next-to-comments-and-strings-qualifies-with-its-original-line-number', async (t) => {
    const worker = ['"""Memory sizing.', `${CODE_REQUEST.trim()} is described here, in the docstring.`, '"""', '# requested_memory = old.total * 0.9', 'def request_memory(snapshot, cache_config):', `    ${CODE_REQUEST.trim()}  # the product of the two`, '    return requested_memory', ''].join('\n');
    const utils = ["'''", CODE_TOTAL.trim(), "'''", 'class MemorySnapshot:', '    def measure(self):', `        ${CODE_TOTAL.trim()}  # (free, total)`, ''].join('\n');
    const evidence = await sizingOutcome(t, { worker, utils });
    assert.equal(evidence.verdict.qualifiable, true, JSON.stringify(evidence.verdict));
    // The hits are the executable lines, by their line in the original file (the docstring and comment lines are not hits).
    assert.deepEqual(evidence.sizing.rules.request.map((hit) => [hit.file, hit.line]), [['v1/worker/utils.py', 6]]);
    assert.deepEqual(evidence.sizing.rules.total.map((hit) => [hit.file, hit.line]), [['utils/mem_utils.py', 6]]);
    assert.match(evidence.sizing.rules.request[0].text, /requested_memory = snapshot\.total_memory \* cache_config\.gpu_memory_utilization/);
});

test('CAL.stripping-blanks-comments-and-literals-and-keeps-every-line', () => {
    const cases = [
        ['x = 1  # total_memory\n', 'x = 1  \n'],
        ['s = "a # b"  # c\n', 's = "     "  \n'],
        ["t = '''a\nb''' + y\n", "t = ''' \n ''' + y\n"],
        ['u = "q\\"q"; v = 2\n', 'u = "    "; v = 2\n'],
        ['w = "unterminated\nz = 3\n', `w = "${' '.repeat(12)}\nz = 3\n`],
        ['p = rb"\\"" + q\n', 'p = rb"  " + q\n'],
    ];
    for (const [input, expected] of cases) {
        const stripped = stripPythonNonCode(input);
        assert.equal(stripped, expected, JSON.stringify(input));
        assert.equal(stripped.split('\n').length, input.split('\n').length, 'every line is kept');
    }
    const source = '"""doc\nmore\n"""\nx = total_memory * gpu_memory_utilization  # c\n';
    assert.equal(stripPythonNonCode(source).split('\n')[3].includes('total_memory * gpu_memory_utilization'), true);
    assert.equal(stripPythonNonCode(source).split('\n').slice(0, 3).join('').replace(/["\s]/g, ''), '');
});

// --- CAL3: vLLM 0.30.0 reads its denominator through torch.accelerator.get_memory_info ----------------------------------
// Minimal excerpts of the REAL vLLM 0.30.0 tree (the files the live calibration scans), each at its real path under the vllm package, copied
// verbatim from the tree the reviewer verified against the wheel's RECORD sha256 values. Only the lines around the sizing statements are kept.
const REAL_STATEMENT = 'self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)';
const REAL_MEM_UTILS = [
    '@dataclass',
    'class MemorySnapshot:',
    '    """Memory snapshot."""',
    '',
    '    torch_peak: int = 0',
    '    torch_allocated: int = 0',
    '    free_memory: int = 0',
    '    total_memory: int = 0',
    '    cuda_memory: int = 0',
    '    torch_memory: int = 0',
    '    non_torch_memory: int = 0',
    '    timestamp: float = 0.0',
    '',
    '    device: torch.types.Device = None',
    '    auto_measure: bool = True',
    '',
    '    def __post_init__(self) -> None:',
    '        if self.device is None:',
    '            device_fn = current_platform.current_device',
    '            assert device_fn is not None',
    '            self.device_ = torch.device(device_fn())',
    '        else:',
    '            self.device_ = torch.device(self.device)',
    '',
    '        if self.auto_measure:',
    '            self.measure()',
    '',
    '    def measure(self) -> None:',
    '        device = self.device_',
    '',
    '        # we measure the torch peak memory usage via allocated_bytes,',
    '        # rather than `torch.accelerator.memory_reserved()` .',
    '        # After `torch.accelerator.reset_peak_memory_stats()`,',
    '        # `torch.accelerator.memory_reserved()` will keep growing, and only shrink',
    '        # when we call `torch.accelerator.empty_cache()` or OOM happens.',
    '        stats = torch.accelerator.memory_stats(device)',
    '        self.torch_peak = stats.get("allocated_bytes.all.peak", 0)',
    '        self.torch_allocated = stats.get("allocated_bytes.all.current", 0)',
    '',
    `        ${REAL_STATEMENT}`,
    '        if current_platform.is_integrated_gpu(device.index):',
    '            # On UMA (Unified Memory Architecture) platforms where CPU and',
    '            # GPU share physical memory (e.g. GH200, DGX Spark, Jetson Orin),',
    '            # cudaMemGetInfo underreports free memory because it does not',
    '            # account for reclaimable OS memory (page cache, buffers).',
    '            # Use psutil to get the true available memory.',
    '            self.free_memory = psutil.virtual_memory().available',
    '',
    '        self.cuda_memory = self.total_memory - self.free_memory',
    '',
    '        # torch.accelerator.memory_reserved() is how many bytes',
    '        # PyTorch gets from cuda (by calling cudaMalloc, etc.)',
    '        # this is used to measure the non-torch memory usage',
    '        self.torch_memory = torch.accelerator.memory_reserved(device)',
    '',
    '        self.non_torch_memory = self.cuda_memory - self.torch_memory',
    '        self.timestamp = time.time()',
    '',
].join('\n');
// The same call in the unrelated sleep-mode path of the worker: it must never satisfy the total.
const REAL_SLEEP_LINE = 'free_bytes_after_sleep, total = torch.accelerator.get_memory_info()';
const REAL_GPU_WORKER = [
    'class Worker(WorkerBase):',
    '    def sleep(self, level: int = 1) -> None:',
    '        torch.accelerator.synchronize()',
    '        deadline = time.monotonic() + (5.0 if current_platform.is_rocm() else 0)',
    '        while True:',
    `            ${REAL_SLEEP_LINE}`,
    '            freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep',
    '            if freed_bytes >= 0 or time.monotonic() >= deadline:',
    '                break',
    '            time.sleep(0.1)',
    '',
    '        used_bytes = total - free_bytes_after_sleep',
    '',
    '    def init_device(self):',
    '            # take current memory snapshot',
    '            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)',
    '            self.requested_memory = request_memory(init_snapshot, self.cache_config)',
    '',
].join('\n');
const REAL_REQUEST = [
    'def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:',
    '    """',
    '    Calculate the amount of memory required by vLLM, then validate',
    '    that the current amount of free memory is sufficient for that.',
    '    """',
    '    requested_memory = math.ceil(',
    '        init_snapshot.total_memory * cache_config.gpu_memory_utilization',
    '    )',
    '',
    '    if init_snapshot.free_memory < requested_memory:',
    '        raise ValueError(',
    '            f"Free memory on device {init_snapshot.device_} "',
    '            f"({cache_config.gpu_memory_utilization}, "',
    '        )',
    '',
    '    return requested_memory',
    '',
].join('\n');
const lineOf = (text, needle) => text.split('\n').findIndex((line) => line.includes(needle)) + 1;
const REAL_STATEMENT_LINE = lineOf(REAL_MEM_UTILS, REAL_STATEMENT);
const REAL_REQUEST_LINE = lineOf(REAL_REQUEST, 'init_snapshot.total_memory * cache_config.gpu_memory_utilization');
const REAL_SIZING = Object.freeze({ worker: REAL_REQUEST, utils: REAL_MEM_UTILS, gpuWorker: REAL_GPU_WORKER });
// A statement in the place the reviewed rule names (the class MemorySnapshot, its method measure), so that only its own text decides.
const inMeasure = (...body) => ['class MemorySnapshot:', '    def measure(self):', ...body.map((line) => `        ${line}`), ''].join('\n');
// The `total` rule of the reviewed entry as of d501bfc, before the accelerator change: only mem_get_info forms, in no file in particular.
const OLD_TOTAL_RULE = Object.freeze([
    String.raw`\b[\w.]*total[\w.]*\s*=\s*(?:torch\.cuda|current_platform)\.mem_get_info\(`,
    String.raw`,\s*[\w.]*total[\w.]*\s*=\s*(?:torch\.cuda|current_platform)\.mem_get_info\(`,
]);

test('CAL.the-real-vllm-0-30-0-statement-was-refused-by-the-old-rule-and-the-real-tree-excerpts-qualify-with-exactly-the-mem-utils-line', async (t) => {
    const code = stripPythonNonCode(REAL_MEM_UTILS).split('\n');
    // The d501bfc rule matches no executable line of the real method: that is the live refusal "no statement reads the device total".
    assert.equal(OLD_TOTAL_RULE.some((source) => code.some((line) => new RegExp(source).test(line))), false);
    const old = sizingVerdict({ installedVersion: '0.30.0', lockVersion: '0.30.0', scan: { filesScanned: 2, truncated: false, unreadable: 0, rules: { request: [{}], total: [] } } });
    assert.equal(old.ok, false); assert.ok(old.reasons.some((entry) => /no statement reads the device total from mem_get_info/.test(entry)), JSON.stringify(old.reasons));
    // The reviewed rule of this change matches exactly that line of the method, and only that one.
    const matching = REVIEWED_SIZING['0.30.0'].total.patterns.flatMap((source) => code.flatMap((line, index) => (new RegExp(source).test(line) ? [index + 1] : [])));
    assert.deepEqual([...new Set(matching)], [REAL_STATEMENT_LINE]);
    // Through the scan of the real tree excerpts (sleep path included), with their files and original lines, and through a whole calibration.
    const root = scratch(t);
    installVllm(root, { sizing: REAL_SIZING });
    const site = path.join(root, 'vllm', '0.30.0', 'venv', 'lib', 'python3.13', 'site-packages', 'vllm');
    const scan = scanSizingSource({ root: site, version: '0.30.0' });
    assert.deepEqual(scan.rules.total.map((hit) => [hit.file, hit.line, hit.text]), [['utils/mem_utils.py', REAL_STATEMENT_LINE, REAL_STATEMENT]], 'the sleep-mode line is no hit');
    assert.deepEqual(scan.rules.request.map((hit) => [hit.file, hit.line]), [['v1/worker/utils.py', REAL_REQUEST_LINE]]);
    assert.ok(scan.lines.some((line) => line.file === 'v1/worker/gpu_worker.py' && /get_memory_info/.test(line.text)), 'the sleep-mode line is listed as keyword evidence, not as a rule hit');
    assert.ok(scan.lines.some((line) => line.file === 'utils/mem_utils.py' && line.line === REAL_STATEMENT_LINE && /get_memory_info/.test(line.text)), 'the accelerator statement is listed as sizing evidence');
    assert.equal(sizingVerdict({ installedVersion: '0.30.0', lockVersion: '0.30.0', scan }).ok, true);
    const evidence = (await calibrate(t, { install: { sizing: REAL_SIZING } })).report.evidence;
    assert.equal(evidence.verdict.qualifiable, true, JSON.stringify(evidence.verdict));
    assert.deepEqual(evidence.verdict.failed, []);
    assert.deepEqual(evidence.sizing.rules.total.map((hit) => [hit.file, hit.line]), [['utils/mem_utils.py', REAL_STATEMENT_LINE]]);
});

test('CAL.a-denominator-from-a-constant-or-the-device-properties-in-the-real-tree-is-blocked-even-with-the-sleep-mode-line-present', async (t) => {
    const mutate = (replacement) => REAL_MEM_UTILS.replace(`        ${REAL_STATEMENT}`, replacement);
    for (const [label, utils] of [
        ['a constant', mutate('        self.free_memory, self.total_memory = 0, 6 * 1024 ** 3')],
        ['the device properties', mutate('        self.free_memory = torch.accelerator.get_memory_info(device)[0]\n        self.total_memory = torch.cuda.get_device_properties(device).total_memory')],
        ['another source', mutate('        self.free_memory, self.total_memory = some_cache.read()')],
    ]) {
        const evidence = await sizingOutcome(t, { ...REAL_SIZING, utils });
        assert.equal(evidence.verdict.qualifiable, false, `${label}: blocked`);
        assert.deepEqual(evidence.verdict.failed, ['sizingEvidence'], label);
        assert.deepEqual(evidence.sizing.rules.total, [], `${label}: the sleep-mode line is not the total`);
        assert.equal(evidence.sizing.rules.request.length, 1, label);
        assert.ok(evidence.sizing.lines.some((line) => line.file === 'v1/worker/gpu_worker.py' && line.text.includes('free_bytes_after_sleep')), `${label}: the sleep line is still evidence, not proof`);
    }
});

test('CAL.a-statement-counts-only-in-its-reviewed-file-class-and-method', async (t) => {
    const A = 'torch.accelerator.get_memory_info(device)';
    // The sleep line alone, and the real total statement in the wrong file, class, method or at module level, never satisfy the total.
    const noTotal = await sizingOutcome(t, { worker: REAL_REQUEST, utils: '# nothing here\n', gpuWorker: REAL_GPU_WORKER });
    assert.deepEqual(noTotal.verdict.failed, ['sizingEvidence']); assert.deepEqual(noTotal.sizing.rules.total, []);
    for (const [label, install] of [
        ['the real statement in another file of utils', { worker: REAL_REQUEST, utils: '# nothing\n', extra: { 'utils/other.py': inMeasure(REAL_STATEMENT) } }],
        ['the real statement in the worker file', { worker: REAL_REQUEST, utils: '# nothing\n', gpuWorker: inMeasure(REAL_STATEMENT) }],
        ['another class with the same method', { worker: REAL_REQUEST, utils: ['class Other:', '    def measure(self):', `        ${REAL_STATEMENT}`, ''].join('\n') }],
        ['the same class, another method', { worker: REAL_REQUEST, utils: ['class MemorySnapshot:', '    def other(self):', `        ${REAL_STATEMENT}`, ''].join('\n') }],
        ['a function named measure outside the class', { worker: REAL_REQUEST, utils: ['def measure(self):', `    ${REAL_STATEMENT}`, ''].join('\n') }],
        ['module level', { worker: REAL_REQUEST, utils: `${REAL_STATEMENT}\n` }],
        ['a nested class and method of another name', { worker: REAL_REQUEST, utils: ['class MemorySnapshot:', '    def measure(self):', '        def inner():', `            ${REAL_STATEMENT}`, '        return inner', ''].join('\n') }],
    ]) {
        const { extra, ...sizing } = install;
        const evidence = await sizingOutcome(t, sizing, extra);
        assert.deepEqual(evidence.verdict.failed, ['sizingEvidence'], label);
        assert.deepEqual(evidence.sizing.rules.total, [], label);
    }
    // The request statement counts only in v1/worker/utils.py.
    for (const [label, install] of [
        ['the request in the worker file', { worker: '# nothing\n', utils: REAL_MEM_UTILS, gpuWorker: REAL_REQUEST }],
        ['the request in another file of v1/worker', { worker: '# nothing\n', utils: REAL_MEM_UTILS }],
    ]) {
        const evidence = await sizingOutcome(t, install, label.includes('another file') ? { 'v1/worker/other.py': REAL_REQUEST } : undefined);
        assert.deepEqual(evidence.verdict.failed, ['sizingEvidence'], label);
        assert.deepEqual(evidence.sizing.rules.request, [], label);
        assert.ok(evidence.sizing.verdict.reasons.some((entry) => /no statement computes the requested memory as the device total times gpu_memory_utilization in v1\/worker\/utils\.py/.test(entry)), `${label}: ${JSON.stringify(evidence.sizing.verdict.reasons)}`);
        assert.equal(evidence.sizing.rules.total.length, 1, `${label}: the total is still found`);
    }
    // Exact-version gating and executable-only matching are unchanged around the anchored rule.
    const other = await calibrate(t, { install: { version: '0.29.0', sizing: REAL_SIZING } });
    assert.ok(other.report.evidence.verdict.failed.includes('sizingEvidence') && other.report.evidence.verdict.failed.includes('identityMatchesLock'));
    void A;
});

test('CAL.the-enclosing-python-scope-is-read-from-the-indentation-of-the-executable-code', () => {
    const code = stripPythonNonCode(['class A:', '    def one(self):', '        if x:', '            target = 1', '    def two(self):', '        return 2', 'def top():', '    y = 1', 'z = 3', ''].join('\n')).split('\n');
    assert.deepEqual(enclosingPythonScope(code, 3), { class: 'A', function: 'one' });
    assert.deepEqual(enclosingPythonScope(code, 5), { class: 'A', function: 'two' });
    assert.deepEqual(enclosingPythonScope(code, 7), { class: null, function: 'top' });
    assert.deepEqual(enclosingPythonScope(code, 8), { class: null, function: null });
    assert.deepEqual(enclosingPythonScope(stripPythonNonCode(['class B:', '    # def fake(self):', '    value = 1', ''].join('\n')).split('\n'), 2), { class: 'B', function: null });
    // A nested function defined earlier in the method does not change which method a later line is in.
    const nested = stripPythonNonCode(['class N:', '    def outer(self):', '        def inner():', '            pass', '        target = 1', ''].join('\n')).split('\n');
    assert.deepEqual(enclosingPythonScope(nested, 4), { class: 'N', function: 'outer' });
    // A class body inside a function has the class, and no method of its own.
    const local = stripPythonNonCode(['def factory():', '    class Local:', '        value = 1', ''].join('\n')).split('\n');
    assert.deepEqual(enclosingPythonScope(local, 2), { class: 'Local', function: null });
    assert.deepEqual(enclosingPythonScope(stripPythonNonCode(['class C:', '    s = """', '    def fake(self):', '    """', '    value = 2', ''].join('\n')).split('\n'), 4), { class: 'C', function: null });
});

test('CAL.only-the-accelerator-total-in-its-reviewed-position-and-executable-code-is-a-sizing-statement', async (t) => {
    const control = await sizingOutcome(t, REAL_SIZING);
    assert.equal(control.verdict.qualifiable, true);
    const A = 'torch.accelerator.get_memory_info(device)';
    // Every case sits in the reviewed place (MemorySnapshot.measure of utils/mem_utils.py), so that only its own text decides.
    const negatives = [
        ['commented out', inMeasure(`#   ${REAL_STATEMENT}`, 'self.free_memory, self.total_memory = configured_cache_limit, configured_cache_limit')],
        ['a trailing comment', inMeasure(`x = 1  # ${REAL_STATEMENT}`)],
        ['a docstring', inMeasure('"""', REAL_STATEMENT, '"""')],
        ['a string literal', inMeasure(`NOTE = "${REAL_STATEMENT}"`)],
        ['a triple-quoted block', inMeasure("NOTE = '''", REAL_STATEMENT, "'''")],
        ['absent', inMeasure('# see torch.accelerator.get_memory_info for the free and total memory', 'value = total_memory')],
        ['the total from the device properties', inMeasure(`self.free_memory = ${A}[0]`, 'self.total_memory = torch.cuda.get_device_properties(device).total_memory')],
        ['the total from a constant', inMeasure('self.free_memory, self.total_memory = 0, 6 * 1024 ** 3')],
        ['the unpacking in the wrong order', inMeasure(`self.total_memory, self.free_memory = ${A}`)],
        ['the free value taken as the total', inMeasure(`self.total_memory = ${A}[0]`)],
        ['the whole pair bound to the total', inMeasure(`self.total_memory = ${A}`)],
        ['another module\'s function', inMeasure('self.free_memory, self.total_memory = some_cache.accelerator.get_memory_info(device)')],
        ['the call of another prefix', inMeasure('self.free_memory, self.total_memory = my_torch.accelerator.get_memory_info(device)')],
        ['a second value that is not a total', inMeasure(`self.free_memory, self.used_memory = ${A}`)],
    ];
    for (const [label, utils] of negatives) {
        const evidence = await sizingOutcome(t, { worker: REAL_REQUEST, utils });
        assert.equal(evidence.verdict.qualifiable, false, `${label}: not qualifiable`);
        assert.deepEqual(evidence.verdict.failed, ['sizingEvidence'], label);
        assert.deepEqual(evidence.sizing.rules.total, [], `${label}: no total statement matched`);
        assert.equal(evidence.sizing.rules.request.length, 1, `${label}: the request statement is still found`);
        assert.ok(evidence.sizing.verdict.reasons.some((entry) => /no statement reads the device total from mem_get_info or torch\.accelerator\.get_memory_info in utils\/mem_utils\.py \(MemorySnapshot\.measure\)/.test(entry)), `${label}: ${JSON.stringify(evidence.sizing.verdict.reasons)}`);
    }
    // The second value of the returned pair, bound by index or by an unpacking that ignores the free value, is the same statement.
    for (const [label, utils] of [
        ['an index', inMeasure(`self.total_memory = ${A}[1]`)],
        ['an ignored free value', inMeasure('free, total = torch.accelerator.get_memory_info(0)')],
        ['an underscore', inMeasure('_, total_bytes = torch.accelerator.get_memory_info(device)')],
    ]) {
        const evidence = await sizingOutcome(t, { worker: REAL_REQUEST, utils });
        assert.equal(evidence.verdict.qualifiable, true, `${label}: ${JSON.stringify(evidence.verdict)}`);
        assert.equal(evidence.sizing.rules.total.length, 1, label);
        assert.ok(evidence.sizing.lines.some((line) => /get_memory_info/.test(line.text)), `${label}: the accelerator statement is listed as sizing evidence`);
    }
    // The earlier wheels' statement stays accepted in the same place, and the exact-version gating is unchanged: another version never qualifies.
    assert.equal((await sizingOutcome(t, { worker: REAL_REQUEST, utils: inMeasure('self.free_memory, self.total_memory = torch.cuda.mem_get_info()') })).verdict.qualifiable, true);
    const other = await calibrate(t, { install: { version: '0.29.0', sizing: REAL_SIZING } });
    assert.ok(other.report.evidence.verdict.failed.includes('sizingEvidence') && other.report.evidence.verdict.failed.includes('identityMatchesLock'));
    assert.deepEqual(Object.keys(REVIEWED_SIZING), ['0.30.0']);
});

test('CAL.the-accelerator-api-is-measured-under-both-limits-and-recorded-raw-next-to-the-cuda-and-driver-views', async (t) => {
    assert.ok(TORCH_QUERY.includes('torch.accelerator.get_memory_info(0)'), 'the probe calls the exact API vLLM 0.30.0 calls');
    const queryLines = TORCH_QUERY.split('\n');
    assert.ok(queryLines.findIndex((line) => line.includes('torch.accelerator.get_memory_info')) > queryLines.findIndex((line) => line.includes('if out["cudaAvailable"]')), 'it runs only when CUDA is available');
    const { report, calls } = await calibrate(t);
    assert.equal(calls.filter((call) => call.args[1] === TORCH_QUERY).length, 2, 'the torch query, with the accelerator call, runs under both limits');
    const m = report.evidence.measurements;
    const free = USABLE - 200 * MIB;
    assert.deepEqual(m.acceleratorShare, { free, total: USABLE }); assert.deepEqual(m.acceleratorTight, { free, total: USABLE });
    // The raw (free, total) is in the torch document as well, next to torch.cuda's pair and the driver's.
    assert.deepEqual(m.torchShare.acceleratorMemoryInfo, { free, total: USABLE }); assert.deepEqual(m.torchTight.acceleratorMemoryInfo, { free, total: USABLE });
    assert.deepEqual(m.torchShare.memGetInfo, { free, total: USABLE }); assert.equal(m.ctypesShare.total, USABLE); assert.equal(m.ctypesTight.total, USABLE);
    for (const name of ['torchViewsAgree', 'acceleratorAgreesWithDriver']) assert.equal(report.evidence.verdict.checks[name], true, name);
    // The raw values are part of the digest: changing one changes it.
    const doctored = structuredClone(report.evidence); doctored.measurements.acceleratorTight.total += 1;
    assert.notEqual(evidenceDigest(doctored), report.evidence.evidenceDigest);
    // Under a share-following device both limits move it, and the accelerator follows (the denominator stays 'share').
    const share = (await calibrate(t, { views: shareViews })).report.evidence;
    assert.deepEqual([share.measurements.acceleratorShare.total, share.measurements.acceleratorTight.total], [5529 * MIB, 2048 * MIB]);
});

test('CAL.the-accelerator-view-must-agree-exactly-with-the-cuda-views-and-with-the-driver-under-each-limit', async (t) => {
    const TIGHT = '0=2048M';
    const cases = [
        ['the accelerator total one byte off under the saved limit', { torch: (l, total) => (l !== TIGHT ? { acceleratorMemoryInfo: { free: total - 200 * MIB, total: total - 1 } } : {}) }, ['torchViewsAgree']],
        ['the accelerator total one MiB off under the tighter limit', { torch: (l, total) => (l === TIGHT ? { acceleratorMemoryInfo: { free: total - 200 * MIB, total: total - MIB } } : {}) }, ['torchViewsAgree']],
        ['the accelerator total far from the driver under both limits (with every torch view following it)', {
            torch: (_l, total) => ({ memGetInfo: { free: total - 200 * MIB, total: total - 8 * MIB }, totalMemory: total - 8 * MIB, acceleratorMemoryInfo: { free: total - 200 * MIB, total: total - 8 * MIB } }),
        }, ['acceleratorAgreesWithDriver']],
        ['the accelerator free memory above its total', { torch: (_l, total) => ({ acceleratorMemoryInfo: { free: total + 1, total } }) }, ['acceleratorAgreesWithDriver']],
        ['the driver far from the accelerator under the tighter limit only', { ctypes: (l, total) => (l === TIGHT ? { total: total - 4 * MIB } : {}) }, ['acceleratorAgreesWithDriver']],
    ];
    for (const [label, options, expected] of cases) {
        const { report } = await calibrate(t, options);
        const evidence = report.evidence;
        assert.equal(evidence.verdict.qualifiable, false, label);
        for (const name of expected) { assert.ok(evidence.verdict.failed.includes(name), `${label}: ${evidence.verdict.failed}`); assert.equal(evidence.verdict.checks[name], false, label); }
        assert.equal(report.proposed, undefined, label); assert.throws(() => renderQualificationEntry(evidence), /Cannot render a qualification entry/, label);
    }
    // One isolated check: only the accelerator's free value is implausible, everything else holds.
    const isolated = (await calibrate(t, { torch: (_l, total) => ({ acceleratorMemoryInfo: { free: total + 1, total } }) })).report.evidence;
    assert.deepEqual(isolated.verdict.failed, ['acceleratorAgreesWithDriver']);
    const exactOnly = (await calibrate(t, { torch: (_l, total) => ({ acceleratorMemoryInfo: { free: total - 200 * MIB, total: total - 1 } }) })).report.evidence;
    assert.ok(exactOnly.verdict.failed.includes('torchViewsAgree') && !exactOnly.verdict.failed.includes('acceleratorAgreesWithDriver'), `a one-byte gap is inside the driver tolerance but not the exact torch rule: ${exactOnly.verdict.failed}`);
});

test('CAL.an-unavailable-accelerator-api-is-a-blocking-prerequisite-with-a-clear-message', async (t) => {
    const TIGHT = '0=2048M';
    const missing = { acceleratorMemoryInfo: null, acceleratorError: "AttributeError: module 'torch' has no attribute 'accelerator'" };
    for (const [label, torch, which] of [
        ['under both limits', missing, 'saved limit and the tighter limit'],
        ['under the tighter limit only', (l) => (l === TIGHT ? missing : {}), 'tighter limit'],
        ['under the saved limit only', (l) => (l !== TIGHT ? missing : {}), 'saved limit'],
    ]) {
        const { report } = await calibrate(t, { torch });
        assert.equal(report.ok, false, label);
        const blocker = report.blockers.find((entry) => entry.code === 'accelerator_memory_api_unavailable');
        assert.ok(blocker, `${label}: ${JSON.stringify(report.blockers)}`);
        assert.match(blocker.message, /torch\.accelerator\.get_memory_info\(0\), the call vLLM 0\.30\.0 reads its device total through, did not return the device memory/);
        assert.ok(blocker.message.includes(which) && blocker.message.includes("has no attribute 'accelerator'") && /cannot qualify/.test(blocker.message), `${label}: ${blocker.message}`);
        assert.equal(report.evidence.verdict.qualifiable, false, label);
        assert.ok(report.evidence.verdict.failed.includes('torchViewsAgree') && report.evidence.verdict.failed.includes('acceleratorAgreesWithDriver'), `${label}: ${report.evidence.verdict.failed}`);
        assert.equal(report.proposed, undefined, label);
        assert.equal(report.evidence.measurements.acceleratorShare === null || report.evidence.measurements.acceleratorTight === null, true, label);
    }
    // A result that is not a (free, total) pair of whole numbers with a positive total is no measurement either.
    for (const bad of [{ free: 0, total: 0 }, { free: 100, total: 0 }, { free: -1, total: USABLE }, { free: 1.5, total: USABLE }, { free: '100', total: USABLE }, { total: USABLE }]) {
        const { report } = await calibrate(t, { torch: { acceleratorMemoryInfo: bad } });
        assert.ok(report.blockers.some((entry) => entry.code === 'accelerator_memory_api_unavailable'), JSON.stringify(bad));
        assert.equal(report.evidence.verdict.qualifiable, false, JSON.stringify(bad));
    }
    // A CUDA that is unavailable is its own blocker, not this one.
    const noCuda = await calibrate(t, { torch: { cudaAvailable: false, memGetInfo: undefined, totalMemory: undefined, acceleratorMemoryInfo: null } });
    assert.equal(noCuda.report.blockers.some((entry) => entry.code === 'accelerator_memory_api_unavailable'), false);
});

// The probe's Python itself, run with python3 against a stub torch: the call is made with device index 0, its (free, total) pair is recorded
// under the documented keys, and a torch without the API is recorded with its error instead of failing the whole query.
test('CAL.the-torch-query-runs-and-records-the-accelerator-pair-or-its-error', (t) => {
    const root = scratch(t);
    const stub = path.join(root, 'stub'); fs.mkdirSync(path.join(stub, 'torch'), { recursive: true });
    fs.writeFileSync(path.join(stub, 'torch', '__init__.py'), [
        'import os', '__version__ = "2.13.0"', 'MiB = 1024 * 1024',
        'class _V:', '    cuda = "13.0"', 'version = _V()',
        'class _P:', '    total_memory = 6000 * MiB', '    name = "Stub GPU"', '    major = 8', '    minor = 6', '    multi_processor_count = 30',
        'class _Cuda:', '    @staticmethod', '    def is_available(): return True', '    @staticmethod', '    def mem_get_info(): return (5000 * MiB, 6000 * MiB)',
        '    @staticmethod', '    def get_device_properties(index): return _P()', '    @staticmethod', '    def get_arch_list(): return ["sm_86"]',
        'cuda = _Cuda()',
        'class _Accelerator:', '    @staticmethod', '    def get_memory_info(index=None):', '        assert index == 0, index', '        return (4800 * MiB, 6000 * MiB)',
        'if not os.environ.get("STUB_NO_ACCELERATOR"):', '    accelerator = _Accelerator()', '',
    ].join('\n'));
    const run = (extraEnv) => spawnSync('python3', ['-c', TORCH_QUERY], { env: { PATH: '/usr/bin:/bin', PYTHONPATH: stub, ...extraEnv }, encoding: 'utf8', timeout: 20_000 });
    const present = run({});
    assert.equal(present.status, 0, present.stderr);
    const value = JSON.parse(present.stdout.trim().split('\n').at(-1));
    assert.deepEqual(value.acceleratorMemoryInfo, { free: 4800 * MIB, total: 6000 * MIB });
    assert.deepEqual(value.memGetInfo, { free: 5000 * MIB, total: 6000 * MIB }); assert.equal(value.totalMemory, 6000 * MIB);
    assert.equal(Object.hasOwn(value, 'acceleratorError'), false);
    const absent = run({ STUB_NO_ACCELERATOR: '1' });
    assert.equal(absent.status, 0, absent.stderr);
    const missing = JSON.parse(absent.stdout.trim().split('\n').at(-1));
    assert.equal(missing.acceleratorMemoryInfo, null); assert.match(missing.acceleratorError, /^AttributeError: /);
    assert.deepEqual(missing.memGetInfo, { free: 5000 * MIB, total: 6000 * MIB }, 'the rest of the query is unaffected');
});
