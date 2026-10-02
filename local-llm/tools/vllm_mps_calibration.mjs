// LIVE-L3 for vLLM under a Ploinky MPS GPU share (Ploinky hardware-limits plan
// section 12.2): step 0, the prerequisite check, and stage 1, the calibration
// that reads the denominator the installed vLLM wheel actually sees, without
// launching a model. It runs inside the local-llm container as the agent's own
// user, started by the live runner with `container exec`:
//
//   node /code/tools/vllm_mps_calibration.mjs prerequisites --pins '<json>'
//   node /code/tools/vllm_mps_calibration.mjs calibrate [--host-nvml-bytes N]
//   node /code/tools/vllm_mps_calibration.mjs render --evidence FILE
//
// Every number that production derives is derived here by production's own
// functions, never by a copy: the lock entry and its digest (runnerLock.mjs,
// vllmRunnerLockDigest), the tuple (vllmMpsTuple), the GPU view (readGpu), the
// share (parseMpsBudget, effectiveGpu), the intended utilization (admitVllm) and
// the final argv (the vLLM adapter's buildLaunch). The probe prints one JSON
// document, changes nothing, and never modifies source: adding a reviewed tuple
// to REVIEWED_QUALIFICATIONS is a separate, later step (applyQualificationEntry).

import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { VLLM_USABLE_SHARE, admitVllm } from '../src/controller/admission.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { readGpu, readMemory } from '../src/controller/hardware.mjs';
import { effectiveGpu, parseMpsBudget } from '../src/controller/ploinkyBudget.mjs';
import { DEFAULT_RUNNER_LOCK, loadRunnerLocks } from '../src/controller/runnerLock.mjs';
import { createVllmMpsQualificationResolver, vllmRunnerLockDigest } from '../src/controller/vllmMpsQualification.mjs';
import { vllmMpsTuple, vllmRunner } from '../src/runners/vllm.mjs';

export const PREREQUISITE_SCHEMA = 'local-llm.vllm-prerequisites/v1';
export const CALIBRATION_SCHEMA = 'local-llm.vllm-mps-calibration/v1';
export const TUPLE_FIELDS = Object.freeze(['runnerLockDigest', 'driverVersion', 'gpuPciDeviceId', 'computeCapability', 'deviceTotalBytes']);

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const MODEL_ID = 'qwen3-4b-awq';
const NVIDIA_LIB_DIR = '/usr/local/nvidia/lib64';
const RUN_ROOT = '/opt/runners';
const UV = '/usr/local/bin/uv';
const PYTHON = '/usr/bin/python3';
// The lowest NVIDIA driver (major) that runs a CUDA runtime of this major version.
const MIN_DRIVER_MAJOR = Object.freeze({ 12: 525, 13: 580 });
// Estimates, named as such in the evidence: the verified cache holds the wheels
// (plus 5 %, as the installer's own free-space check), the runnable copy is built
// from a staged copy of them and then unpacked into a virtual environment.
const CACHE_RESERVE = 1.05;
const RUNNABLE_FACTOR = 3.5;
// A pinned memory limit may move torch's view by at most this much and still be
// "independent of the limit"; CUDA's own context reservation is far below it.
const INDEPENDENCE_TOLERANCE_BYTES = 64 * MIB;
// vLLM requests gpu_memory_utilization x torch's total: that must fit the pinned
// limit with room for the CUDA context (G1 measured about 148 MiB on this device).
const CONTEXT_MARGIN_BYTES = 256 * MIB;
// The torch total is CUDA's usable share of the device (about 94 %, admission's
// VLLM_USABLE_SHARE); it must agree with that to within this fraction of the device.
const USABLE_TOLERANCE = 0.03;
const OUTPUT_LIMIT = 48 * 1024;
const SIZING_PATTERN = /gpu_memory_utilization|mem_get_info|requested_memory|total_memory/;
const SIZING_DIRECTORIES = Object.freeze(['v1/worker', 'utils']);
const MAX_SIZING_FILES = 120;
const MAX_SIZING_LINES = 40;

// ---------------------------------------------------------------------------
// Canonical JSON and the evidence digest.

export function canonicalJson(value) {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
    if (value && typeof value === 'object') {
        return `{${Object.keys(value).filter((key) => value[key] !== undefined).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
    }
    return JSON.stringify(value ?? null);
}

/** The sha256 of the evidence document without its own `evidenceDigest` field. */
export function evidenceDigest(document) {
    const { evidenceDigest: _own, ...rest } = document || {};
    return createHash('sha256').update(canonicalJson(rest)).digest('hex');
}

// ---------------------------------------------------------------------------
// Bounded process runner and file readers (injectable for tests).

export function runBounded(file, args, { env = {}, timeoutMs = 120_000, maxBuffer = 64 * 1024, execFileImpl = execFile } = {}) {
    return new Promise((resolve) => {
        execFileImpl(file, args, { env, timeout: timeoutMs, killSignal: 'SIGKILL', maxBuffer, encoding: 'utf8', shell: false }, (error, stdout, stderr) => {
            resolve({
                ok: !error, status: error ? (Number.isInteger(error.code) ? error.code : null) : 0,
                stdout: String(stdout || ''), stderr: String(stderr || '').slice(-2000), error: error ? String(error.code || error.message).slice(0, 120) : null,
                timedOut: Boolean(error?.killed),
            });
        });
    });
}

const parseJsonLine = (text) => {
    try { return JSON.parse(String(text).trim().split('\n').filter(Boolean).at(-1) || ''); } catch { return null; }
};

// ---------------------------------------------------------------------------
// Wheel names: platform, CPython tag and the CUDA runtime a lock pins.

/** The CPython tag of a wheel file name: { kind: 'abi3'|'cpython'|'any', minor } */
export function wheelPythonTag(name) {
    const parts = String(name).replace(/\.whl$/, '').split('-');
    if (parts.length < 5) return null;
    const python = parts.at(-3); const abi = parts.at(-2); const platform = parts.at(-1);
    const minor = /^cp3(\d+)$/.exec(python)?.[1];
    if (abi === 'none' && /^py3/.test(python)) return { kind: 'any', minor: null, platform };
    if (abi === 'abi3' && minor) return { kind: 'abi3', minor: Number(minor), platform };
    if (minor && abi === `cp3${minor}`) return { kind: 'cpython', minor: Number(minor), platform };
    return { kind: 'other', minor: null, platform };
}

const wheelPlatformOk = (platform) => platform === 'any' || /(?:^|\.)(?:manylinux[^.]*_x86_64|linux_x86_64|manylinux2014_x86_64)$/.test(platform) || /x86_64/.test(platform);

export function lockWheelFacts(entry) {
    const wheels = entry.files.filter((file) => file.name.endsWith('.whl'));
    const torch = wheels.find((file) => /^torch-\d/.test(file.name)) || null;
    const cudaWheels = wheels.filter((file) => /^nvidia_(?:cuda_runtime|cuda_nvrtc|cublas|cudnn)/.test(file.name));
    const nvrtc = wheels.find((file) => /^nvidia_cuda_nvrtc-\d+\.\d+/.test(file.name)) || cudaWheels.find((file) => /-\d+\.\d+/.test(file.name)) || null;
    const cuda = nvrtc ? /-(\d+)\.(\d+)/.exec(nvrtc.name) : null;
    const lines = entry.files.map((file) => `${file.name}:${file.size}:${file.sha256}`).sort();
    const largest = [...entry.files].sort((a, b) => b.size - a.size).slice(0, 12).map((file) => ({ name: file.name, size: file.size, sha256: file.sha256 }));
    return {
        wheelCount: wheels.length, otherFiles: entry.files.length - wheels.length,
        torchWheel: torch ? { name: torch.name, size: torch.size, sha256: torch.sha256 } : null,
        cudaWheelCount: cudaWheels.length, cudaRuntime: cuda ? { major: Number(cuda[1]), minor: Number(cuda[2]), wheel: nvrtc.name } : null,
        fileListDigest: createHash('sha256').update(lines.join('\n')).digest('hex'), largest,
        hosts: [...new Set(entry.files.map((file) => new URL(file.url).hostname))].sort(),
    };
}

// ---------------------------------------------------------------------------
// Step 0: the prerequisite check. A missing prerequisite is a blocker with the
// exact evidence; nothing here ever reports success for an install that cannot
// work, and nothing installs.

function diskFacts(fsApi, statfs, target) {
    try {
        const stat = fsApi.statSync(target);
        const info = statfs(target);
        return { path: target, dev: String(stat.dev), freeBytes: Number(info.bavail) * Number(info.bsize), totalBytes: Number(info.blocks) * Number(info.bsize) };
    } catch (error) {
        return { path: target, dev: null, freeBytes: null, totalBytes: null, error: String(error?.code || error?.message).slice(0, 80) };
    }
}

export async function prerequisiteReport({
    pins = null, env = process.env, arch = process.arch, fsApi = fs, statfs = (target) => fs.statfsSync(target), run = runBounded,
    imageLockFile = DEFAULT_RUNNER_LOCK, agentLockFile = undefined, dataDir = '/data', runRoot = RUN_ROOT,
    readGpuImpl = readGpu, catalog = null, uv = UV, python = PYTHON,
} = {}) {
    const blockers = [];
    const block = (code, message, evidence = {}) => blockers.push({ code, message, evidence });
    const facts = { arch, node: process.versions.node, lock: null, python: null, driverVersion: null, tools: {}, disk: null };

    // The image's lock, by its own validator (the same one the controller uses).
    let locks = null;
    let lockBytes = null;
    try {
        lockBytes = fsApi.readFileSync(imageLockFile);
        locks = loadRunnerLocks({ image: imageLockFile, ...(agentLockFile === undefined ? {} : { agent: agentLockFile }), fsApi });
    } catch (error) {
        block('lock_unreadable', `The image's runner lock ${imageLockFile} cannot be read or is invalid: ${String(error?.message || error).slice(0, 200)}`);
    }
    const entry = locks?.runners?.vllm || null;
    facts.lock = { file: imageLockFile, sha256: lockBytes ? createHash('sha256').update(lockBytes).digest('hex') : null, runners: locks ? Object.keys(locks.runners).sort() : [], origin: locks?.origin?.vllm ?? null };
    if (locks && !entry) block('vllm_entry_missing', `The image's runner lock has no vLLM entry (it lists: ${facts.lock.runners.join(', ') || 'nothing'}), so vLLM cannot be installed from this image.`, { runners: facts.lock.runners });
    if (entry && entry.kind !== 'python') block('vllm_entry_not_python', `The vLLM lock entry is of kind ${entry.kind}, not python.`, { kind: entry.kind });

    let cuda = null;
    if (entry) {
        const wheels = lockWheelFacts(entry);
        const runnerLockDigest = vllmRunnerLockDigest(entry);
        facts.lock.vllm = {
            version: entry.version, kind: entry.kind, entryDigest: entry.digest, runnerLockDigest, files: entry.files.length, downloadBytes: entry.totalBytes,
            licence: { name: entry.licence.name, requiresAcceptance: entry.licence.requiresAcceptance },
            distributions: entry.check.distributions, ...wheels,
        };
        cuda = wheels.cudaRuntime;
        if (!runnerLockDigest) block('lock_digest_unavailable', 'The production lock digest cannot be derived for the vLLM entry.', {});
        if (entry.licence.requiresAcceptance) block('licence_acceptance_required', 'The vLLM licence needs an acceptance this unattended check never gives.', { licence: entry.licence.name });
        if (!wheels.torchWheel || !cuda) block('no_cuda_wheels', 'The vLLM entry pins no CUDA-enabled PyTorch stack (a torch wheel and the NVIDIA CUDA runtime wheels).', { torch: wheels.torchWheel?.name ?? null, cudaWheels: wheels.cudaWheelCount });
    }

    // Platform: this container's architecture and every wheel's platform tag.
    if (arch !== 'x64') block('unsupported_platform', `This container runs on ${arch}; the vLLM entry's wheels are for linux/amd64.`, { arch });
    if (entry) {
        const wrongPlatform = entry.files.filter((file) => file.name.endsWith('.whl') && !wheelPlatformOk(wheelPythonTag(file.name)?.platform ?? ''));
        if (wrongPlatform.length) block('wheel_platform', `${wrongPlatform.length} locked wheels are not for linux x86_64.`, { examples: wrongPlatform.slice(0, 5).map((file) => file.name) });
    }

    // The interpreter the installer builds the environment from.
    const version = await run(python, ['-c', 'import sys,sysconfig,os;print(sys.version_info.major,sys.version_info.minor,sysconfig.get_paths()["include"])'], { env: { PATH: '/usr/bin:/bin' }, timeoutMs: 20_000 });
    const parsedVersion = version.ok ? /^(\d+) (\d+) (\S+)\s*$/.exec(version.stdout.trim()) : null;
    if (!parsedVersion) block('python_unavailable', `${python} cannot report its version: ${version.error || version.stderr.slice(0, 120)}`);
    else {
        facts.python = { executable: python, major: Number(parsedVersion[1]), minor: Number(parsedVersion[2]), include: parsedVersion[3] };
        if (entry) {
            const incompatible = entry.files.filter((file) => file.name.endsWith('.whl')).map((file) => ({ file, tag: wheelPythonTag(file.name) }))
                .filter(({ tag }) => tag && ((tag.kind === 'cpython' && tag.minor !== facts.python.minor) || (tag.kind === 'abi3' && tag.minor > facts.python.minor)));
            if (incompatible.length) block('python_abi', `${incompatible.length} locked wheels do not fit Python ${facts.python.major}.${facts.python.minor}.`, { examples: incompatible.slice(0, 5).map(({ file }) => file.name) });
        }
    }

    // The GPU driver against the CUDA runtime the wheels carry.
    let gpu = null;
    try { gpu = await readGpuImpl({ env }); } catch (error) { block('gpu_unreadable', `The GPU cannot be read from this container: ${String(error?.message || error).slice(0, 160)}`); }
    if (gpu && !gpu.available) block('gpu_unavailable', `No usable GPU is visible to this agent: ${String(gpu.reason || gpu.state).slice(0, 200)}`, { state: gpu.state ?? null });
    if (gpu?.available) {
        facts.driverVersion = gpu.driverVersion;
        facts.gpu = { name: gpu.name, memoryModel: gpu.memoryModel, totalBytes: gpu.totalBytes, device: gpu.device ?? null };
        const major = Number(String(gpu.driverVersion).split('.')[0]);
        const needed = cuda ? MIN_DRIVER_MAJOR[cuda.major] : null;
        if (cuda && needed === undefined) block('cuda_unknown', `The wheels carry CUDA ${cuda.major}.${cuda.minor}, whose driver requirement this check does not know.`, { cuda });
        else if (cuda && !(major >= needed)) block('driver_too_old', `Driver ${gpu.driverVersion} is older than the ${needed} that CUDA ${cuda.major}.${cuda.minor} wheels need.`, { driverVersion: gpu.driverVersion, needed });
        if (gpu.memoryModel !== 'dedicated') block('memory_model', `The GPU's memory model is ${gpu.memoryModel}, not dedicated.`, { memoryModel: gpu.memoryModel });
    }

    // Tools the wheels need at run time: uv builds the environment, Triton compiles a C launcher.
    const present = (file) => { try { fsApi.accessSync(file, fs.constants.X_OK); return true; } catch { return false; } };
    facts.tools = { uv: present(uv), gcc: present('/usr/bin/gcc'), pythonHeaders: false };
    if (facts.python) { try { fsApi.accessSync(path.join(facts.python.include, 'Python.h')); facts.tools.pythonHeaders = true; } catch { /* reported below */ } }
    const missingTools = Object.entries({ [uv]: facts.tools.uv, '/usr/bin/gcc': facts.tools.gcc, 'Python.h': facts.tools.pythonHeaders }).filter(([, ok]) => !ok).map(([name]) => name);
    if (missingTools.length) block('toolchain_missing', `The image lacks what installing and running vLLM needs: ${missingTools.join(', ')}.`, { missing: missingTools });

    // Free disk: the verified cache and the model under /data, the runnable copy under /opt/runners.
    if (entry) {
        const models = catalog || safeCatalog();
        const model = models?.find((value) => value.id === MODEL_ID);
        const modelBytes = model?.sources?.hf?.size ?? 0;
        if (!modelBytes) block('model_unpinned', `The catalog has no pinned ${MODEL_ID} snapshot.`);
        const data = diskFacts(fsApi, statfs, dataDir); const runnable = diskFacts(fsApi, statfs, fsApi.existsSync(runRoot) ? runRoot : path.dirname(runRoot));
        const need = {
            dataBytes: Math.ceil((entry.totalBytes + modelBytes) * CACHE_RESERVE), runnableBytes: Math.ceil(entry.totalBytes * RUNNABLE_FACTOR),
            basis: `the cache holds the ${entry.totalBytes} locked bytes and the ${modelBytes}-byte model with 5 % reserve; the runnable copy is estimated at ${RUNNABLE_FACTOR} times the locked bytes`,
        };
        const shared = data.dev !== null && data.dev === runnable.dev;
        facts.disk = { data, runnable, need, sameFilesystem: shared };
        const short = [];
        if (shared) { if (data.freeBytes < need.dataBytes + need.runnableBytes) short.push(`${dataDir} and ${runRoot} share one filesystem with ${data.freeBytes} bytes free; ${need.dataBytes + need.runnableBytes} are needed`); }
        else {
            if (!(data.freeBytes >= need.dataBytes)) short.push(`${dataDir} has ${data.freeBytes} bytes free; ${need.dataBytes} are needed`);
            if (!(runnable.freeBytes >= need.runnableBytes)) short.push(`${runnable.path} has ${runnable.freeBytes} bytes free; ${need.runnableBytes} are needed`);
        }
        if (short.length) block('insufficient_disk', `Not enough free disk to install vLLM and hold the model: ${short.join('; ')}.`, { data, runnable, need });
    }

    // The operator's pins of the image's lock entry: nothing unpinned is ever installed.
    if (!pins) block('pins_missing', 'The run supplied no pins for the vLLM lock entry, so an unpinned install is refused.');
    else if (entry) {
        const actual = { version: entry.version, runnerLockDigest: facts.lock.vllm.runnerLockDigest, files: entry.files.length, downloadBytes: entry.totalBytes };
        const differs = Object.keys(actual).filter((key) => pins[key] !== actual[key]);
        if (differs.length) block('pin_mismatch', `The image's vLLM lock entry differs from the pinned one in ${differs.join(', ')}.`, { expected: Object.fromEntries(differs.map((key) => [key, pins[key] ?? null])), actual: Object.fromEntries(differs.map((key) => [key, actual[key]])) });
    }
    return { schema: PREREQUISITE_SCHEMA, ok: blockers.length === 0, blockers, facts };
}

function safeCatalog() {
    try { return loadSeedCatalog(); } catch { return null; }
}

// ---------------------------------------------------------------------------
// Stage 1: the calibration.

// Bounded in-client queries. Both print one JSON line. torch's two total-memory
// views come from the same process; ctypes asks the driver directly (the same
// cuMemGetInfo the G1 CUDA probe reports).
export const TORCH_QUERY = [
    'import json,os,sys',
    'import torch',
    'out={"python":sys.version.split()[0],"torch":torch.__version__,"torchCuda":torch.version.cuda,"cudaAvailable":bool(torch.cuda.is_available())}',
    'if out["cudaAvailable"]:',
    '    free,total=torch.cuda.mem_get_info()',
    '    p=torch.cuda.get_device_properties(0)',
    '    out["memGetInfo"]={"free":int(free),"total":int(total)}',
    '    out["totalMemory"]=int(p.total_memory)',
    '    out["name"]=p.name',
    '    out["capability"]=[int(p.major),int(p.minor)]',
    '    out["multiProcessorCount"]=int(p.multi_processor_count)',
    '    out["archList"]=list(torch.cuda.get_arch_list())',
    'out["mps"]={k:os.environ.get(k) for k in ("CUDA_MPS_PIPE_DIRECTORY","CUDA_MPS_ACTIVE_THREAD_PERCENTAGE","CUDA_MPS_PINNED_DEVICE_MEM_LIMIT")}',
    'print(json.dumps(out))',
].join('\n');
export const CTYPES_QUERY = [
    'import ctypes as c,json,os',
    'cuda=c.CDLL("/usr/local/nvidia/lib64/libcuda.so.1")',
    'assert cuda.cuInit(0)==0',
    'dev=c.c_int()',
    'assert cuda.cuDeviceGet(c.byref(dev),0)==0',
    'ctx=c.c_void_p()',
    'assert cuda.cuDevicePrimaryCtxRetain(c.byref(ctx),dev)==0',
    'assert cuda.cuCtxSetCurrent(ctx)==0',
    'free=c.c_size_t()',
    'total=c.c_size_t()',
    'assert cuda.cuMemGetInfo_v2(c.byref(free),c.byref(total))==0',
    'print(json.dumps({"free":free.value,"total":total.value}))',
].join('\n');

/** The source lines that size vLLM's request against the device, with file and line (bounded). */
export function scanSizingSource({ fsApi = fs, root }) {
    const lines = []; let scanned = 0; let truncated = false;
    for (const directory of SIZING_DIRECTORIES) {
        let names = [];
        try { names = fsApi.readdirSync(path.join(root, directory)).filter((name) => name.endsWith('.py')).sort(); } catch { names = []; }
        for (const name of names) {
            if (scanned >= MAX_SIZING_FILES) { truncated = true; break; }
            scanned += 1;
            let text;
            try { text = fsApi.readFileSync(path.join(root, directory, name), 'utf8'); } catch { continue; }
            text.split('\n').forEach((line, index) => {
                if (!SIZING_PATTERN.test(line)) return;
                if (lines.length >= MAX_SIZING_LINES) { truncated = true; return; }
                lines.push({ file: `${directory}/${name}`, line: index + 1, text: line.trim().slice(0, 160) });
            });
        }
    }
    return { root, filesScanned: scanned, lines, truncated };
}

function vllmPackage({ fsApi = fs, runnerDir }) {
    const lib = path.join(runnerDir, 'venv', 'lib');
    try {
        for (const python of fsApi.readdirSync(lib).filter((name) => /^python3\.\d+$/.test(name)).sort()) {
            const site = path.join(lib, python, 'site-packages');
            if (fsApi.existsSync(path.join(site, 'vllm'))) {
                const dist = fsApi.readdirSync(site).find((name) => /^vllm-[0-9][^/]*\.dist-info$/.test(name)) || null;
                let version = null;
                if (dist) { try { version = /^Version:\s*(\S+)/m.exec(fsApi.readFileSync(path.join(site, dist, 'METADATA'), 'utf8'))?.[1] ?? null; } catch { version = null; } }
                return { site, root: path.join(site, 'vllm'), dist, version };
            }
        }
    } catch { /* not installed */ }
    return null;
}

/** Classify the denominator the installed wheel sees from the two measured views. */
export function classifyDenominator({ torchShare, torchTight, ctypesShare, ctypesTight, nvmlBytes, shareBytes, tightBytes }) {
    const facts = { views: { torchShare, torchTight, ctypesShare, ctypesTight }, nvmlBytes, shareBytes, tightBytes };
    if (![torchShare, torchTight, ctypesShare, ctypesTight, nvmlBytes, shareBytes, tightBytes].every((value) => Number.isSafeInteger(value) && value > 0)) {
        return { denominator: 'unknown', reason: 'a required memory view is missing or malformed', ...facts };
    }
    if (Math.abs(torchShare - ctypesShare) > MIB || Math.abs(torchTight - ctypesTight) > MIB) {
        return { denominator: 'unknown', reason: 'torch and the CUDA driver disagree about the total', ...facts };
    }
    const moved = Math.abs(torchShare - torchTight) > INDEPENDENCE_TOLERANCE_BYTES;
    if (!moved && torchShare >= 0.85 * nvmlBytes && torchShare <= nvmlBytes) {
        return { denominator: 'physical-device', reason: 'the total does not follow the pinned limit and is the device\'s usable memory', ...facts };
    }
    if (moved && torchShare <= shareBytes && torchTight <= tightBytes) return { denominator: 'share', reason: 'the total follows the pinned limit', ...facts };
    return { denominator: 'unknown', reason: 'the total neither follows the pinned limit nor matches the physical device', ...facts };
}

const envOf = (launchEnv, budget, extra = {}) => {
    const { VLLM_API_KEY: _key, ...rest } = launchEnv;
    return { PATH: '/usr/bin:/bin', HOME: '/tmp', ...rest, ...budget.environment, ...extra };
};

export async function calibrationReport({
    env = process.env, fsApi = fs, run = runBounded, readGpuImpl = readGpu, readMemoryImpl = readMemory, locksImpl = loadRunnerLocks,
    hostNvmlBytes = null, runRoot = RUN_ROOT, now = () => new Date(), catalog = null,
} = {}) {
    const blockers = [];
    const block = (code, message, evidence = {}) => blockers.push({ code, message, evidence });
    const result = (document = null) => ({ schema: CALIBRATION_SCHEMA, ok: blockers.length === 0 && Boolean(document), blockers, evidence: document });

    const budget = parseMpsBudget(env);
    if (budget.state !== 'known') { block('no_mps_share', 'This agent has no complete Ploinky GPU share, so there is nothing to calibrate under MPS.', { state: budget.state, reason: budget.reason ?? null }); return result(); }
    let raw;
    try { raw = await readGpuImpl({ env }); } catch (error) { block('gpu_unreadable', `The GPU cannot be read: ${String(error?.message || error).slice(0, 160)}`); return result(); }
    if (!raw?.available || raw.memoryModel !== 'dedicated' || !raw.device?.pciDeviceId || !raw.device?.computeCapability) {
        block('gpu_unusable', 'The GPU is unavailable, not dedicated-memory, or its PCI device id or compute capability is unknown.', { state: raw?.state ?? null, memoryModel: raw?.memoryModel ?? null, device: raw?.device ?? null });
        return result();
    }
    let locks;
    try { locks = locksImpl({ fsApi }); } catch (error) { block('lock_unreadable', `The runner lock cannot be read: ${String(error?.message || error).slice(0, 160)}`); return result(); }
    const entry = locks.runners?.vllm;
    if (!entry) { block('vllm_entry_missing', 'The runner lock has no vLLM entry.'); return result(); }

    // The tuple: production's own function over production's own readings.
    const tuple = vllmMpsTuple({ gpu: raw, runnerLockEntry: entry });
    const self = { ...tuple, denominator: 'physical-device', evidenceDigest: '0'.repeat(64) };
    if (!createVllmMpsQualificationResolver(() => [self])(tuple).qualified) {
        block('tuple_invalid', 'Production rejects the tuple read from this host as malformed or incomplete.', { tuple });
        return result();
    }

    const runnerDir = path.join(runRoot, 'vllm', entry.version);
    const python = path.join(runnerDir, 'venv', 'bin', 'python');
    const installed = vllmPackage({ fsApi, runnerDir });
    if (!fsApi.existsSync(python) || !installed) { block('vllm_not_installed', `vLLM ${entry.version} is not installed under ${runnerDir}; install it through the Runners path first.`, { python }); return result(); }
    let ready = null;
    try { ready = JSON.parse(String(fsApi.readFileSync(path.join(runnerDir, '.ready.json'), 'utf8')).slice(0, 4000)); } catch { ready = null; }

    // The model and the intended budget: production's admission over the share.
    const models = catalog || loadSeedCatalog();
    const model = models.find((value) => value.id === MODEL_ID);
    const source = model?.sources?.hf;
    if (!model || !source) { block('model_unpinned', `The catalog has no pinned ${MODEL_ID} snapshot.`); return result(); }
    const params = vllmRunner.normalizeParams({}, { model, profile: 'dedicated' });
    const gpu = effectiveGpu(raw, budget);
    const memory = readMemoryImpl();
    const admission = admitVllm({ model, source, params, gpu, memory, disk: { freeBytes: Number.MAX_SAFE_INTEGER, totalBytes: Number.MAX_SAFE_INTEGER }, remainingDownloadBytes: 0 });
    const utilization = admission.estimate?.gpuMemoryUtilization;
    const snapshotPath = `/data/models/hf/${source.repo}/${source.commit}`;
    // Production's launch builder over the admitted utilization. When admission does not
    // fit the share there is no valid launch: that is evidence (no argv), never an exception.
    let launch = null;
    let launchError = null;
    try {
        launch = vllmRunner.buildLaunch({
            runnerDir, artifactPath: snapshotPath, params, port: 18082, apiKey: 'k'.repeat(43), model, gpuMemoryUtilization: utilization,
            cacheDir: path.join(runRoot, '.cache', 'vllm'), profile: 'dedicated',
        });
    } catch (error) { launchError = String(error?.message || error).slice(0, 200); }
    const launchEnv = launch?.env ?? {};

    // The in-client queries, as the runner would see them, and once more under a tighter limit.
    const shareMiB = Math.floor(budget.gpuShare.vramBytes / MIB);
    const tightMiB = Math.max(512, Math.min(2048, Math.floor(shareMiB / 2)));
    const tightEnv = envOf(launchEnv, budget, { CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: `0=${tightMiB}M` });
    const shareEnv = envOf(launchEnv, budget);
    const query = async (file, script, queryEnv) => {
        const reply = await run(file, ['-c', script], { env: queryEnv, timeoutMs: 180_000 });
        const value = reply.ok ? parseJsonLine(reply.stdout) : null;
        return { ok: Boolean(value), value, status: reply.status, error: value ? null : (reply.error || reply.stderr.slice(-300)) };
    };
    const [torchShare, torchTight, ctypesShare, ctypesTight] = [
        await query(python, TORCH_QUERY, shareEnv), await query(python, TORCH_QUERY, tightEnv),
        await query('/usr/bin/python3', CTYPES_QUERY, shareEnv), await query('/usr/bin/python3', CTYPES_QUERY, tightEnv),
    ];
    for (const [name, reply] of Object.entries({ torchShare, torchTight, ctypesShare, ctypesTight })) {
        if (!reply.ok) block('query_failed', `The bounded ${name} query did not return a document: ${String(reply.error).slice(0, 200)}`, { query: name, status: reply.status });
    }
    const sizing = scanSizingSource({ fsApi, root: installed.root });

    const torchTotal = torchShare.value?.memGetInfo?.total ?? null;
    const classification = classifyDenominator({
        torchShare: torchTotal, torchTight: torchTight.value?.memGetInfo?.total ?? null, ctypesShare: ctypesShare.value?.total ?? null,
        ctypesTight: ctypesTight.value?.total ?? null, nvmlBytes: raw.totalBytes, shareBytes: budget.gpuShare.vramBytes, tightBytes: tightMiB * MIB,
    });
    const capability = torchShare.value?.capability || [];
    const arch = `sm_${capability[0]}${capability[1]}`;
    const archList = torchShare.value?.archList || [];
    const archSupported = archList.includes(arch) || archList.some((value) => /^compute_\d+$/.test(value) && Number(value.slice(8)) <= Number(`${capability[0]}${capability[1]}`));
    const intendedBytes = Number.isFinite(utilization) ? Math.round(utilization * raw.totalBytes * VLLM_USABLE_SHARE) : null;
    const requestedBytes = Number.isFinite(utilization) && torchTotal ? Math.round(utilization * torchTotal) : null;
    const checks = {
        cudaAvailable: torchShare.value?.cudaAvailable === true,
        archSupported,
        torchViewsAgree: torchShare.value?.memGetInfo?.total === torchShare.value?.totalMemory && torchTight.value?.memGetInfo?.total === torchTight.value?.totalMemory,
        denominatorIsPhysical: classification.denominator === 'physical-device',
        matchesIntended: Boolean(torchTotal) && Math.abs(torchTotal - raw.totalBytes * VLLM_USABLE_SHARE) <= USABLE_TOLERANCE * raw.totalBytes,
        fitsShare: requestedBytes !== null && requestedBytes + CONTEXT_MARGIN_BYTES <= budget.gpuShare.vramBytes,
        admissionFits: admission.status === 'ok',
        argvBuilt: launch !== null,
        hostAgrees: hostNvmlBytes === null || hostNvmlBytes === raw.totalBytes,
    };
    const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => name);
    const document = {
        schema: CALIBRATION_SCHEMA, createdAt: now().toISOString(), model: MODEL_ID,
        tuple,
        install: { version: entry.version, runnerDir, packageVersion: installed.version, dist: installed.dist, ready },
        lock: { entryDigest: entry.digest, runnerLockDigest: tuple.runnerLockDigest, downloadBytes: entry.totalBytes, files: entry.files.length, distributions: entry.check.distributions },
        share: { smPercent: budget.gpuShare.smPercent, vramBytes: budget.gpuShare.vramBytes, tightBytes: tightMiB * MIB },
        measurements: {
            hostNvmlTotalBytes: hostNvmlBytes, containerNvmlTotalBytes: raw.totalBytes, nvmlName: raw.name,
            torchShare: torchShare.value, torchTight: torchTight.value, ctypesShare: ctypesShare.value, ctypesTight: ctypesTight.value,
            utilization, intendedBytes, requestedBytes, usableShare: VLLM_USABLE_SHARE,
        },
        denominator: classification,
        sizing,
        argv: launch ? { command: launch.command, args: launch.args, envNames: Object.keys(launch.env).sort(), mpsEnvironment: budget.environment } : { error: launchError, mpsEnvironment: budget.environment },
        admission: { status: admission.status, reason: admission.reason, reasonCode: admission.reasonCode ?? null, estimate: admission.estimate },
        verdict: { qualifiable: failed.length === 0 && blockers.length === 0, denominator: classification.denominator, checks, failed },
    };
    document.evidenceDigest = evidenceDigest(document);
    return result(document);
}

// ---------------------------------------------------------------------------
// The reviewed qualification entry, rendered from stage-1 evidence. This is the
// data a later, separate writer step adds to REVIEWED_QUALIFICATIONS.

export function renderQualificationEntry(evidence) {
    const fail = (message) => { throw new Error(`Cannot render a qualification entry: ${message}`); };
    if (!evidence || evidence.schema !== CALIBRATION_SCHEMA) fail('the document is not a stage-1 calibration');
    if (evidence.verdict?.qualifiable !== true || evidence.verdict?.denominator !== 'physical-device') fail('the calibration did not establish the physical-device denominator');
    const digest = evidenceDigest(evidence);
    if (evidence.evidenceDigest !== undefined && evidence.evidenceDigest !== digest) fail('the document does not match its own digest');
    const tuple = Object.fromEntries(TUPLE_FIELDS.map((field) => [field, evidence.tuple?.[field]]));
    const entry = { ...tuple, denominator: 'physical-device', evidenceDigest: digest };
    // Production itself must accept the entry for this tuple, and refuse it for any changed field.
    const resolve = createVllmMpsQualificationResolver(() => [entry]);
    const accepted = resolve(tuple);
    if (!accepted.qualified || accepted.evidenceDigest !== digest) fail('the production resolver does not accept the rendered entry for its own tuple');
    const changed = { runnerLockDigest: '0'.repeat(64), driverVersion: '0.0', gpuPciDeviceId: '0x00000000', computeCapability: '0.0', deviceTotalBytes: tuple.deviceTotalBytes + 1 };
    for (const field of TUPLE_FIELDS) {
        if (resolve({ ...tuple, [field]: changed[field] }).qualified) fail(`the production resolver accepts a changed ${field}`);
    }
    const body = TUPLE_FIELDS.map((field) => `        ${field}: ${JSON.stringify(entry[field])},`).concat([`        denominator: 'physical-device',`, `        evidenceDigest: '${digest}',`]).join('\n');
    return { entry, digest, source: `    Object.freeze({\n${body}\n    }),` };
}

/**
 * The reviewed data entry itself: returns the module source with `entry` added
 * to REVIEWED_QUALIFICATIONS. Only that list changes; anything else in the
 * module, or a list that already holds the tuple, is refused.
 */
export function applyQualificationEntry(moduleSource, rendered) {
    const empty = /(const REVIEWED_QUALIFICATIONS = Object\.freeze\(\[)(\]\);)/;
    const filled = /(const REVIEWED_QUALIFICATIONS = Object\.freeze\(\[\n)([\s\S]*?)(\]\);)/;
    if (!rendered?.source || !rendered.entry) throw new Error('A rendered entry is required');
    const current = /const REVIEWED_QUALIFICATIONS = Object\.freeze\(\[([\s\S]*?)\]\);/.exec(moduleSource);
    if (!current) throw new Error('REVIEWED_QUALIFICATIONS was not found in the module');
    if (current[1].includes(rendered.entry.evidenceDigest)) throw new Error('This evidence is already in REVIEWED_QUALIFICATIONS');
    const sameTuple = current[1].split(/Object\.freeze\(\{/).slice(1).some((block) => TUPLE_FIELDS.every((field) => block.includes(`${field}: ${JSON.stringify(rendered.entry[field])},`)));
    if (sameTuple) throw new Error('REVIEWED_QUALIFICATIONS already holds an entry for this tuple');
    if (empty.test(moduleSource)) return moduleSource.replace(empty, `$1\n${rendered.source}\n$2`);
    if (filled.test(moduleSource)) return moduleSource.replace(filled, (_all, head, body, tail) => `${head}${body}${rendered.source}\n${tail}`);
    throw new Error('REVIEWED_QUALIFICATIONS has an unsupported layout');
}

// ---------------------------------------------------------------------------
// Command line.

function parseArguments(argv) {
    const [command, ...rest] = argv;
    const options = {};
    for (let index = 0; index < rest.length; index += 2) {
        const key = rest[index];
        if (!/^--[a-z-]+$/.test(key || '') || rest[index + 1] === undefined) throw new Error(`Unexpected argument ${key}`);
        options[key.slice(2)] = rest[index + 1];
    }
    return { command, options };
}

export async function main(argv = process.argv.slice(2), { out = (text) => process.stdout.write(text), ...deps } = {}) {
    let document;
    try {
        const { command, options } = parseArguments(argv);
        if (command === 'prerequisites') {
            let pins = null;
            if (options.pins !== undefined) pins = JSON.parse(options.pins);
            document = await prerequisiteReport({ pins, ...deps });
        } else if (command === 'calibrate') {
            const host = options['host-nvml-bytes'] === undefined ? null : Number(options['host-nvml-bytes']);
            if (host !== null && !(Number.isSafeInteger(host) && host > 0)) throw new Error('--host-nvml-bytes must be a positive integer');
            document = await calibrationReport({ hostNvmlBytes: host, ...deps });
        } else if (command === 'render') {
            const evidence = JSON.parse(fs.readFileSync(options.evidence, 'utf8'));
            const rendered = renderQualificationEntry(evidence.evidence ?? evidence);
            document = { ok: true, entry: rendered.entry, digest: rendered.digest, source: rendered.source };
        } else throw new Error(`Unknown command ${command}`);
    } catch (error) {
        document = { ok: false, blockers: [{ code: 'tool_error', message: String(error?.message || error).slice(0, 300), evidence: {} }] };
    }
    let text = JSON.stringify(document);
    if (Buffer.byteLength(text) > OUTPUT_LIMIT) {
        text = JSON.stringify({ ...document, evidence: undefined, truncated: true, blockers: [...(document.blockers || []), { code: 'output_too_large', message: `The report exceeds ${OUTPUT_LIMIT} bytes.`, evidence: {} }], ok: false });
    }
    out(`${text}\n`);
    return document.ok ? 0 : 3;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    process.exitCode = await main();
}

export { GIB };
