// vLLM (runners plan §5.5, Phase R5). Installed on demand from the image's
// runner lock (DS004); it runs from its runnable copy, reads a Hugging Face
// snapshot (weight format hf), listens on loopback with a per-start key, and
// is admitted only when its weights and KV cache fit the GPU unless the admin
// explicitly offloads weights to RAM (admitVllm).

import fs from 'node:fs';
import path from 'node:path';

import { admitVllm } from '../controller/admission.mjs';
import {
    assertAbsolutePath,
    assertApiKey,
    assertPort,
    codedError,
    deepFreeze,
    recommendedFor,
    validateParams
} from './params.mjs';

const ID = 'vllm';
const PORT = 18082;
const PINNED_VERSION = '0.30.0';
const NVIDIA_LIB_DIR = '/usr/local/nvidia/lib64';
// vLLM's front end and engine talk over ZMQ ipc:// sockets and set up
// torch.distributed through a file: in nested rootless Podman the container's
// /tmp (fuse-overlayfs) makes new socket nodes unusable, while the agent's own
// /dev/shm is a private tmpfs (Ploinky gives every agent its own IPC namespace).
const RPC_DIR_NAME = 'local-llm-vllm';
const RPC_DIR = `/dev/shm/${RPC_DIR_NAME}`;

const paramSchema = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        maxModelLen: {
            type: 'integer', minimum: 512, maximum: 131072, default: 4096,
            title: 'Max model length',
            description: 'Maximum context length per request, in tokens; the KV cache is sized for it.'
        },
        gpuMemoryUtilization: {
            type: ['number', 'null'], minimum: 0.1, maximum: 0.95, default: null,
            title: 'GPU memory utilization',
            description: 'Share of the GPU vLLM may use for weights and KV cache; empty takes what is free now (at most 0.9).'
        },
        cpuOffloadGb: {
            type: 'number', minimum: 0, maximum: 512, default: 0,
            title: 'CPU offload (GiB)',
            description: 'GiB of model weights kept in system RAM; lets a larger model run, much more slowly.'
        },
        dtype: {
            type: 'string', enum: ['auto', 'float16', 'bfloat16'], default: 'auto',
            title: 'Data type',
            description: 'Data type for model weights and activations.'
        },
        quantization: {
            type: ['string', 'null'], pattern: '^[a-z0-9_]{1,32}$', maxLength: 32, default: null,
            title: 'Quantization',
            description: 'Quantization method name (e.g. awq_marlin, gptq); empty lets vLLM detect it.'
        },
        kvCacheDtype: {
            type: 'string', enum: ['auto', 'fp8'], default: 'auto',
            title: 'KV cache type',
            description: 'Data type of the KV cache.'
        },
        enforceEager: {
            type: 'boolean', default: true,
            title: 'Enforce eager mode',
            description: 'Skip CUDA graph capture and compilation: a faster start, slightly slower generation.'
        },
        maxNumSeqs: {
            type: 'integer', minimum: 1, maximum: 256, default: 1,
            title: 'Max sequences',
            description: 'Maximum number of requests processed concurrently.'
        }
    }
});

function normalizeParams(params = {}, { model } = {}) {
    return validateParams(paramSchema, params, { defaults: recommendedFor(model, ID) });
}

function describeContext(params = {}, { model } = {}) {
    const { maxModelLen, maxNumSeqs } = normalizeParams(params, { model });
    // vLLM's paged KV pool is shared by all sequences; each may grow up to maxModelLen.
    return { totalContext: maxModelLen, perRequestContext: maxModelLen, parallel: maxNumSeqs, kvUnified: true };
}

// Installed on demand (DS004): detection asks the installer, which reads the
// cache and its install record; it never imports Python packages.
async function detect({ installer } = {}) {
    if (!installer?.installable(ID)) {
        return { installed: false, version: null, reason: 'This image\'s runner lock has no vLLM entry.' };
    }
    const info = await installer.describe(ID);
    if (!info.installed) return { installed: false, version: null, reason: 'Not installed. An admin can Install it under Runners.' };
    return { installed: true, version: info.version, reason: null };
}

function assertModelId(model) {
    if (typeof model?.id !== 'string' || model.id.length === 0) {
        throw codedError('invalid_launch', 'model.id must be a non-empty string', { field: 'model.id' });
    }
    return model.id;
}

function buildLaunch({ runnerDir, artifactPath, params, port, apiKey, model, gpuMemoryUtilization, cacheDir, rpcDir = RPC_DIR } = {}) {
    const values = normalizeParams(params, { model });
    const utilization = values.gpuMemoryUtilization ?? gpuMemoryUtilization;
    if (typeof utilization !== 'number' || !(utilization >= 0.1 && utilization <= 0.95)) {
        throw codedError('invalid_launch', 'gpuMemoryUtilization must come from the parameters or from admission', { field: 'gpuMemoryUtilization' });
    }
    const runRoot = assertAbsolutePath(runnerDir, 'runnerDir');
    const cache = assertAbsolutePath(cacheDir, 'cacheDir');
    const args = [
        '-m', 'vllm.entrypoints.openai.api_server',
        '--model', assertAbsolutePath(artifactPath, 'artifactPath'),
        '--served-model-name', assertModelId(model),
        '--host', '127.0.0.1',
        '--port', assertPort(port),
        '--max-model-len', values.maxModelLen,
        '--gpu-memory-utilization', utilization,
        '--max-num-seqs', values.maxNumSeqs,
        '--dtype', values.dtype,
    ];
    if (values.enforceEager) args.push('--enforce-eager');
    if (values.quantization) args.push('--quantization', values.quantization);
    if (values.kvCacheDtype !== 'auto') args.push('--kv-cache-dtype', values.kvCacheDtype);
    if (values.cpuOffloadGb > 0) args.push('--cpu-offload-gb', values.cpuOffloadGb);
    return {
        command: path.join(runRoot, 'venv', 'bin', 'python'),
        args: args.map(String),
        env: {
            // vLLM reads its key from the environment, so it is neither in the
            // process arguments nor in the "non-default args" line vLLM logs.
            VLLM_API_KEY: assertApiKey(apiKey),
            LD_LIBRARY_PATH: NVIDIA_LIB_DIR,
            // Triton compiles a small C launcher and links the granted driver
            // library by its versioned name, libcuda.so.1.
            TRITON_LIBCUDA_PATH: NVIDIA_LIB_DIR,
            TRITON_CACHE_DIR: path.join(cache, 'triton'),
            XDG_CACHE_HOME: path.join(cache, 'xdg'),
            VLLM_CACHE_ROOT: path.join(cache, 'vllm'),
            VLLM_RPC_BASE_PATH: assertAbsolutePath(rpcDir, 'rpcDir'),
            TMPDIR: rpcDir,
            // torch.distributed's Gloo and NCCL groups (created even for one
            // GPU) otherwise listen on the container's Box-network address,
            // where other agents could reach them; keep them on loopback.
            GLOO_SOCKET_IFNAME: 'lo',
            NCCL_SOCKET_IFNAME: 'lo',
            VLLM_HOST_IP: '127.0.0.1',
            // The FlashInfer sampler compiles CUDA code at run time; the image has no CUDA toolkit.
            VLLM_USE_FLASHINFER_SAMPLER: '0',
            VLLM_NO_USAGE_STATS: '1',
            DO_NOT_TRACK: '1',
            HF_HUB_OFFLINE: '1',
            TRANSFORMERS_OFFLINE: '1',
        },
    };
}

// The runnable copy is rebuilt per container; the Triton and vLLM caches sit
// beside it in the container's own filesystem, so compiled kernels never
// outlive the container either.
async function start(ctx) {
    const cacheDir = path.join(path.dirname(path.dirname(ctx.runnerDir)), '.cache', ID);
    const rpcDir = path.join(ctx.shmDir || '/dev/shm', RPC_DIR_NAME);
    for (const dir of [cacheDir, rpcDir]) fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
    const process = ctx.launch(ctx.runner.buildLaunch({
        runnerDir: ctx.runnerDir,
        artifactPath: ctx.weights.path,
        params: ctx.params,
        port: ctx.port,
        apiKey: ctx.apiKey,
        model: ctx.model,
        gpuMemoryUtilization: ctx.admission?.estimate?.gpuMemoryUtilization,
        cacheDir,
        rpcDir,
    }));
    const base = `http://127.0.0.1:${ctx.port}`;
    await ctx.waitForHttp(`${base}/health`, { process });
    await ctx.waitForHttp(`${base}/v1/models`, { headers: { authorization: `Bearer ${ctx.apiKey}` }, process });
    return {};
}

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

// vLLM logs the memory its weights took and the KV cache it allocated.
function parseReport(lines) {
    const report = { modelMiB: null, kvMiB: null, kvTokens: null, computeMiB: null, offloaded: null, device: null };
    for (const { line } of lines) {
        const loaded = /Model loading took ([\d.]+) GiB/.exec(line);
        if (loaded) report.modelMiB = Math.round(Number(loaded[1]) * GIB / MIB);
        const kv = /Available KV cache memory: ([\d.]+) GiB/.exec(line);
        if (kv) report.kvMiB = Math.round(Number(kv[1]) * GIB / MIB);
        const tokens = /GPU KV cache size: ([\d,]+) tokens/.exec(line);
        if (tokens) report.kvTokens = Number(tokens[1].replaceAll(',', ''));
    }
    const known = [report.modelMiB, report.kvMiB].filter((value) => value !== null);
    report.totalMiB = known.length ? known.reduce((total, value) => total + value, 0) : null;
    return report;
}

const vllmRuntime = Object.freeze({
    id: ID,
    displayName: 'vLLM',
    weightFormat: 'hf',
    pinnedVersion: PINNED_VERSION,
    supported: true,
    executable: null,
    port: PORT,
    apiKey: true,
    paramSchema,
    basicParams: Object.freeze(['maxModelLen', 'cpuOffloadGb']),
    moeParams: Object.freeze([]),
    normalizeParams,
    describeContext,
    detect,
    buildLaunch,
    start,
    chatModel: (deployment) => deployment.modelId,
    admit: admitVllm,
    parseReport,
});

export { vllmRuntime, vllmRuntime as vllmRunner };
