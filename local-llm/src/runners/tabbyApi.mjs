// ExLlamaV3 + TabbyAPI (runners plan §5.6, Phase R6). Installed on demand
// after an admin accepts TabbyAPI's AGPL-3.0 notice (DS004); it runs from its
// source directory in the runnable copy, reads an EXL3 snapshot (weight
// format exl3), listens on loopback and requires its per-start key.

import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';

import { admitTabbyApi } from '../controller/admission.mjs';
import {
    assertAbsolutePath,
    assertApiKey,
    assertPort,
    codedError,
    deepFreeze,
    recommendedFor,
    validateParams,
    ParamError
} from './params.mjs';

const ID = 'tabbyapi';
const PORT = 18083;
const PINNED_VERSION = 'f07131c';
const NVIDIA_LIB_DIR = '/usr/local/nvidia/lib64';
// A private tmpfs (Ploinky gives every agent its own IPC namespace); the
// container's /tmp (fuse-overlayfs) cannot hold a usable socket.
const TMP_DIR_NAME = 'local-llm-tabbyapi';
const TMP_DIR = `/dev/shm/${TMP_DIR_NAME}`;

const paramSchema = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        maxSeqLen: {
            type: 'integer', minimum: 512, maximum: 131072, default: 4096,
            title: 'Max sequence length',
            description: 'Maximum context length per request, in tokens.'
        },
        cacheSize: {
            type: ['integer', 'null'], minimum: 512, maximum: 262144, default: null,
            title: 'Cache size',
            description: 'KV cache size in tokens, at least the max sequence length; empty uses the max sequence length.'
        },
        cacheMode: {
            type: 'string', enum: ['FP16', 'Q8', 'Q6', 'Q4'], default: 'FP16',
            title: 'Cache mode',
            description: 'KV cache precision; Q4 takes a quarter of FP16.'
        },
        chunkSize: {
            type: 'integer', minimum: 256, maximum: 8192, default: 2048,
            title: 'Chunk size',
            description: 'Prompt tokens processed per step; smaller needs less GPU memory.'
        }
    }
});

function normalizeParams(params = {}, { model } = {}) {
    const values = validateParams(paramSchema, params, { defaults: recommendedFor(model, ID) });
    if (values.cacheSize !== null && values.cacheSize < values.maxSeqLen) {
        throw new ParamError('cacheSize', `must be at least maxSeqLen (${values.maxSeqLen})`);
    }
    return values;
}

function describeContext(params = {}, { model } = {}) {
    const { maxSeqLen } = normalizeParams(params, { model });
    return { totalContext: maxSeqLen, perRequestContext: maxSeqLen, parallel: 1, kvUnified: true };
}

// Installed on demand (DS004): detection reads the installer's record only.
async function detect({ installer } = {}) {
    if (!installer?.installable(ID)) {
        return { installed: false, version: null, reason: 'This image\'s runner lock has no TabbyAPI entry.' };
    }
    const info = await installer.describe(ID);
    if (!info.installed) return { installed: false, version: null, reason: 'Not installed. An admin can Install it under Runners.' };
    return { installed: true, version: info.version.slice(0, 7), reason: null };
}

/** TabbyAPI's key file: the per-start key for both the user and the admin role. */
function authFile(apiKey) {
    const key = assertApiKey(apiKey);
    return `api_key: ${key}\nadmin_key: ${key}\n`;
}

function sourceDir(runnerDir) {
    return path.join(assertAbsolutePath(runnerDir, 'runnerDir'), 'tabbyAPI');
}

function buildLaunch({ runnerDir, artifactPath, params, port, apiKey, model, cacheDir, tmpDir = TMP_DIR } = {}) {
    const values = normalizeParams(params, { model });
    assertApiKey(apiKey);
    const snapshot = assertAbsolutePath(artifactPath, 'artifactPath');
    const cache = assertAbsolutePath(cacheDir, 'cacheDir');
    const cwd = sourceDir(runnerDir);
    const args = [
        path.join(cwd, 'main.py'),
        '--host', '127.0.0.1',
        '--port', assertPort(port),
        '--backend', 'exllamav3',
        '--model-dir', path.dirname(snapshot),
        '--model-name', path.basename(snapshot),
        '--max-seq-len', values.maxSeqLen,
        '--cache-size', values.cacheSize ?? values.maxSeqLen,
        '--cache-mode', values.cacheMode,
        '--chunk-size', values.chunkSize,
    ];
    return {
        command: path.join(path.dirname(cwd), 'venv', 'bin', 'python'),
        args: args.map(String),
        cwd,
        env: {
            LD_LIBRARY_PATH: NVIDIA_LIB_DIR,
            // TabbyAPI's Triton (3.5) links -lcuda, the unversioned name the
            // granted driver directory lacks; start() makes a directory of links.
            TRITON_LIBCUDA_PATH: path.join(cache, 'libcuda'),
            TRITON_CACHE_DIR: path.join(cache, 'triton'),
            XDG_CACHE_HOME: path.join(cache, 'xdg'),
            TMPDIR: assertAbsolutePath(tmpDir, 'tmpDir'),
            GLOO_SOCKET_IFNAME: 'lo',
            NCCL_SOCKET_IFNAME: 'lo',
            HF_HUB_OFFLINE: '1',
            TRANSFORMERS_OFFLINE: '1',
            DO_NOT_TRACK: '1',
        },
    };
}

function portIsFree(port) {
    return new Promise((resolve) => {
        const probe = net.createServer();
        probe.once('error', () => resolve(false));
        probe.listen(port, '127.0.0.1', () => probe.close(() => resolve(true)));
    });
}

async function start(ctx) {
    // TabbyAPI moves to port + 1 when its port is taken; then something else
    // would answer on the port the controller probes. Refuse instead.
    if (!(await portIsFree(ctx.port))) {
        throw codedError('port_busy', `Port ${ctx.port} is in use; TabbyAPI would move to another port.`, { port: ctx.port });
    }
    const cacheDir = path.join(path.dirname(path.dirname(ctx.runnerDir)), '.cache', ID);
    const tmpDir = path.join(ctx.shmDir || '/dev/shm', TMP_DIR_NAME);
    for (const dir of [cacheDir, tmpDir, path.join(cacheDir, 'libcuda')]) fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
    for (const name of ['libcuda.so', 'libcuda.so.1']) {
        const link = path.join(cacheDir, 'libcuda', name);
        fs.rmSync(link, { force: true });
        fs.symlinkSync(path.join(NVIDIA_LIB_DIR, 'libcuda.so.1'), link);
    }
    // TabbyAPI reads api_tokens.yml from its working directory, the source
    // directory in this container's runnable copy; a new key on every start.
    const tokens = path.join(sourceDir(ctx.runnerDir), 'api_tokens.yml');
    fs.rmSync(tokens, { force: true });
    fs.writeFileSync(tokens, authFile(ctx.apiKey), { mode: 0o600, flag: 'wx' });
    const process = ctx.launch(ctx.runner.buildLaunch({
        runnerDir: ctx.runnerDir, artifactPath: ctx.weights.path, params: ctx.params, port: ctx.port,
        apiKey: ctx.apiKey, model: ctx.model, cacheDir, tmpDir,
    }));
    const base = `http://127.0.0.1:${ctx.port}`;
    await ctx.waitForHttp(`${base}/health`, { process });
    // /v1/model answers only once a model is loaded.
    await ctx.waitForHttp(`${base}/v1/model`, { headers: { authorization: `Bearer ${ctx.apiKey}` }, process });
    return {};
}

// ExLlamaV3 does not log buffer sizes the way llama-server does.
function parseReport() {
    return { modelMiB: null, kvMiB: null, computeMiB: null, offloaded: null, device: null, totalMiB: null };
}

export const tabbyApiRunner = Object.freeze({
    id: ID,
    displayName: 'TabbyAPI (ExLlamaV3)',
    weightFormat: 'exl3',
    pinnedVersion: PINNED_VERSION,
    supported: true,
    executable: null,
    port: PORT,
    apiKey: true,
    paramSchema,
    basicParams: Object.freeze(['maxSeqLen', 'cacheMode']),
    moeParams: Object.freeze([]),
    normalizeParams,
    describeContext,
    detect,
    authFile,
    buildLaunch,
    start,
    chatModel: (deployment) => deployment.modelId,
    admit: admitTabbyApi,
    parseReport,
});
