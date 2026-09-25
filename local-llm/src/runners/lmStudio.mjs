// LM Studio (runners plan §5.7, Phase R7; LOCAL_LLM_LMSTUDIO_HANDOFF.md).
// Its headless daemon, llmster, is proprietary: it is installed on demand from
// the image's runner lock after an admin accepts LM Studio's Terms (DS004),
// for internal use only, and only where the deployment's operator turned the
// switch on (LOCAL_LLM_LMSTUDIO=internal-use). It never ships in the image.
//
// It runs straight from the verified runnable copy, with its home inside that
// copy: on its first start llmster moves its engines out of the copy into the
// home, so the two must live and die together (in the container's own
// filesystem, never in /data). It reads the agent's own GGUF files, imported
// by symbolic link. It is controlled only through LM Studio's published
// interfaces: the `lms` CLI (import, server, the loaded list and unloads) and
// the MIT @lmstudio/sdk (load with every memory flag set; an SDK client may
// load but not start the server). The exact MoE layer count
// and threads go through the SDK's argument override, and the engine's command
// line is checked afterwards, so admission sizes what really runs.

import fs from 'node:fs';
import path from 'node:path';

import { admitLmStudio } from '../controller/admission.mjs';
import { defaultThreads, physicalCoreCount } from '../controller/hardware.mjs';
import { LLAMA_SERVER_PARAM_SCHEMA } from './llamaServer.mjs';
import { ParamError, assertAbsolutePath, assertPort, codedError, deepFreeze, recommendedFor, validateParams } from './params.mjs';

const ID = 'lmstudio';
const PORT = 18084;
const PINNED_VERSION = '0.0.25-1';
export const LMSTUDIO_SWITCH = 'LOCAL_LLM_LMSTUDIO';
const SWITCH_VALUE = 'internal-use';
const DAEMON_URL = 'ws://127.0.0.1:41343';
const NVIDIA_LIB_DIR = '/usr/local/nvidia/lib64';
// LM Studio's MIT SDK, pinned in the image (container-image-builds sources.lock.json).
export const SDK_DIR = '/opt/local-llm/lmstudio-sdk';
// The bundled engine only (handoff decision L7): LM Studio's CUDA 12 engine
// 2.41.0, llama.cpp b11026. An engine from any other directory is refused.
export const ENGINE_DIR = 'llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.41.0';
// The LM Studio "user/repo" our imports go under.
const USER_REPO = 'local-llm';
const HELPER = path.join(import.meta.dirname, 'lmStudioLoad.mjs');

const pick = (keys) => Object.fromEntries(keys.map((key) => [key, LLAMA_SERVER_PARAM_SCHEMA.properties[key]]));

// The llama-server parameters LM Studio can be given exactly. Flash attention
// is on or off (LM Studio has no auto), and mlock is not offered: it has no
// effect in a rootless container (VmLck stays 0).
const paramSchema = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        ...pick(['ctxSize', 'nGpuLayers', 'nCpuMoe']),
        flashAttn: {
            type: 'string', enum: ['on', 'off'], default: 'on',
            title: 'Flash attention',
            description: 'Flash attention on or off.'
        },
        ...pick(['cacheTypeK', 'cacheTypeV', 'threads', 'parallel', 'batchSize', 'ubatchSize', 'noMmap']),
    }
});

function normalizeParams(params = {}, { model } = {}) {
    const values = validateParams(paramSchema, params, { defaults: recommendedFor(model, ID) });
    if (values.ubatchSize > values.batchSize) {
        throw new ParamError('ubatchSize', `must be <= batchSize (${values.batchSize})`);
    }
    return values;
}

function describeContext(params = {}, { model } = {}) {
    const { ctxSize, parallel } = normalizeParams(params, { model });
    // Several parallel predictions share one unified KV cache, as with llama.cpp.
    return { totalContext: ctxSize, perRequestContext: ctxSize, parallel, kvUnified: parallel > 1 };
}

let autoThreads = null;
function threadsFor(values) {
    return values.threads ?? (autoThreads ??= defaultThreads(physicalCoreCount()));
}

/** The SDK load config: every flag that affects memory, set explicitly. */
export function loadConfig(values, { threads }) {
    const cacheType = (type) => (type === 'f16' ? false : type);
    return {
        contextLength: values.ctxSize,
        gpu: { ratio: 1, numCpuExpertLayersRatio: 0 },
        maxParallelPredictions: values.parallel,
        useUnifiedKvCache: values.parallel > 1,
        evalBatchSize: values.batchSize,
        physicalBatchSize: values.ubatchSize,
        flashAttention: values.flashAttn === 'on',
        llamaKCacheQuantizationType: cacheType(values.cacheTypeK),
        llamaVCacheQuantizationType: cacheType(values.cacheTypeV),
        keepModelInMemory: false,
        tryMmap: !values.noMmap,
        contextCheckpoints: 0,
        // The SDK's MoE setting is a ratio that LM Studio maps to a layer count
        // non-linearly (0.708 of gpt-oss's 24 layers gave 18, 1 gave 35), so
        // the exact counts are appended; llama-server uses the last value.
        llamaCppArgumentsOverride: {
            enabled: true,
            disabledParameters: [],
            overrideParameters: [
                { key: '--n-gpu-layers', value: String(values.nGpuLayers) },
                { key: '--n-cpu-moe', value: String(values.nCpuMoe) },
                { key: '--threads', value: String(threads) },
            ],
            excludeAllConfig: false,
        },
    };
}

/** What the engine's command line must say for admission's estimate to hold. */
export function expectedEngineFlags(values, { threads }) {
    return {
        '--ctx-size': String(values.ctxSize),
        '--n-gpu-layers': String(values.nGpuLayers),
        '--n-cpu-moe': String(values.nCpuMoe),
        '--batch-size': String(values.batchSize),
        '--ubatch-size': String(values.ubatchSize),
        '--parallel': String(values.parallel),
        '--threads': String(threads),
        '--cache-type-k': values.cacheTypeK,
        '--cache-type-v': values.cacheTypeV,
        '--flash-attn': values.flashAttn,
        '--load-mode': values.noMmap ? 'none' : 'mmap',
        '--ctx-checkpoints': '0',
        unifiedKv: values.parallel > 1,
    };
}

/** llama-server's view of its arguments: for a repeated flag the last value counts. */
export function effectiveFlags(argv) {
    const flags = {};
    for (let index = 1; index < argv.length; index += 1) {
        const arg = argv[index];
        if (!arg.startsWith('--')) continue;
        const next = argv[index + 1];
        if (next !== undefined && !next.startsWith('--')) {
            flags[arg] = next;
            index += 1;
        } else {
            flags[arg] = true;
        }
    }
    return flags;
}

export function engineFlagProblems(argv, expected) {
    const problems = [];
    if (!argv[0]?.endsWith(`/extensions/backends/${ENGINE_DIR}/llama-server`)) {
        problems.push(`engine ${argv[0]} is not the pinned engine (${ENGINE_DIR})`);
    }
    const flags = effectiveFlags(argv);
    for (const [flag, value] of Object.entries(expected)) {
        if (flag === 'unifiedKv') continue;
        if (flags[flag] !== value) problems.push(`${flag} is ${flags[flag] ?? 'missing'}, expected ${value}`);
    }
    const unified = argv.lastIndexOf('--kv-unified') > argv.lastIndexOf('--no-kv-unified');
    if (unified !== expected.unifiedKv) problems.push(`the KV cache is ${unified ? '' : 'not '}unified (--kv-unified), expected ${expected.unifiedKv ? '' : 'not '}unified`);
    return problems;
}

/** Processes left from an LM Studio start: any whose executable lies under its runnable copy. */
export function strayProcesses(procs, runDir, { keep = [] } = {}) {
    const root = `${runDir}/`;
    return procs.filter((proc) => typeof proc.exe === 'string' && proc.exe.startsWith(root) && !keep.includes(proc.pid))
        .map((proc) => proc.pid);
}

function listProcesses(procDir = '/proc') {
    const procs = [];
    for (const name of fs.readdirSync(procDir)) {
        if (!/^\d+$/.test(name)) continue;
        let exe = null;
        try { exe = fs.readlinkSync(path.join(procDir, name, 'exe')); } catch {}
        procs.push({ pid: Number(name), exe });
    }
    return procs;
}

function readArgv(pid, procDir = '/proc') {
    try {
        return fs.readFileSync(path.join(procDir, String(pid), 'cmdline'), 'utf8').split('\0').filter(Boolean);
    } catch {
        return null;
    }
}

/**
 * llmster's stdout echoes every API request and response as multi-line JSON.
 * The controller keeps the runner's output in /data/logs/runner.log, which
 * persists and which admins read, so the filter is an allowlist: only log
 * headers ("[Provider] …", "I [LMSExternal] …", the engine's "0.00.000.000 I …")
 * and llmster's own lifecycle messages pass, each cut at its first brace so no
 * JSON body follows it; a line that opens a JSON block starts a skip that ends
 * at the block's closing line; everything else, indented or bare text, is
 * dropped. The runner process never passes on the rest of an over-long line.
 */
const LOGGED_LINES = Object.freeze([
    /^\[/,
    /^[DIWE] \[LMS/,
    /^\d+\.\d+\.\d+\.\d+ [DIWE] /,
    /^(llmster started successfully|App is quitting|Received SIGTERM|Daemon (starting|changed)|WARNING - |Hardware survey |Forking |Unloading model: |Failed to |Try increasing RLIMIT_MEMLOCK|\(node:\d+\) )/,
]);

// LM Studio's server log lines ("[2026-09-25 12:30:36][INFO] …") also carry
// request text on a single line ("Received request to embed: <input>"), so of
// those only the ones naming a client or an endpoint, and fixed server
// messages, pass; an error or a warning keeps what precedes its first colon.
const SERVER_LOG_LINE = /^\[\d{4}-\d\d-\d\d [\d:]+\]\[([A-Z]+)\]/;
const SERVER_LOG_KEPT = /^(\[LMSAuthenticator\]|\[LM STUDIO SERVER\]|\s*(Server started\.|Server stopped\.|Just-in-time model loading active\.)$)/;

function serverLogLine(line, match) {
    const rest = line.slice(match[0].length);
    if (SERVER_LOG_KEPT.test(rest)) return line;
    if (!['ERROR', 'WARN'].includes(match[1])) return null;
    const colon = rest.indexOf(':');
    return colon >= 0 ? `${match[0]}${rest.slice(0, colon + 1)} …` : line;
}

export function createOutputFilter({ maxLength = 400 } = {}) {
    let inBody = false;
    return (line) => {
        if (inBody) {
            if (/^[}\]]/.test(line)) inBody = false;
            return null;
        }
        let trimmed = line.trimEnd();
        if (!LOGGED_LINES.some((pattern) => pattern.test(trimmed))) return null;
        const server = SERVER_LOG_LINE.exec(trimmed);
        if (server) {
            // A request body that opens here still has to be skipped.
            if (/[{[]$/.test(trimmed)) inBody = true;
            trimmed = serverLogLine(trimmed, server);
            if (trimmed === null) return null;
        }
        const brace = trimmed.indexOf('{');
        if (/[{[]$/.test(trimmed)) {
            inBody = true;
            return `${trimmed.slice(0, brace >= 0 ? brace : trimmed.length - 1)}{…}`.slice(0, maxLength);
        }
        return (brace >= 0 ? `${trimmed.slice(0, brace)}{…}` : trimmed).slice(0, maxLength);
    };
}

function enabled(env = {}) {
    if (env[LMSTUDIO_SWITCH] === SWITCH_VALUE) return { enabled: true, reason: null };
    return {
        enabled: false,
        reason: 'LM Studio is not enabled on this deployment (internal use only). Its Terms allow only personal and internal '
            + 'business use and forbid offering it as a service, so it stays off on deployments offered to other people. '
            + `This deployment's operator enables it with "ploinky var ${LMSTUDIO_SWITCH} ${SWITCH_VALUE}" and a restart of local-llm.`,
    };
}

// Installed on demand (DS004): detection reads the installer's record only.
async function detect({ installer } = {}) {
    if (!installer?.installable(ID)) {
        return { installed: false, version: null, reason: 'This image\'s runner lock has no LM Studio entry.' };
    }
    const info = await installer.describe(ID);
    if (!info.installed) return { installed: false, version: null, reason: 'Not installed. An admin can Install it under Runners.' };
    return { installed: true, version: info.version, reason: null };
}

function homeOf(runnerDir) {
    return path.join(assertAbsolutePath(runnerDir, 'runnerDir'), 'home');
}

function buildLaunch({ runnerDir } = {}) {
    const dir = assertAbsolutePath(runnerDir, 'runnerDir');
    return {
        command: path.join(dir, 'llmster'),
        args: [],
        cwd: dir,
        // llmster keeps its home in $HOME/.lmstudio and writes its own
        // ~/.lmstudio-home-pointer there.
        env: { HOME: homeOf(dir), LD_LIBRARY_PATH: NVIDIA_LIB_DIR },
        // A factory: the runner process makes one filter per output stream.
        outputFilter: createOutputFilter,
    };
}

// LM Studio's server log keeps every request and response, and no published
// setting turns it off; it lives in the container-local home and is deleted
// before each start and after each exit (the user's decision, runners plan §11).
export function serverLogsDir(runnerDir) {
    return path.join(homeOf(runnerDir), '.lmstudio', 'server-logs');
}

function toolEnv(runnerDir) {
    return { PATH: '/usr/local/bin:/usr/bin:/bin', HOME: homeOf(runnerDir), LANG: 'C.UTF-8' };
}

function lmsOf(runnerDir) {
    return path.join(runnerDir, '.bundle', 'lms');
}

/**
 * LM Studio loads any indexed model a request names, and JIT loading cannot
 * be turned off headlessly, so anything but our model is unloaded (`lms ps`,
 * `lms unload`). Returns the identifiers it unloaded; fails if ours is gone.
 */
async function unloadOthers({ exec, runnerDir, modelId, quiet = false }) {
    const env = toolEnv(runnerDir);
    const listed = await exec({ label: 'checking what LM Studio has loaded', command: lmsOf(runnerDir), args: ['ps', '--json'], env, json: true, quiet });
    const identifiers = (Array.isArray(listed) ? listed : []).map((entry) => entry?.identifier).filter((id) => typeof id === 'string');
    if (!identifiers.includes(modelId)) {
        throw codedError('runner_step_failed', `LM Studio reports ${identifiers.join(', ') || 'nothing'} loaded; ${modelId} is not loaded.`);
    }
    const others = identifiers.filter((id) => id !== modelId);
    for (const id of others) {
        await exec({ label: `unloading ${id} from LM Studio`, command: lmsOf(runnerDir), args: ['unload', id], env, quiet });
    }
    return others;
}

async function start(ctx, { procDir = '/proc', kill = (pid, signal) => globalThis.process.kill(pid, signal), node = globalThis.process.execPath } = {}) {
    const runDir = assertAbsolutePath(ctx.runnerDir, 'runnerDir');
    const home = homeOf(runDir);
    // A process left by an earlier start in this container (none is expected:
    // the engine stays in llmster's process group, which a stop kills).
    for (const pid of strayProcesses(listProcesses(procDir), runDir)) {
        try { kill(pid, 'SIGKILL'); } catch {}
    }
    fs.mkdirSync(home, { recursive: true, mode: 0o700 });
    fs.rmSync(serverLogsDir(runDir), { recursive: true, force: true });
    // No updates at run time (L7): llmster checks for updates on every start;
    // anything staged means the pinned bytes would no longer be what runs.
    const staged = path.join(home, '.lmstudio', '.internal', 'staged-updates-app');
    if (fs.existsSync(staged) && fs.readdirSync(staged).length > 0) {
        throw codedError('runner_changed', 'LM Studio has staged an update in its home; restart local-llm to rebuild LM Studio from its verified download.');
    }
    const values = normalizeParams(ctx.params, { model: ctx.model });
    const threads = threadsFor(values);
    const runner = ctx.launch(buildLaunch({ runnerDir: runDir }));
    // Our own imports only: the agent's GGUF file, linked, never moved or copied.
    const imports = path.join(home, '.lmstudio', 'models', USER_REPO);
    fs.rmSync(imports, { recursive: true, force: true });
    const weights = assertAbsolutePath(ctx.weights.path, 'weights');
    const lms = lmsOf(runDir);
    await ctx.exec({ label: 'waiting for LM Studio', command: node, args: [HELPER, 'wait', JSON.stringify({ daemon: DAEMON_URL, sdk: SDK_DIR })], env: toolEnv(runDir) });
    await ctx.exec({ label: 'importing the model into LM Studio', command: lms,
        args: ['import', weights, '--symbolic-link', '-y', '--user-repo', `${USER_REPO}/${ctx.model.id}`], env: toolEnv(runDir) });
    const loaded = await ctx.exec({ label: 'loading the model in LM Studio', command: node, json: true,
        args: [HELPER, 'load', JSON.stringify({
            daemon: DAEMON_URL, sdk: SDK_DIR, path: `${USER_REPO}/${ctx.model.id}/${path.basename(weights)}`,
            identifier: ctx.model.id, config: loadConfig(values, { threads }),
        })], env: toolEnv(runDir) });
    // On loopback only (its default, stated anyway).
    await ctx.exec({ label: 'starting LM Studio\'s server on 127.0.0.1', command: lms,
        args: ['server', 'start', '--port', String(assertPort(ctx.port)), '--bind', '127.0.0.1'], env: toolEnv(runDir) });
    await unloadOthers({ exec: ctx.exec, runnerDir: runDir, modelId: ctx.model.id });
    const base = `http://127.0.0.1:${ctx.port}`;
    await ctx.waitForHttp(`${base}/v1/models`, { process: runner });
    // Admission sized the flags we asked for: the engine must run exactly those.
    const engine = listProcesses(procDir).find((proc) => proc.exe?.startsWith(`${runDir}/`) && proc.exe.endsWith('/llama-server'));
    const argv = engine ? readArgv(engine.pid, procDir) : null;
    if (!argv) throw codedError('runner_flags', 'LM Studio loaded the model, but its engine process was not found.');
    const problems = engineFlagProblems(argv, expectedEngineFlags(values, { threads }));
    if (problems.length) {
        throw codedError('runner_flags', `LM Studio's engine does not run with the flags admission sized: ${problems.join('; ')}.`, { problems });
    }
    return { lmstudio: { modelKey: loaded?.modelKey ?? null, engine: { pid: engine.pid, dir: ENGINE_DIR } } };
}

/** After llmster's process group is gone: delete the server log, and kill anything left in its copy. */
function afterExit({ runnerDir, log = () => {} } = {}, { procDir = '/proc', kill = (pid, signal) => globalThis.process.kill(pid, signal) } = {}) {
    if (!runnerDir) return;
    for (const pid of strayProcesses(listProcesses(procDir), runnerDir)) {
        try {
            kill(pid, 'SIGKILL');
            log(`killed a leftover LM Studio process (${pid})`);
        } catch {}
    }
    fs.rmSync(serverLogsDir(runnerDir), { recursive: true, force: true });
}

// While the model is ready: LM Studio loads any indexed model a request names,
// so anything but ours is unloaded (the user's decision on L6).
const watchdog = Object.freeze({
    intervalMs: 30_000,
    async check({ exec, modelId, runnerDir }) {
        const unloaded = await unloadOthers({ exec, runnerDir, modelId, quiet: true });
        return unloaded.length ? `unloaded ${unloaded.join(', ')}, which a request had loaded` : null;
    },
});

export const lmStudioRunner = Object.freeze({
    id: ID,
    displayName: 'LM Studio (internal use only)',
    weightFormat: 'gguf',
    pinnedVersion: PINNED_VERSION,
    supported: true,
    executable: null,
    port: PORT,
    // LM Studio's API authentication can only be turned on in its desktop app.
    // Its server listens on loopback in the agent's own network namespace,
    // behind the closed agent-port relay (DS001).
    apiKey: false,
    paramSchema,
    basicParams: Object.freeze(['ctxSize', 'nCpuMoe']),
    moeParams: Object.freeze(['nCpuMoe']),
    enabled,
    normalizeParams,
    describeContext,
    detect,
    buildLaunch,
    start,
    watchdog,
    afterExit,
    // --identifier makes the model id the name the server answers to.
    chatModel: (deployment) => deployment.modelId,
    admit: admitLmStudio,
});
