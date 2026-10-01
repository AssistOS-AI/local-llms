import { describe, it, test } from 'node:test';
import assert from 'node:assert/strict';
import { admit } from '../src/controller/admission.mjs';
import { cpuFloorBytes, cpuHostReserveBytes } from '../src/controller/profiles.mjs';
import { ParamError, validateParams } from '../src/runners/params.mjs';
import { RUNNERS, getRunner, offeredIn, runnerSummaries, runnerSummary } from '../src/runners/index.mjs';
import { ikLlamaCppRunner } from '../src/runners/ikLlamaCpp.mjs';
import { llamaCppCpuRunner, llamaCppRunner } from '../src/runners/llamaCpp.mjs';
import { createLlamaServerRunner } from '../src/runners/llamaServer.mjs';
import { parseRunnerReport } from '../src/controller/runnerProcess.mjs';
import { defaultThreads, physicalCoreCount } from '../src/controller/hardware.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { ollamaRunner } from '../src/runners/ollama.mjs';
import { vllmRunner, vllmRuntime } from '../src/runners/vllm.mjs';

const API_KEY = 'k'.repeat(24) + '_-AZaz09k';
const MODEL = Object.freeze({ id: 'qwen3-8b-q4' });
const SHELL_META = /[;&|`$<>\n\\]/;

// Default --threads on this machine (runners plan, I2).
const AUTO_THREADS = String(defaultThreads(physicalCoreCount()));

function llamaLaunch(params, overrides = {}) {
    return llamaCppRunner.buildLaunch({
        artifactPath: '/data/models/gguf/model.gguf',
        params,
        port: 18080,
        apiKey: API_KEY,
        model: MODEL,
        ...overrides
    });
}

function hasPair(args, flag, value) {
    return args.some((arg, index) => arg === flag && args[index + 1] === value);
}

function assertParamError(fn, field) {
    assert.throws(fn, (error) => {
        assert.ok(error instanceof ParamError);
        assert.equal(error.code, 'invalid_params');
        if (field) {
            assert.equal(error.details.field, field);
        }
        return true;
    });
}

function assertCode(fn, code) {
    assert.throws(fn, (error) => error.code === code);
}

function fakeSpawn(result) {
    const calls = [];
    const spawnSync = (command, args, options) => {
        calls.push({ command, args, options });
        return result;
    };
    return { spawnSync, calls };
}

describe('validateParams', () => {
    const schema = {
        type: 'object',
        additionalProperties: false,
        properties: {
            n: { type: 'integer', minimum: 1, maximum: 10, default: 5 },
            maybe: { type: ['integer', 'null'], minimum: 1, maximum: 4, default: null },
            mixed: { type: ['string', 'number'], enum: ['off', 'max'], minimum: 0, maximum: 1 },
            nested: { type: 'object', additionalProperties: false, properties: { a: { type: 'boolean' } } }
        }
    };

    it('applies defaults, returns a frozen plain object', () => {
        const result = validateParams(schema, {}, { defaults: { n: 7 } });
        assert.deepEqual(result, { n: 7, maybe: null });
        assert.ok(Object.isFrozen(result));
        assert.equal(Object.getPrototypeOf(result), Object.prototype);
    });

    it('rejects non-plain inputs and prototype-named keys', () => {
        for (const input of [[], 'x', null, 5, new Map(), Object.create({ n: 1 })]) {
            assertParamError(() => validateParams(schema, input));
        }
        for (const key of ['constructor', 'toString', '__proto__', 'hasOwnProperty']) {
            assertParamError(() => validateParams(schema, JSON.parse(`{"${key}": 1}`)), key);
        }
        assert.deepEqual(validateParams(schema, Object.create(null)), { n: 5, maybe: null });
    });

    it('scopes enum to strings and range to numbers in unions', () => {
        assert.equal(validateParams(schema, { mixed: 0.5 }).mixed, 0.5);
        assert.equal(validateParams(schema, { mixed: 'max' }).mixed, 'max');
        assertParamError(() => validateParams(schema, { mixed: 'min' }), 'mixed');
        assertParamError(() => validateParams(schema, { mixed: 1.5 }), 'mixed');
        assertParamError(() => validateParams(schema, { mixed: true }), 'mixed');
    });

    it('validates nested objects and freezes them', () => {
        const result = validateParams(schema, { nested: { a: true } });
        assert.ok(Object.isFrozen(result.nested));
        assertParamError(() => validateParams(schema, { nested: { b: 1 } }), 'nested.b');
        assertParamError(() => validateParams(schema, { nested: [] }), 'nested');
    });

    it('treats null as unset only for nullable fields', () => {
        assert.equal(validateParams(schema, { maybe: null }).maybe, null);
        assertParamError(() => validateParams(schema, { n: null }), 'n');
        assertParamError(() => validateParams(schema, { n: Number.NaN }), 'n');
    });

    it('validates defaults like input', () => {
        assertParamError(() => validateParams(schema, {}, { defaults: { n: 99 } }), 'n');
    });
});

describe('llama.cpp runner', () => {
    it('builds argv pairs for context, MoE offload, slots, host and key', () => {
        const { command, args, env } = llamaLaunch({ ctxSize: 16384, nCpuMoe: 20, parallel: 1 });
        assert.equal(command, '/opt/llama.cpp/llama-server');
        assert.ok(hasPair(args, '--ctx-size', '16384'));
        assert.ok(hasPair(args, '--n-cpu-moe', '20'));
        assert.ok(hasPair(args, '-np', '1'));
        assert.ok(hasPair(args, '--host', '127.0.0.1'));
        assert.ok(hasPair(args, '--api-key', API_KEY));
        assert.ok(!args.includes('--kv-unified'));
        assert.deepEqual(env, { LD_LIBRARY_PATH: '/usr/local/nvidia/lib64' });
        assert.ok(args.every((arg) => typeof arg === 'string'));
    });

    it('emits the documented argument order with defaults', () => {
        const { args } = llamaLaunch({});
        assert.deepEqual(args, [
            '-m', '/data/models/gguf/model.gguf', '--host', '127.0.0.1', '--port', '18080',
            '--api-key', API_KEY, '--no-webui', '-lv', '4', '--alias', MODEL.id,
            '--ctx-size', '16384', '--n-gpu-layers', '99',
            '--flash-attn', 'auto', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
            '--threads', AUTO_THREADS, '-np', '1', '--batch-size', '2048', '--ubatch-size', '512'
        ]);
    });

    it('rejects invalid and unknown params', () => {
        assertParamError(() => llamaCppRunner.normalizeParams({ nCpuMoe: -1 }), 'nCpuMoe');
        assertParamError(() => llamaCppRunner.normalizeParams({ '--foo': 1 }), '--foo');
        assertParamError(() => llamaCppRunner.normalizeParams({ ctxSize: '16384; rm -rf /' }), 'ctxSize');
        assertParamError(() => llamaCppRunner.normalizeParams({ ctxSize: '16384' }), 'ctxSize');
        assertParamError(() => llamaCppRunner.normalizeParams({ ctxSize: 16384.5 }), 'ctxSize');
        assertParamError(() => llamaCppRunner.normalizeParams({ ctxSize: 256 }), 'ctxSize');
        assertParamError(() => llamaCppRunner.normalizeParams({ batchSize: 256, ubatchSize: 512 }), 'ubatchSize');
        assertParamError(() => llamaCppRunner.normalizeParams({ flashAttn: 'auto;id' }), 'flashAttn');
        assertParamError(() => llamaCppRunner.normalizeParams({ cacheTypeK: 'f16 && id' }), 'cacheTypeK');
        assertParamError(() => llamaCppRunner.buildLaunch({ params: { threads: 0 } }), 'threads');
    });

    it('unifies the KV cache when parallel > 1', () => {
        const { args } = llamaLaunch({ parallel: 4 });
        assert.ok(hasPair(args, '-np', '4'));
        assert.ok(args.includes('--kv-unified'));
        assert.deepEqual(llamaCppRunner.describeContext({ parallel: 4, ctxSize: 32768 }), {
            totalContext: 32768, perRequestContext: 32768, parallel: 4, kvUnified: true
        });
        assert.equal(llamaCppRunner.describeContext({}).kvUnified, false);
    });

    it('applies model recommended params over schema defaults', () => {
        const model = { id: 'gpt-oss-20b', recommended: { dedicated: { 'llama.cpp': { ctxSize: 65536, nCpuMoe: 12 } } } };
        const values = llamaCppRunner.normalizeParams({}, { model });
        assert.equal(values.ctxSize, 65536);
        assert.equal(values.nCpuMoe, 12);
        assert.equal(values.parallel, 1);
        assert.equal(llamaCppRunner.normalizeParams({ ctxSize: 8192 }, { model }).ctxSize, 8192);
        const { args } = llamaLaunch({}, { model });
        assert.ok(hasPair(args, '--ctx-size', '65536'));
        assert.ok(hasPair(args, '--alias', 'gpt-oss-20b'));
        const bad = { id: 'x', recommended: { dedicated: { 'llama.cpp': { ctxSize: 1 } } } };
        // Recommended values of another profile never apply.
        const unifiedOnly = { id: 'u', recommended: { unified: { 'llama.cpp': { ctxSize: 65536 } } } };
        assert.equal(llamaCppRunner.normalizeParams({}, { model: unifiedOnly }).ctxSize, 16384);
        assertParamError(() => llamaCppRunner.normalizeParams({}, { model: bad }), 'ctxSize');
    });

    it('caps a default context at the model\'s training context and refuses an explicit one above it', () => {
        const short = Object.freeze({ id: 'short', contextLength: 4096 });
        for (const runner of [llamaCppRunner, ikLlamaCppRunner]) {
            // The schema default (16384) and a catalog recommendation above the training context are capped, per slot.
            assert.equal(runner.normalizeParams({}, { model: short }).ctxSize, 4096);
            assert.equal(runner.normalizeParams({ parallel: 2 }, { model: short }).ctxSize, 8192);
            const recommended = { ...short, recommended: { dedicated: { [runner.id]: { ctxSize: 65536 } } } };
            assert.equal(runner.normalizeParams({}, { model: recommended }).ctxSize, 4096);
            // A default inside the training context, and an explicit value, are kept as they are.
            assert.equal(runner.normalizeParams({}, { model: { id: 'long', contextLength: 40960 } }).ctxSize, 16384);
            assert.equal(runner.normalizeParams({ ctxSize: 2048 }, { model: short }).ctxSize, 2048);
            assertParamError(() => runner.normalizeParams({ ctxSize: 8192 }, { model: short }), 'ctxSize');
            // The Run form sends the capped values back explicitly; they validate unchanged.
            const capped = runner.normalizeParams({}, { model: short });
            assert.deepEqual(runner.normalizeParams(capped, { model: short }), capped);
            assert.deepEqual(runner.describeContext({}, { model: short }).totalContext, 4096);
        }
        assert.ok(hasPair(llamaLaunch({}, { model: short }).args, '--ctx-size', '4096'));
        // The unified default (32768) is capped the same way.
        assert.equal(llamaCppRunner.normalizeParams({}, { model: short, profile: 'unified' }).ctxSize, 4096);
        assertParamError(() => llamaCppRunner.normalizeParams({ ctxSize: 8192 }, { model: short, profile: 'unified' }), 'ctxSize');
    });

    it('adds --jinja only when the model requires it', () => {
        assert.ok(llamaLaunch({}, { model: { id: 'm', requiresJinja: true } }).args.includes('--jinja'));
        assert.ok(!llamaLaunch({}).args.includes('--jinja'));
    });

    it('maps loadMode one to one to --load-mode and omits auto', () => {
        const modeOf = (params) => {
            const { args } = llamaLaunch(params);
            const index = args.indexOf('--load-mode');
            return index === -1 ? null : args[index + 1];
        };
        assert.equal(modeOf({}), null);
        assert.equal(modeOf({ loadMode: 'auto' }), null);
        for (const mode of ['none', 'mmap', 'mlock', 'mmap+mlock', 'dio']) assert.equal(modeOf({ loadMode: mode }), mode);
        const { args } = llamaLaunch({ loadMode: 'mlock' });
        assert.ok(!args.includes('--mlock') && !args.includes('--no-mmap'));
        assertParamError(() => llamaLaunch({ loadMode: 'direct' }), 'loadMode');
        // The flags b10875 removed are not parameters any more.
        assertParamError(() => llamaLaunch({ noMmap: true }), 'noMmap');
        assertParamError(() => llamaLaunch({ mlock: true }), 'mlock');
    });

    it('passes chatTemplateKwargs as one JSON argument', () => {
        const { args } = llamaLaunch({ chatTemplateKwargs: { reasoning_effort: 'high' } });
        assert.ok(hasPair(args, '--chat-template-kwargs', '{"reasoning_effort":"high"}'));
        assert.ok(!llamaLaunch({}).args.includes('--chat-template-kwargs'));
        assertParamError(() => llamaLaunch({ chatTemplateKwargs: { reasoning_effort: 'max' } }));
        assertParamError(() => llamaLaunch({ chatTemplateKwargs: { other: 1 } }), 'chatTemplateKwargs.other');
    });

    it('passes --threads: the admin-set value, else physical cores minus 2', () => {
        assert.ok(hasPair(llamaLaunch({ threads: 8 }).args, '--threads', '8'));
        assert.ok(hasPair(llamaLaunch({ threads: null }).args, '--threads', AUTO_THREADS));
    });

    it('rejects bad port, apiKey, artifactPath and model id', () => {
        for (const port of [80, 70000, '18080', 18080.5, undefined]) {
            assertCode(() => llamaLaunch({}, { port }), 'invalid_launch');
        }
        for (const apiKey of ['short', `${'a'.repeat(40)};id`, 'a'.repeat(129), undefined]) {
            assertCode(() => llamaLaunch({}, { apiKey }), 'invalid_launch');
        }
        assertCode(() => llamaLaunch({}, { artifactPath: 'relative.gguf' }), 'invalid_launch');
        assertCode(() => llamaLaunch({}, { model: {} }), 'invalid_launch');
    });

    it('parses the version from stderr and reports ENOENT', () => {
        const output = 'version: 0.5.0 (build 11159, commit 6b790a9c2)\nbuilt with GNU 13\n';
        const { spawnSync, calls } = fakeSpawn({ status: 0, stdout: '', stderr: output });
        assert.deepEqual(llamaCppRunner.detect({ spawnSync }), { installed: true, version: 'b11159', reason: null });
        assert.equal(calls[0].command, '/opt/llama.cpp/llama-server');
        assert.deepEqual(calls[0].args, ['--version']);
        assert.equal(calls[0].options.timeout, 10000);
        assert.equal(calls[0].options.shell, undefined);
        const enoent = Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' });
        const missing = llamaCppRunner.detect(fakeSpawn({ error: enoent }));
        assert.equal(missing.installed, false);
        assert.match(missing.reason, /not found/);
        const garbage = llamaCppRunner.detect(fakeSpawn({ status: 1, stdout: 'segfault', stderr: '' }));
        assert.equal(garbage.installed, false);
        assert.match(garbage.reason, /Unrecognized/);
        const other = llamaCppRunner.detect(fakeSpawn({ status: 0, stdout: 'version: 1 (build 11200, commit x)' }));
        assert.equal(other.version, 'b11200');
        assert.match(other.reason, /differs from pinned b11159/);
    });
});

describe('ik_llama.cpp runner', () => {
    const [GPT] = loadSeedCatalog();
    const PATH = '/data/models/gguf/ggml-org/gpt-oss-20b-GGUF/c/gpt-oss-20b-MXFP4.gguf';
    const ikLaunch = (params, model = GPT) => ikLlamaCppRunner.buildLaunch({ artifactPath: PATH, params, port: 18081, apiKey: API_KEY, model });

    it('reads the same GGUF as llama.cpp, on its own port, off PATH', () => {
        assert.equal(ikLlamaCppRunner.id, 'ik_llama.cpp');
        assert.equal(ikLlamaCppRunner.weightFormat, 'gguf');
        assert.equal(ikLlamaCppRunner.port, 18081);
        assert.equal(ikLlamaCppRunner.executable, '/opt/ik_llama.cpp/llama-server');
        assert.equal(ikLlamaCppRunner.pinnedVersion, '20f7a72');
        assert.deepEqual(ikLlamaCppRunner.basicParams, ['ctxSize', 'nCpuMoe']);
        // The same parameters as llama.cpp, except that ik has no direct I/O.
        const { loadMode: ikLoad, ...ikRest } = ikLlamaCppRunner.paramSchema.properties;
        const { loadMode: llamaLoad, ...llamaRest } = llamaCppRunner.paramSchema.properties;
        assert.deepEqual(ikRest, llamaRest);
        assert.deepEqual(ikLoad.enum, ['auto', 'none', 'mmap', 'mlock', 'mmap+mlock']);
        assert.deepEqual(llamaLoad.enum, ['auto', 'none', 'mmap', 'mlock', 'mmap+mlock', 'dio']);
        // ik has no unified-memory policy.
        assert.equal(ikLlamaCppRunner.paramSchemaFor('unified'), null);
        assert.equal(typeof ikLlamaCppRunner.admitUnified, 'undefined');
    });

    it('launches gpt-oss-20b with the flags ik understands, in the documented order', () => {
        const launch = ikLaunch({});
        assert.equal(launch.command, '/opt/ik_llama.cpp/llama-server');
        assert.deepEqual(launch.env, { LD_LIBRARY_PATH: '/usr/local/nvidia/lib64' });
        assert.deepEqual(launch.args, [
            '-m', PATH, '--host', '127.0.0.1', '--port', '18081', '--api-key', API_KEY,
            '--webui', 'none', '--alias', 'gpt-oss-20b', '--ctx-size', '16384', '--n-gpu-layers', '99',
            '--n-cpu-moe', '17', '--flash-attn', 'auto', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
            '--threads', AUTO_THREADS, '-np', '1',
            '--batch-size', '256', '--ubatch-size', '256', '--chat-template-kwargs', '{"reasoning_effort":"low"}', '--jinja',
        ]);
    });

    it('never passes the llama.cpp flags ik rejects, nor run-time repacking', () => {
        const full = ikLaunch({ ctxSize: 32768, nCpuMoe: 8, threads: 12, parallel: 2, loadMode: 'mlock',
            flashAttn: 'off', cacheTypeK: 'q8_0', cacheTypeV: 'q4_0', batchSize: 4096, ubatchSize: 1024 });
        for (const rejected of ['--no-webui', '-lv', '--kv-unified', '--load-mode', '-rtr', '--run-time-repack', '-fmoe']) {
            assert.equal(full.args.includes(rejected), false, rejected);
        }
        assert.ok(hasPair(full.args, '-np', '2'));
        assert.ok(full.args.includes('--mlock') && full.args.includes('--no-mmap'));
        const flags = (loadMode) => ikLaunch({ loadMode }).args.filter((arg) => ['--mlock', '--no-mmap'].includes(arg));
        assert.deepEqual([flags('auto'), flags('mmap'), flags('none'), flags('mmap+mlock')], [[], [], ['--no-mmap'], ['--mlock']]);
        assert.throws(() => ikLaunch({ loadMode: 'dio' }), /loadMode/);
        assert.ok(hasPair(full.args, '--flash-attn', 'off'));
        assert.ok(full.args.every((arg) => !SHELL_META.test(arg)));
        // Without the options, neither switch is passed.
        const plain = ikLaunch({});
        assert.equal(plain.args.includes('--mlock') || plain.args.includes('--no-mmap'), false);
        // ik turns --jinja off by default, so it is always passed.
        assert.ok(ikLaunch({}, { id: 'dense', requiresJinja: false }).args.includes('--jinja'));
    });

    it('splits the context across parallel slots, as ik has no unified KV cache', () => {
        assert.deepEqual(ikLlamaCppRunner.describeContext({ ctxSize: 16384, parallel: 2, batchSize: 512, ubatchSize: 512 }),
            { totalContext: 16384, perRequestContext: 8192, parallel: 2, kvUnified: false });
        assert.deepEqual(llamaCppRunner.describeContext({ ctxSize: 16384, parallel: 2, batchSize: 512, ubatchSize: 512 }),
            { totalContext: 16384, perRequestContext: 16384, parallel: 2, kvUnified: true });
    });

    it('reads its version from the build commit', () => {
        const stderr = 'version: 1 (20f7a72)\nbuilt with cc (Ubuntu 13.3.0) 13.3.0 for x86_64-linux-gnu\n';
        assert.deepEqual(ikLlamaCppRunner.detect(fakeSpawn({ status: 0, stdout: '', stderr })),
            { installed: true, version: '20f7a72', reason: null });
        const longer = ikLlamaCppRunner.detect(fakeSpawn({ status: 0, stdout: '', stderr: 'version: 4956 (20f7a72ed)\n' }));
        assert.equal(longer.version, '20f7a72');
        const other = ikLlamaCppRunner.detect(fakeSpawn({ status: 0, stdout: '', stderr: 'version: 5000 (abcdef1)\n' }));
        assert.equal(other.installed, true);
        assert.match(other.reason, /abcdef1 differs from pinned 20f7a72/);
        const upstream = ikLlamaCppRunner.detect(fakeSpawn({ status: 0, stdout: '', stderr: 'version: 0.5.0 (build 11159, commit 6b790a9)\n' }));
        assert.equal(upstream.installed, false);
    });

    it('its log report finds the device, the offload and the CUDA buffers', () => {
        const lines = [
            'ggml_cuda_init: found 1 CUDA devices:',
            '  Device 0: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6, VMM: yes, VRAM: 6143 MiB',
            'llm_load_tensors: offloaded 25/25 layers to GPU',
            'llm_load_tensors:      CUDA0 buffer size =  4073.34 MiB',
            'llm_load_tensors:        CPU buffer size =  7100.00 MiB',
            'llama_kv_cache_init:      CUDA0 KV buffer size =   384.00 MiB',
            'llama_init_from_model:      CUDA0 compute buffer size =   175.00 MiB',
        ].map((line, index) => ({ seq: index + 1, line }));
        assert.deepEqual(ikLlamaCppRunner.parseReport(lines), {
            modelMiB: 4073.34, kvMiB: 384, computeMiB: 175, offloaded: { layers: 25, of: 25 },
            device: 'CUDA0 (NVIDIA GeForce RTX 3060 Laptop GPU)', totalMiB: 4632,
        });
        assert.equal(ikLlamaCppRunner.parseReport, parseRunnerReport);
    });
});

describe('Ollama runner', () => {
    const dataDir = '/data/local-llm';
    const launch = (params, port = 11434) => ollamaRunner.buildLaunch({ params, port, dataDir, model: MODEL });

    it('builds serve with a loopback, single-model env', () => {
        const { command, args, env } = launch({ kvCacheType: 'q8_0' }, 18081);
        assert.equal(command, '/opt/ollama/bin/ollama');
        assert.deepEqual(args, ['serve']);
        assert.deepEqual(env, {
            HOME: '/data/local-llm/home',
            OLLAMA_MODELS: '/data/local-llm/models/ollama',
            OLLAMA_HOST: '127.0.0.1:18081',
            OLLAMA_NUM_PARALLEL: '1',
            OLLAMA_MAX_LOADED_MODELS: '1',
            OLLAMA_KV_CACHE_TYPE: 'q8_0',
            OLLAMA_CONTEXT_LENGTH: '4096',
            OLLAMA_KEEP_ALIVE: '30m',
            OLLAMA_NO_CLOUD: '1',
            LD_LIBRARY_PATH: '/usr/local/nvidia/lib64'
        });
        assert.ok(env.HOME.startsWith('/data/'));
    });

    it('sets OLLAMA_FLASH_ATTENTION only when chosen', () => {
        assert.equal(launch({ flashAttention: true }).env.OLLAMA_FLASH_ATTENTION, '1');
        assert.equal(launch({ flashAttention: false }).env.OLLAMA_FLASH_ATTENTION, '0');
        assert.ok(!Object.hasOwn(launch({}).env, 'OLLAMA_FLASH_ATTENTION'));
    });

    it('rejects bad dataDir, port and params', () => {
        assertCode(() => ollamaRunner.buildLaunch({ params: {}, port: 11434, dataDir: 'data' }), 'invalid_launch');
        assertCode(() => launch({}, 22), 'invalid_launch');
        for (const keepAlive of ['5m;id', '1d', '00m', '-2', '123456m', '']) {
            assertParamError(() => launch({ keepAlive }), 'keepAlive');
        }
        assertParamError(() => launch({ kvCacheType: 'q8_0 ' }), 'kvCacheType');
        assertParamError(() => launch({ numGpu: -1 }), 'numGpu');
        assertParamError(() => launch({ OLLAMA_HOST: '0.0.0.0' }), 'OLLAMA_HOST');
        assert.equal(launch({ keepAlive: '-1' }).env.OLLAMA_KEEP_ALIVE, '-1');
        assert.equal(launch({ keepAlive: '24h' }).env.OLLAMA_KEEP_ALIVE, '24h');
    });

    it('maps request options', () => {
        assert.deepEqual(ollamaRunner.requestOptions({ numCtx: 8192, numGpu: 20, numThread: 6, keepAlive: '5m' }), {
            options: { num_ctx: 8192, num_gpu: 20, num_thread: 6 },
            keep_alive: '5m'
        });
        assert.deepEqual(ollamaRunner.requestOptions({}), { options: { num_ctx: 4096 }, keep_alive: '30m' });
        assert.equal(ollamaRunner.requestOptions({ keepAlive: '-1' }).keep_alive, -1);
        assert.equal(ollamaRunner.requestOptions({ numGpu: 0 }).options.num_gpu, 0);
        assert.deepEqual(ollamaRunner.describeContext({ numCtx: 8192 }), {
            totalContext: 8192, perRequestContext: 8192, parallel: 1, kvUnified: false
        });
    });

    it('parses the client version and reports ENOENT', () => {
        const stdout = 'Warning: could not connect to a running Ollama instance\nWarning: client version is 0.34.4\n';
        const found = ollamaRunner.detect(fakeSpawn({ status: 0, stdout, stderr: '' }));
        assert.deepEqual(found, { installed: true, version: '0.34.4', reason: null });
        const server = ollamaRunner.detect(fakeSpawn({ status: 0, stdout: 'ollama version is 0.34.4\n' }));
        assert.equal(server.version, '0.34.4');
        const enoent = Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' });
        assert.equal(ollamaRunner.detect(fakeSpawn({ error: enoent })).installed, false);
    });
});

describe('vLLM runner', () => {
    const throwingSpawn = () => {
        throw new Error('must not spawn');
    };

    it('detects through the installer, never by spawning', async () => {
        assert.equal(vllmRunner.supported, true);
        assert.equal(vllmRunner.pinnedVersion, '0.30.0');
        // Installed on demand: without an installer lock entry it cannot be installed.
        assert.deepEqual(await vllmRunner.detect({ spawnSync: throwingSpawn }), {
            installed: false,
            version: null,
            reason: 'This image\'s runner lock has no vLLM entry.'
        });
        // It needs its runnable copy, weights and a GPU share to launch.
        assertCode(() => vllmRunner.buildLaunch({ params: {}, port: 18082, apiKey: API_KEY, model: MODEL }), 'invalid_launch');
    });

    it('validates vLLM params', () => {
        assert.equal(vllmRuntime, vllmRunner);
        const values = vllmRunner.normalizeParams({ maxModelLen: 8192 });
        assert.equal(values.gpuMemoryUtilization, null);
        assert.equal(values.maxNumSeqs, 1);
        assert.equal(values.quantization, null);
        assert.equal(values.enforceEager, true);
        assertParamError(() => vllmRunner.normalizeParams({ gpuMemoryUtilization: 1 }), 'gpuMemoryUtilization');
        assertParamError(() => vllmRunner.normalizeParams({ quantization: 'awq;id' }), 'quantization');
        assertParamError(() => vllmRunner.normalizeParams({ cpuOffloadParams: 'experts.*' }), 'cpuOffloadParams');
    });
});

describe('runner registry', () => {
    it('resolves known runners and rejects others', () => {
        // Intended change (Phase 3): llama.cpp's CPU build is registered, and offered on the cpu profile only.
        assert.deepEqual(Object.keys(RUNNERS), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio', 'llama.cpp-cpu']);
        assert.equal(getRunner('llama.cpp'), llamaCppRunner);
        assert.equal(getRunner('ollama'), ollamaRunner);
        for (const id of ['toString', 'constructor', 'llamacpp', undefined, null]) {
            assertCode(() => getRunner(id), 'unknown_runner');
        }
    });

    it('summarizes runners with JSON-serializable, frozen schemas', () => {
        const summaries = runnerSummaries();
        assert.deepEqual(summaries.map((s) => s.id), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio']);
        for (const summary of summaries) {
            assert.deepEqual(Object.keys(summary).sort(),
                ['basicParams', 'displayName', 'id', 'moeParams', 'paramSchema', 'pinnedVersion', 'supported', 'weightFormat']);
            assert.deepEqual(JSON.parse(JSON.stringify(summary.paramSchema)), summary.paramSchema);
            assert.ok(Object.isFrozen(summary.paramSchema.properties));
            for (const prop of Object.values(summary.paramSchema.properties)) {
                assert.equal(typeof prop.title, 'string');
                assert.equal(typeof prop.description, 'string');
            }
        }
    });
});

describe('argv safety', () => {
    it('never puts shell metacharacters from input into argv', () => {
        const hostile = ['; rm -rf /', '$(id)', '`id`', '| nc x 1', '&& id', '> /etc/passwd', 'a\nb'];
        for (const value of hostile) {
            for (const field of ['flashAttn', 'cacheTypeK', 'cacheTypeV', 'ctxSize', 'threads']) {
                assertParamError(() => llamaLaunch({ [field]: value }), field);
            }
            assertParamError(() => llamaLaunch({ chatTemplateKwargs: { reasoning_effort: value } }));
            assertParamError(() => ollamaRunner.normalizeParams({ keepAlive: `5m${value}` }), 'keepAlive');
        }
        const full = llamaLaunch({
            ctxSize: 32768, nCpuMoe: 8, threads: 12, parallel: 2, loadMode: 'mlock',
            flashAttn: 'on', cacheTypeK: 'q8_0', cacheTypeV: 'q4_0', batchSize: 4096, ubatchSize: 1024
        }, { model: { id: 'm', requiresJinja: true } });
        assert.ok(full.args.every((arg) => !SHELL_META.test(arg)));
    });
});

it('version probes get a minimal environment without tokens or agent secrets', (t) => {
    const saved = { HF_TOKEN: process.env.HF_TOKEN, PLOINKY_AGENT_SECRET: process.env.PLOINKY_AGENT_SECRET };
    process.env.HF_TOKEN = 'hf_secret_token_value';
    process.env.PLOINKY_AGENT_SECRET = 'agent-secret';
    t.after(() => {
        for (const [key, value] of Object.entries(saved)) {
            if (value === undefined) delete process.env[key];
            else process.env[key] = value;
        }
    });
    for (const runner of [llamaCppRunner, ikLlamaCppRunner, ollamaRunner]) {
        let seen = null;
        runner.detect({ spawnSync: (_file, _args, options) => { seen = options.env; return { status: 0, stdout: '', stderr: '' }; } });
        assert.deepEqual(Object.keys(seen).sort(), ['HOME', 'LANG', 'LD_LIBRARY_PATH', 'PATH'], runner.id);
        assert.equal(seen.LD_LIBRARY_PATH, '/usr/local/nvidia/lib64');
    }
});

// ---------------------------------------------------------------- Ollama on the cpu profile (DS005)

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const M1_TOTAL = 6036128 * 1024;
const M1_AVAILABLE = 3122576 * 1024;
const OLLAMA_MODEL = Object.freeze({ id: 'qwen', displayName: 'Qwen 0.5B', seed: true, contextLength: 32768, memory: Object.freeze({ layers: 24, kvBytesPerToken: 12288 }) });
const CPU_DECISION = Object.freeze({ cause: 'absent', reason: 'No GPU is attached to this agent: GPU not applied to this Box yet' });

function admitOllamaOnCpu({ size, model = OLLAMA_MODEL, params, memory = { totalBytes: M1_TOTAL, availableBytes: M1_AVAILABLE }, cgroupMemory = null,
    disk = { freeBytes: 100 * GIB }, remainingDownloadBytes = 0 }) {
    const source = { type: 'ollama', tag: 'qwen2.5:0.5b', ...(size === undefined ? {} : { size }) };
    return admit({
        runner: ollamaRunner, model, source, params: params ?? ollamaRunner.normalizeParams({}, { model, profile: 'cpu' }),
        snapshot: { gpu: { available: false, state: 'absent' }, memory, cgroupMemory, disk }, profile: 'cpu', decision: CPU_DECISION, remainingDownloadBytes,
    });
}

test('Ollama on the cpu profile is admitted from its pinned size, and refused when the size is not pinned', () => {
    // The need: the pinned size, an f16 KV cache for numCtx 4096 (12,288 bytes a token), compute buffers and the server (768 MiB).
    const need = (size) => size + 12288 * 4096 + Math.round((135 + 0.16 * 512) * MIB) + 768 * MIB;
    const small = admitOllamaOnCpu({ size: 400 * MIB });
    assert.equal(small.status, 'ok', small.reason);
    assert.equal(small.estimate.weightsBytes, 400 * MIB);
    assert.equal(small.estimate.kvBytes, 48 * MIB);
    assert.equal(small.estimate.computeBytes, Math.round(216.92 * MIB));
    assert.equal(small.estimate.runtimeBytes, 768 * MIB);
    assert.equal(small.estimate.ramBytes, need(400 * MIB));
    assert.deepEqual([small.estimate.poolBytes, small.estimate.floorBytes, small.estimate.hostReserveBytes, small.estimate.measured],
        [M1_TOTAL, cpuFloorBytes(M1_TOTAL), cpuHostReserveBytes(M1_TOTAL), false]);
    assert.deepEqual(small.estimate.defaulted, []);
    assert.match(small.warnings[0], /^Runs on the CPU: no NVIDIA GPU is attached \(No GPU is attached to this agent: GPU not applied to this Box yet\)\. Generation is much slower than on a GPU\.$/);
    // A size that is not pinned is refused, with the way out, before any estimate.
    for (const size of [undefined, 0]) {
        const unpinned = admitOllamaOnCpu({ size });
        assert.equal(unpinned.status, 'incompatible');
        assert.equal(unpinned.reason, 'The tag\'s size is not pinned; update the model entry so it is pinned.');
        assert.equal(unpinned.estimate.ramBytes, undefined);
    }
    // Exactly the need plus the floor fits now; one byte less does not.
    const size = 1000 * MIB;
    const floor = cpuFloorBytes(M1_TOTAL);
    assert.equal(admitOllamaOnCpu({ size, memory: { totalBytes: M1_TOTAL, availableBytes: need(size) + floor } }).status, 'ok');
    const busy = admitOllamaOnCpu({ size, memory: { totalBytes: M1_TOTAL, availableBytes: need(size) + floor - 1 } });
    assert.equal(busy.status, 'insufficient-now');
    assert.match(busy.reason, /Other processes on this machine hold the rest/);
    // More than the pool less the host reserve is never possible, here and for the 13.8 GB seed tag.
    const reserve = cpuHostReserveBytes(M1_TOTAL);
    assert.equal(admitOllamaOnCpu({ size: M1_TOTAL - reserve - need(0) }).status, 'insufficient-now');
    const never = admitOllamaOnCpu({ size: M1_TOTAL - reserve - need(0) + 1 });
    assert.equal(never.status, 'incompatible');
    assert.match(never.reason, /which must keep 1\.50 GiB for the host\. Reduce the context or pick a smaller model\./);
    assert.equal(admitOllamaOnCpu({ size: 13793441244 }).status, 'incompatible');
    // A container memory limit caps the pool, and says so.
    const capped = admitOllamaOnCpu({ size: 1000 * MIB, cgroupMemory: { maxBytes: 2 * GIB, currentBytes: 0 } });
    assert.equal(capped.status, 'incompatible');
    assert.equal(capped.estimate.poolBytes, 2 * GIB);
    assert.ok(capped.warnings.some((warning) => /container memory limit of 2\.00 GiB applies/.test(warning)));
    // The disk the pull needs, and unreadable memory.
    assert.equal(admitOllamaOnCpu({ size: 400 * MIB, remainingDownloadBytes: 400 * MIB, disk: { freeBytes: 100 * MIB } }).status, 'insufficient-now');
    const unreadable = admitOllamaOnCpu({ size: 400 * MIB, memory: {} });
    assert.equal(unreadable.status, 'incompatible');
    assert.match(unreadable.reason, /cannot be read/);
    // The KV figure falls back to the default and is named; the context changes the need.
    const bare = admitOllamaOnCpu({ size: 400 * MIB, model: { id: 'q', displayName: 'Q', seed: true } });
    assert.deepEqual(bare.estimate.defaulted, ['memory.kvBytesPerToken']);
    assert.equal(bare.estimate.kvBytes, 256 * MIB);
    assert.match(bare.warnings.join(' '), /Estimated without memory\.kvBytesPerToken/);
    const bigContext = admitOllamaOnCpu({ size: 400 * MIB, params: ollamaRunner.normalizeParams({ numCtx: 8192 }, { model: OLLAMA_MODEL, profile: 'cpu' }) });
    assert.equal(bigContext.estimate.kvBytes, 96 * MIB);
});

describe('Ollama on the cpu profile: parameters, launch and requests', () => {
    const dataDir = '/data/local-llm';
    const launch = (params, options = {}) => ollamaRunner.buildLaunch({ params, port: 18434, dataDir, model: MODEL, profile: 'cpu', ...options });

    it('has its own parameters on cpu and none on unified memory; the others are as they were', () => {
        assert.deepEqual(Object.keys(ollamaRunner.paramSchemaFor('cpu').properties), ['numCtx', 'numThread', 'keepAlive']);
        assert.equal(ollamaRunner.paramSchemaFor('dedicated'), ollamaRunner.paramSchema);
        assert.equal(ollamaRunner.paramSchemaFor(), ollamaRunner.paramSchema);
        for (const profile of ['unified', null, 'gpu']) assert.equal(ollamaRunner.paramSchemaFor(profile), null, String(profile));
        assert.deepEqual(ollamaRunner.normalizeParams({}, { profile: 'cpu' }), { numCtx: 4096, numThread: null, keepAlive: '30m' });
        // What a GPU offers is not offered on the CPU, and the bounds hold.
        for (const field of ['numGpu', 'flashAttention', 'kvCacheType']) {
            assertParamError(() => ollamaRunner.normalizeParams({ [field]: field === 'kvCacheType' ? 'q8_0' : 1 }, { profile: 'cpu' }), field);
        }
        assertParamError(() => ollamaRunner.normalizeParams({ numCtx: 511 }, { profile: 'cpu' }), 'numCtx');
        assertParamError(() => ollamaRunner.normalizeParams({ numCtx: 131073 }, { profile: 'cpu' }), 'numCtx');
        assertParamError(() => ollamaRunner.normalizeParams({ numThread: 0 }, { profile: 'cpu' }), 'numThread');
        assert.equal(ollamaRunner.normalizeParams({ numThread: 3, numCtx: 512 }, { profile: 'cpu' }).numThread, 3);
        // Every other profile reads the parameters it always read, including on unified memory, where admission refuses the runner.
        for (const profile of [undefined, 'dedicated', 'unified']) {
            assert.deepEqual(ollamaRunner.normalizeParams({}, { profile }), { numCtx: 4096, numGpu: null, numThread: null, flashAttention: null, kvCacheType: 'f16', keepAlive: '30m' }, String(profile));
        }
        // The model's recommended values are read for the profile.
        const recommended = { id: 'm', recommended: { cpu: { ollama: { numCtx: 2048 } }, dedicated: { ollama: { numCtx: 8192 } } } };
        assert.equal(ollamaRunner.normalizeParams({}, { model: recommended, profile: 'cpu' }).numCtx, 2048);
        assert.equal(ollamaRunner.normalizeParams({}, { model: recommended, profile: 'dedicated' }).numCtx, 8192);
        assert.equal(ollamaRunner.describeContext({ numCtx: 2048 }, { profile: 'cpu' }).totalContext, 2048);
    });

    it('launches the runnable copy with no driver path and no device visible, and the image\'s binary otherwise', () => {
        const fromCopy = launch({}, { runnerDir: '/opt/runners/ollama/0.34.4' });
        assert.equal(fromCopy.command, '/opt/runners/ollama/0.34.4/bin/ollama');
        assert.deepEqual(fromCopy.args, ['serve']);
        assert.deepEqual(fromCopy.env, {
            HOME: '/data/local-llm/home', OLLAMA_MODELS: '/data/local-llm/models/ollama', OLLAMA_HOST: '127.0.0.1:18434',
            OLLAMA_NUM_PARALLEL: '1', OLLAMA_MAX_LOADED_MODELS: '1', OLLAMA_KV_CACHE_TYPE: 'f16', OLLAMA_CONTEXT_LENGTH: '4096',
            OLLAMA_KEEP_ALIVE: '30m', OLLAMA_NO_CLOUD: '1', CUDA_VISIBLE_DEVICES: '',
        });
        assert.equal(launch({}).command, '/opt/ollama/bin/ollama');
        assert.equal(launch({ numCtx: 2048, keepAlive: '5m' }).env.OLLAMA_CONTEXT_LENGTH, '2048');
        assert.ok(!Object.hasOwn(launch({}).env, 'LD_LIBRARY_PATH'));
        // A dedicated launch from a runnable copy keeps the driver path; the path of the copy must be absolute.
        const dedicated = launch({}, { profile: 'dedicated', runnerDir: '/opt/runners/ollama/0.34.4' });
        assert.equal(dedicated.env.LD_LIBRARY_PATH, '/usr/local/nvidia/lib64');
        assert.ok(!Object.hasOwn(dedicated.env, 'CUDA_VISIBLE_DEVICES'));
        assertCode(() => launch({}, { runnerDir: 'opt/runners/ollama' }), 'invalid_launch');
    });

    it('asks for no GPU layer in every request on cpu, and is as it was elsewhere', () => {
        assert.deepEqual(ollamaRunner.requestOptions({}, { profile: 'cpu' }), { options: { num_ctx: 4096, num_gpu: 0 }, keep_alive: '30m' });
        assert.deepEqual(ollamaRunner.requestOptions({ numCtx: 2048, numThread: 2, keepAlive: '-1' }, { profile: 'cpu' }),
            { options: { num_ctx: 2048, num_gpu: 0, num_thread: 2 }, keep_alive: -1 });
        assert.deepEqual(ollamaRunner.requestOptions({}, { profile: 'dedicated' }), { options: { num_ctx: 4096 }, keep_alive: '30m' });
        assert.deepEqual(ollamaRunner.requestOptions({ numGpu: 20 }), { options: { num_ctx: 4096, num_gpu: 20 }, keep_alive: '30m' });
    });

    it('is detected from the image\'s binary, then from the installer', async () => {
        const found = fakeSpawn({ status: 0, stdout: 'ollama version is 0.34.4\n' });
        // The image's binary wins, synchronously, and the installer is not asked.
        assert.equal(ollamaRunner.detect({ ...found, installer: { installable: () => { throw new Error('asked'); } } }).installed, true);
        const enoent = Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' });
        const missing = fakeSpawn({ error: enoent });
        // Without the binary and without a lock entry: as before.
        assert.match(ollamaRunner.detect({ ...missing, installer: { installable: () => false } }).reason, /Executable not found at \/opt\/ollama\/bin\/ollama/);
        assert.equal(ollamaRunner.detect(missing).installed, false);
        // Without the binary and with a lock entry: the install record says.
        const installer = (installed) => ({ installable: (id) => id === 'ollama', describe: async () => ({ installed, version: '0.34.4' }) });
        assert.deepEqual(await ollamaRunner.detect({ ...missing, installer: installer(true) }), { installed: true, version: '0.34.4', reason: null });
        const absent = await ollamaRunner.detect({ ...missing, installer: installer(false) });
        assert.equal(absent.installed, false);
        assert.match(absent.reason, /Install it under Runners/);
    });
});

// ---------------------------------------------------------------- llama.cpp's CPU build (DS004, DS005)

describe('llama.cpp (CPU build)', () => {
    const cpuLaunch = (runner, extra = {}) => runner.buildLaunch({
        artifactPath: '/data/models/gguf/model.gguf', params: {}, port: 18080, apiKey: API_KEY, model: { id: 'm', contextLength: 32768 }, profile: 'cpu', ...extra,
    });

    it('is for the cpu profile alone: a schema and a policy there, none elsewhere', () => {
        assert.equal(getRunner('llama.cpp-cpu'), llamaCppCpuRunner);
        assert.deepEqual([llamaCppCpuRunner.id, llamaCppCpuRunner.displayName, llamaCppCpuRunner.port, llamaCppCpuRunner.weightFormat], ['llama.cpp-cpu', 'llama.cpp (CPU build)', 18085, 'gguf']);
        assert.deepEqual(llamaCppCpuRunner.profiles, ['cpu']);
        assert.equal(llamaCppCpuRunner.executable, null);
        assert.ok(llamaCppCpuRunner.paramSchemaFor('cpu'));
        for (const profile of ['dedicated', 'unified', null, 'gpu']) assert.equal(llamaCppCpuRunner.paramSchemaFor(profile), null, String(profile));
        assert.equal(llamaCppCpuRunner.paramSchema, null);
        // No GPU policy at all; only admitCpu.
        assert.equal(typeof llamaCppCpuRunner.admitCpu, 'function');
        assert.equal('admit' in llamaCppCpuRunner, false);
        assert.equal('admitUnified' in llamaCppCpuRunner, false);
        for (const profile of ['dedicated', 'unified']) {
            assertParamError(() => llamaCppCpuRunner.normalizeParams({}, { profile }));
            assert.throws(() => llamaCppCpuRunner.normalizeParams({}, { profile }), /llama\.cpp \(CPU build\) has no parameters for the /);
        }
        assert.equal(llamaCppCpuRunner.normalizeParams({}, { profile: 'cpu', model: { id: 'm', contextLength: 32768 } }).ctxSize, 4096);
        // Admission: sized on cpu like llama.cpp, and never on a GPU profile.
        const model = { id: 'm', displayName: 'M', seed: true, contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 12288 } };
        const source = { type: 'huggingface', size: 400 * MIB };
        const snapshot = { gpu: { available: false, state: 'absent' }, memory: { totalBytes: M1_TOTAL, availableBytes: M1_AVAILABLE }, disk: { freeBytes: 100 * GIB } };
        const params = llamaCppCpuRunner.normalizeParams({}, { model, profile: 'cpu' });
        assert.equal(admit({ runner: llamaCppCpuRunner, model, source, params, snapshot, profile: 'cpu', decision: CPU_DECISION }).status, 'ok');
        const same = admit({ runner: llamaCppRunner, model, source, params, snapshot, profile: 'cpu', decision: CPU_DECISION });
        assert.deepEqual(admit({ runner: llamaCppCpuRunner, model, source, params, snapshot, profile: 'cpu', decision: CPU_DECISION }).estimate, same.estimate);
        const gpuSnapshot = { ...snapshot, gpu: { available: true, name: 'GPU', memoryModel: 'dedicated', totalBytes: 6 * GIB, freeBytes: 6 * GIB } };
        for (const profile of ['dedicated', 'unified']) {
            const refused = admit({ runner: llamaCppCpuRunner, model, source, params, snapshot: gpuSnapshot, profile });
            assert.equal(refused.status, 'incompatible', profile);
            assert.equal(refused.reason, 'llama.cpp (CPU build) is not supported in this release.', profile);
        }
    });

    it('launches the runnable copy\'s llama-server with exactly llama.cpp\'s CPU arguments and environment', () => {
        const build = cpuLaunch(llamaCppCpuRunner, { runnerDir: '/opt/runners/llama.cpp-cpu/b11295' });
        const image = cpuLaunch(llamaCppRunner);
        assert.equal(build.command, '/opt/runners/llama.cpp-cpu/b11295/llama-server');
        assert.equal(image.command, '/opt/llama.cpp/llama-server');
        assert.deepEqual(build.args, image.args);
        assert.ok(hasPair(build.args, '--device', 'none') && hasPair(build.args, '--n-gpu-layers', '0'));
        assert.deepEqual(build.env, { CUDA_VISIBLE_DEVICES: '' });
        // The copy's path must be given and absolute: there is no binary of its own to fall back to.
        assertCode(() => cpuLaunch(llamaCppCpuRunner), 'invalid_launch');
        assertCode(() => cpuLaunch(llamaCppCpuRunner, { runnerDir: 'opt/runners/llama.cpp-cpu' }), 'invalid_launch');
        // llama.cpp ignores a runnable copy: its binary is the image's.
        assert.equal(cpuLaunch(llamaCppRunner, { runnerDir: '/opt/runners/x' }).command, '/opt/llama.cpp/llama-server');
    });

    it('is shown on the cpu profile only', () => {
        assert.equal(offeredIn(llamaCppCpuRunner, 'cpu'), true);
        for (const profile of ['dedicated', 'unified', null, undefined]) assert.equal(offeredIn(llamaCppCpuRunner, profile), false, String(profile));
        for (const runner of Object.values(RUNNERS).filter((candidate) => candidate !== llamaCppCpuRunner)) {
            for (const profile of ['cpu', 'dedicated', 'unified', null]) assert.equal(offeredIn(runner, profile), true, `${runner.id} ${profile}`);
        }
        assert.equal(runnerSummaries().some((summary) => summary.id === 'llama.cpp-cpu'), false);
        assert.equal(runnerSummaries(RUNNERS, 'unified').some((summary) => summary.id === 'llama.cpp-cpu'), false);
        assert.equal(runnerSummaries(RUNNERS, 'cpu').at(-1).id, 'llama.cpp-cpu');
        assert.ok(runnerSummary(llamaCppCpuRunner, 'cpu').paramSchema);
    });

    it('is detected from the installer, never by starting it', async () => {
        const spawnSync = () => assert.fail('the binary is not started to be detected');
        const installer = (installed) => ({ installable: (id) => id === 'llama.cpp-cpu', describe: async () => ({ installed, version: 'b11295' }) });
        assert.deepEqual(await llamaCppCpuRunner.detect({ spawnSync, installer: installer(true) }), { installed: true, version: 'b11295', reason: null });
        const absent = await llamaCppCpuRunner.detect({ spawnSync, installer: installer(false) });
        assert.equal(absent.installed, false);
        assert.match(absent.reason, /Install it under Runners/);
        assert.match((await llamaCppCpuRunner.detect({ spawnSync, installer: { installable: () => false } })).reason, /no llama\.cpp \(CPU build\) entry/);
        assert.equal((await llamaCppCpuRunner.detect({ spawnSync })).installed, false);
        // A runner in the image is still probed, synchronously.
        assert.equal(llamaCppRunner.detect(fakeSpawn({ status: 0, stdout: 'version: 1 (x)\nbuild 11159\n' })).installed, true);
    });

    it('needs an executable or a runnable name, and leaves the image\'s runners as they were', () => {
        const dialect = { quietArgs: [], loadArgs: () => [], jinja: () => false, parseVersion: () => null, cpu: true };
        assert.throws(() => createLlamaServerRunner({ id: 'x', displayName: 'X', executable: null, pinnedVersion: '1', port: 18999, dialect }), /needs an executable or a runnable name/);
        assert.ok(createLlamaServerRunner({ id: 'x', displayName: 'X', executable: '/opt/x/server', pinnedVersion: '1', port: 18999, dialect }));
        // llama.cpp and ik_llama.cpp keep every policy and their dedicated schema.
        for (const runner of [llamaCppRunner, ikLlamaCppRunner]) {
            assert.equal(runner.paramSchema, runner.paramSchemaFor('dedicated'));
            assert.equal(typeof runner.admit, 'function');
            assert.equal('profiles' in runner, false);
        }
        assert.equal(typeof llamaCppRunner.admitUnified, 'function');
        assert.equal(typeof llamaCppRunner.admitCpu, 'function');
        assert.equal('admitUnified' in ikLlamaCppRunner || 'admitCpu' in ikLlamaCppRunner, false);
    });
});
