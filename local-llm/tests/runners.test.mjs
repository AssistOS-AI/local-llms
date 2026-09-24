import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ParamError, validateParams } from '../src/runners/params.mjs';
import { RUNNERS, getRunner, runnerSummaries } from '../src/runners/index.mjs';
import { ikLlamaCppRunner } from '../src/runners/ikLlamaCpp.mjs';
import { llamaCppRunner } from '../src/runners/llamaCpp.mjs';
import { parseRunnerReport } from '../src/controller/runnerProcess.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { ollamaRunner } from '../src/runners/ollama.mjs';
import { vllmRunner, vllmRuntime } from '../src/runners/vllm.mjs';

const API_KEY = 'k'.repeat(24) + '_-AZaz09k';
const MODEL = Object.freeze({ id: 'qwen3-8b-q4' });
const SHELL_META = /[;&|`$<>\n\\]/;

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
            '-np', '1', '--batch-size', '2048', '--ubatch-size', '512'
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
        const model = { id: 'gpt-oss-20b', recommended: { 'llama.cpp': { ctxSize: 65536, nCpuMoe: 12 } } };
        const values = llamaCppRunner.normalizeParams({}, { model });
        assert.equal(values.ctxSize, 65536);
        assert.equal(values.nCpuMoe, 12);
        assert.equal(values.parallel, 1);
        assert.equal(llamaCppRunner.normalizeParams({ ctxSize: 8192 }, { model }).ctxSize, 8192);
        const { args } = llamaLaunch({}, { model });
        assert.ok(hasPair(args, '--ctx-size', '65536'));
        assert.ok(hasPair(args, '--alias', 'gpt-oss-20b'));
        const bad = { id: 'x', recommended: { 'llama.cpp': { ctxSize: 1 } } };
        assertParamError(() => llamaCppRunner.normalizeParams({}, { model: bad }), 'ctxSize');
    });

    it('adds --jinja only when the model requires it', () => {
        assert.ok(llamaLaunch({}, { model: { id: 'm', requiresJinja: true } }).args.includes('--jinja'));
        assert.ok(!llamaLaunch({}).args.includes('--jinja'));
    });

    it('maps mlock/noMmap to --load-mode and omits the default', () => {
        const modeOf = (params) => {
            const { args } = llamaLaunch(params);
            const index = args.indexOf('--load-mode');
            return index === -1 ? null : args[index + 1];
        };
        assert.equal(modeOf({}), null);
        assert.equal(modeOf({ mlock: true }), 'mmap+mlock');
        assert.equal(modeOf({ noMmap: true }), 'none');
        assert.equal(modeOf({ mlock: true, noMmap: true }), 'mlock');
        const { args } = llamaLaunch({ mlock: true });
        assert.ok(!args.includes('--mlock') && !args.includes('--no-mmap'));
    });

    it('passes chatTemplateKwargs as one JSON argument', () => {
        const { args } = llamaLaunch({ chatTemplateKwargs: { reasoning_effort: 'high' } });
        assert.ok(hasPair(args, '--chat-template-kwargs', '{"reasoning_effort":"high"}'));
        assert.ok(!llamaLaunch({}).args.includes('--chat-template-kwargs'));
        assertParamError(() => llamaLaunch({ chatTemplateKwargs: { reasoning_effort: 'max' } }));
        assertParamError(() => llamaLaunch({ chatTemplateKwargs: { other: 1 } }), 'chatTemplateKwargs.other');
    });

    it('adds --threads only when set', () => {
        assert.ok(hasPair(llamaLaunch({ threads: 8 }).args, '--threads', '8'));
        assert.ok(!llamaLaunch({ threads: null }).args.includes('--threads'));
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
        const output = 'version: 0.4.1-dev (build 11125, commit 94256114c)\nbuilt with GNU 13\n';
        const { spawnSync, calls } = fakeSpawn({ status: 0, stdout: '', stderr: output });
        assert.deepEqual(llamaCppRunner.detect({ spawnSync }), { installed: true, version: 'b11125', reason: null });
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
        assert.match(other.reason, /differs from pinned b11125/);
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
        assert.equal(ikLlamaCppRunner.paramSchema, llamaCppRunner.paramSchema);
    });

    it('launches gpt-oss-20b with the flags ik understands, in the documented order', () => {
        const launch = ikLaunch({});
        assert.equal(launch.command, '/opt/ik_llama.cpp/llama-server');
        assert.deepEqual(launch.env, { LD_LIBRARY_PATH: '/usr/local/nvidia/lib64' });
        assert.deepEqual(launch.args, [
            '-m', PATH, '--host', '127.0.0.1', '--port', '18081', '--api-key', API_KEY,
            '--webui', 'none', '--alias', 'gpt-oss-20b', '--ctx-size', '16384', '--n-gpu-layers', '99',
            '--n-cpu-moe', '17', '--flash-attn', 'auto', '--cache-type-k', 'f16', '--cache-type-v', 'f16', '-np', '1',
            '--batch-size', '256', '--ubatch-size', '256', '--chat-template-kwargs', '{"reasoning_effort":"low"}', '--jinja',
        ]);
    });

    it('never passes the llama.cpp flags ik rejects, nor run-time repacking', () => {
        const full = ikLaunch({ ctxSize: 32768, nCpuMoe: 8, threads: 12, parallel: 2, mlock: true, noMmap: true,
            flashAttn: 'off', cacheTypeK: 'q8_0', cacheTypeV: 'q4_0', batchSize: 4096, ubatchSize: 1024 });
        for (const rejected of ['--no-webui', '-lv', '--kv-unified', '--load-mode', '-rtr', '--run-time-repack', '-fmoe']) {
            assert.equal(full.args.includes(rejected), false, rejected);
        }
        assert.ok(hasPair(full.args, '-np', '2'));
        assert.ok(full.args.includes('--mlock') && full.args.includes('--no-mmap'));
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
        const stdout = 'Warning: could not connect to a running Ollama instance\nWarning: client version is 0.34.3\n';
        const found = ollamaRunner.detect(fakeSpawn({ status: 0, stdout, stderr: '' }));
        assert.deepEqual(found, { installed: true, version: '0.34.3', reason: null });
        const server = ollamaRunner.detect(fakeSpawn({ status: 0, stdout: 'ollama version is 0.34.3\n' }));
        assert.equal(server.version, '0.34.3');
        const enoent = Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' });
        assert.equal(ollamaRunner.detect(fakeSpawn({ error: enoent })).installed, false);
    });
});

describe('unsupported runners', () => {
    const throwingSpawn = () => {
        throw new Error('must not spawn');
    };

    for (const runner of [vllmRunner]) {
        it(`${runner.id} reports unsupported without spawning`, () => {
            assert.equal(runner.supported, false);
            assert.equal(runner.pinnedVersion, null);
            assert.deepEqual(runner.detect({ spawnSync: throwingSpawn }), {
                installed: false,
                version: null,
                reason: 'Not supported or tested in this release; installable in a later release.'
            });
            assertCode(() => runner.buildLaunch({ params: {}, port: 18080, apiKey: API_KEY, model: MODEL }),
                'runner_unsupported');
        });
    }

    it('validates vLLM params', () => {
        assert.equal(vllmRuntime, vllmRunner);
        const values = vllmRunner.normalizeParams({ maxModelLen: 8192 });
        assert.equal(values.gpuMemoryUtilization, 0.9);
        assert.equal(values.maxNumSeqs, 1);
        assert.equal(values.quantization, null);
        assertParamError(() => vllmRunner.normalizeParams({ gpuMemoryUtilization: 1 }), 'gpuMemoryUtilization');
        assertParamError(() => vllmRunner.normalizeParams({ quantization: 'awq;id' }), 'quantization');
        assertParamError(() => vllmRunner.normalizeParams({ cpuOffloadParams: 'a b' }), 'cpuOffloadParams');
        assert.equal(vllmRunner.normalizeParams({ cpuOffloadParams: 'experts.*,lm_head' }).cpuOffloadParams,
            'experts.*,lm_head');
    });
});

describe('runner registry', () => {
    it('resolves known runners and rejects others', () => {
        assert.deepEqual(Object.keys(RUNNERS), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm']);
        assert.equal(getRunner('llama.cpp'), llamaCppRunner);
        assert.equal(getRunner('ollama'), ollamaRunner);
        for (const id of ['toString', 'constructor', 'llamacpp', undefined, null]) {
            assertCode(() => getRunner(id), 'unknown_runner');
        }
    });

    it('summarizes runners with JSON-serializable, frozen schemas', () => {
        const summaries = runnerSummaries();
        assert.deepEqual(summaries.map((s) => s.id), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm']);
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
            ctxSize: 32768, nCpuMoe: 8, threads: 12, parallel: 2, mlock: true, noMmap: true,
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
