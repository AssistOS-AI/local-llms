import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { TOOL_NAMES, TOOL_OPERATIONS, assertAdmin, authInfoFromEnvelope, handleTool } from '../tools/local_llm_tool.mjs';
import { buildRunnerRequest, completionStats, respond } from '../src/chatResponder.mjs';
import { GATEWAY_MODEL, runTestPrompt } from '../src/testPrompt.mjs';

const ROOT = new URL('..', import.meta.url);
const read = (file) => JSON.parse(fs.readFileSync(new URL(file, ROOT), 'utf8'));

test('every tool is admin-tagged, runs from /code, and names itself', () => {
    const config = read('mcp-config.json');
    assert.deepEqual(config.tools.map((tool) => tool.name).sort(), [...TOOL_NAMES].sort());
    for (const tool of config.tools) {
        assert.deepEqual(tool.tags, ['admin'], tool.name);
        assert.equal(tool.cwd, '/code', tool.name);
        assert.deepEqual(tool.env, { TOOL_NAME: tool.name }, tool.name);
        assert.equal(tool.command, 'node');
        assert.deepEqual(tool.args, ['/code/tools/local_llm_tool.mjs']);
        assert.equal(tool.inputSchema.type, 'object');
        assert.equal(tool.inputSchema.additionalProperties, false, tool.name);
        assert.ok(tool.timeoutMs <= 600000, tool.name);
    }
});

test('the manifest declares GPU access and nothing else, and closes the agent-port relay', () => {
    const manifest = read('manifest.json');
    // containerSecurity.gpu implies the one CDI device (Ploinky D14), so no
    // runtimePolicy device entry and no other llmRuntime setting is needed.
    assert.deepEqual(manifest.containerSecurity, { gpu: true });
    assert.equal(manifest.llmRuntime, undefined);
    assert.equal(manifest.routerAccess.agentPorts, false);
    assert.deepEqual(manifest.volumes, { '.data/local-llm': '/data' });
    assert.equal(manifest.agent, 'exec node /code/src/main.mjs');
    assert.deepEqual(manifest.readiness, { protocol: 'mcp' });
    assert.equal(manifest.endpoints.chatCompletions.cwd, '/code');
    assert.equal(manifest.endpoints.chatCompletions.supportsStream, true);
    assert.equal(manifest.profiles?.default?.ports, undefined);
});

test('a missing invocation, a plain user and an admin guest get admin_required from every tool', async () => {
    const call = async () => assert.fail('never reaches the controller');
    const envelope = { tool: 'local_llm_overview', input: {} };
    assert.equal(await authInfoFromEnvelope(envelope, { loadAuth: () => assert.fail('no grant, no verification') }), null);
    for (const authInfo of [null, {}, { user: { roles: ['user'] } }, { user: { roles: ['admin', 'guest'] } }]) {
        for (const name of TOOL_NAMES) {
            await assert.rejects(
                () => handleTool(name, { requestId: 'r-12345678', modelId: 'm', runnerId: 'ollama', prompt: 'x' },
                    { authInfo, call, testPrompt: async () => assert.fail('not authorized') }),
                { code: 'admin_required' },
                `${name} ${JSON.stringify(authInfo)}`,
            );
        }
    }
    assert.doesNotThrow(() => assertAdmin({ user: { roles: ['admin'] } }));
});

test('an admin reaches the controller operation of each tool', async () => {
    const seen = [];
    const call = async (op, args) => { seen.push([op, args]); return { ok: true }; };
    const authInfo = { user: { roles: ['admin'] } };
    await handleTool('local_llm_run', { requestId: 'req-000001', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', params: { ctxSize: 8192 } },
        { authInfo, call });
    await handleTool('local_llm_status', { sinceSeq: 5 }, { authInfo, call });
    await handleTool('local_llm_weights_delete', { modelId: 'gpt-oss-20b', runnerId: 'ollama' }, { authInfo, call });
    const prompt = await handleTool('local_llm_test_prompt', { prompt: 'hi' }, { authInfo, call, testPrompt: async ({ prompt }) => ({ echo: prompt }) });
    assert.deepEqual(seen, [
        ['run', { requestId: 'req-000001', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', params: { ctxSize: 8192 }, replace: false }],
        ['status', { sinceSeq: 5 }],
        ['deleteWeights', { modelId: 'gpt-oss-20b', runnerId: 'ollama' }],
    ]);
    assert.deepEqual(prompt, { echo: 'hi' });
    assert.equal(Object.keys(TOOL_OPERATIONS).length + 1, TOOL_NAMES.length);
});

test('the chat responder forwards to the ready runner with its key, and answers 503 otherwise', async () => {
    const notReady = await respond({ request: { messages: [] } }, {
        call: async () => { throw Object.assign(new Error('No local model is ready.'), { code: 'not_ready' }); },
    });
    assert.deepEqual(notReady, { status: 503, code: 'not_ready', message: 'No local model is ready.' });
    const invalid = await respond({ request: {} }, { call: async () => assert.fail('validated first') });
    assert.equal(invalid.status, 400);

    let forwarded;
    let written = '';
    const result = await respond({ request: { model: 'anything', messages: [{ role: 'user', content: 'hi' }], max_tokens: 5, secret: 'x' } }, {
        call: async () => ({ baseUrl: 'http://127.0.0.1:18080', apiKey: 'k'.repeat(43), model: 'gpt-oss-20b' }),
        fetchImpl: async (url, init) => {
            forwarded = { url, init };
            return { ok: true, json: async () => ({ choices: [{ message: { content: 'hello' } }] }) };
        },
        out: { write(text) { written += text; } },
    });
    assert.equal(result, null);
    assert.equal(forwarded.url, 'http://127.0.0.1:18080/v1/chat/completions');
    assert.equal(forwarded.init.headers.authorization, `Bearer ${'k'.repeat(43)}`);
    assert.deepEqual(JSON.parse(forwarded.init.body), { model: 'gpt-oss-20b', messages: [{ role: 'user', content: 'hi' }], max_tokens: 5 });
    assert.equal(JSON.parse(written).choices[0].message.content, 'hello');
    assert.deepEqual(buildRunnerRequest({ messages: [], stream: true }, { model: 'gpt-oss:20b' }),
        { messages: [], stream: true, max_tokens: 8192, model: 'gpt-oss:20b' });
});

test('the chat responder reports runner timings from a response and from the last stream chunk', async () => {
    const timings = { prompt_n: 20, prompt_per_second: 812.5, predicted_n: 7, predicted_per_second: 61.25 };
    assert.deepEqual(completionStats({ timings, usage: { prompt_tokens: 20, completion_tokens: 7 } }), {
        promptTokens: 20, completionTokens: 7, promptTokensPerSecond: 812.5, generationTokensPerSecond: 61.25, source: 'runner timings',
    });
    assert.deepEqual(completionStats({ usage: { prompt_tokens: 3, completion_tokens: 4 } }).source, 'usage');
    assert.equal(completionStats({ choices: [] }), null);

    const target = { baseUrl: 'http://127.0.0.1:18080', apiKey: 'k', model: 'gpt-oss-20b' };
    const reported = [];
    const call = async (op, args) => {
        if (op === 'chatTarget') return target;
        reported.push([op, args]);
        return { recorded: true };
    };
    await respond({ request: { messages: [{ role: 'user', content: 'hi' }] } }, {
        call,
        fetchImpl: async () => ({ ok: true, json: async () => ({ choices: [], timings }) }),
        out: { write() {} },
    });
    const sse = [
        'data: {"choices":[{"delta":{"content":"he"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"llo"}}],"timings":{"prompt_n":5,"predicted_n":2,"predicted_per_second":44}}\n',
        '\ndata: [DONE]\n\n',
    ];
    let streamed = '';
    await respond({ request: { messages: [], stream: true } }, {
        call,
        fetchImpl: async () => ({ ok: true, body: sse.map((part) => new TextEncoder().encode(part)) }),
        out: { write(chunk) { streamed += new TextDecoder().decode(chunk); } },
    });
    assert.equal(streamed, sse.join(''));
    assert.deepEqual(reported.map(([op, args]) => [op, args.generationTokensPerSecond, args.completionTokens]), [
        ['recordCompletion', 61.25, 7],
        ['recordCompletion', 44, 2],
    ]);
    // A failed report never fails the completion.
    await respond({ request: { messages: [] } }, {
        call: async (op) => { if (op === 'chatTarget') return target; throw new Error('socket gone'); },
        fetchImpl: async () => ({ ok: true, json: async () => ({ timings }) }),
        out: { write() {} },
    });
});

test('the admin test prompt calls only the active loopback runner, with bounds', async () => {
    const recorded = [];
    let sent;
    const call = async (op, args) => {
        if (op === 'chatTarget') return { runnerId: 'llama.cpp', modelId: 'gpt-oss-20b', model: 'gpt-oss-20b', baseUrl: 'http://127.0.0.1:18080', apiKey: 'secret-key' };
        if (op === 'recordCompletion') { recorded.push(args); return { recorded: true }; }
        throw new Error(op);
    };
    const fetchImpl = async (url, init) => {
        sent = { url, init, body: JSON.parse(init.body) };
        return { ok: true, json: async () => ({
            choices: [{ message: { content: 'PONG', reasoning_content: 'think' }, finish_reason: 'stop' }],
            timings: { prompt_n: 9, prompt_per_second: 300, predicted_n: 2, predicted_per_second: 60 },
        }) };
    };
    const result = await runTestPrompt({ prompt: 'Reply with exactly: PONG', maxTokens: 16, call, fetchImpl });
    assert.equal(sent.url, 'http://127.0.0.1:18080/v1/chat/completions');
    assert.equal(sent.init.headers.authorization, 'Bearer secret-key');
    assert.deepEqual(sent.body, { messages: [{ role: 'user', content: 'Reply with exactly: PONG' }], max_tokens: 16, model: 'gpt-oss-20b' });
    assert.ok(sent.init.signal instanceof AbortSignal);
    assert.equal(result.text, 'PONG');
    assert.equal(result.generationTokensPerSecond, 60);
    assert.equal(result.statsSource, 'runner timings');
    assert.equal(result.gatewayModel, GATEWAY_MODEL);
    assert.equal(GATEWAY_MODEL, 'soul_gateway/local-llms/local-llm/default');
    assert.equal(recorded[0].generationTokensPerSecond, 60);
    assert.equal(JSON.stringify(result).includes('secret-key'), false);

    // Ollama's /v1 reports counts only: the speed is labelled as a wall-clock estimate.
    let tick = 0;
    const ollama = await runTestPrompt({
        prompt: 'hi', call, clock: () => (tick += 500),
        fetchImpl: async () => ({ ok: true, json: async () => ({ choices: [{ message: { content: 'x' } }], usage: { prompt_tokens: 3, completion_tokens: 10 } }) }),
    });
    assert.equal(ollama.generationTokensPerSecond, 20);
    assert.equal(ollama.statsSource, 'token count over wall clock');

    const never = async () => assert.fail('no request');
    await assert.rejects(() => runTestPrompt({ prompt: 'x'.repeat(4001), call, fetchImpl: never }), { code: 'invalid_prompt' });
    await assert.rejects(() => runTestPrompt({ prompt: 'x', maxTokens: 1025, call, fetchImpl: never }), { code: 'invalid_prompt' });
    await assert.rejects(() => runTestPrompt({ prompt: 'x', maxTokens: 1.5, call, fetchImpl: never }), { code: 'invalid_prompt' });
    const remote = async (op) => (op === 'chatTarget' ? { baseUrl: 'http://10.0.0.5:18080', model: 'm' } : null);
    await assert.rejects(() => runTestPrompt({ prompt: 'x', call: remote, fetchImpl: never }), { code: 'runner_not_local' });
    const notReady = async () => { throw Object.assign(new Error('No local model is ready.'), { code: 'not_ready' }); };
    await assert.rejects(() => runTestPrompt({ prompt: 'x', call: notReady, fetchImpl: never }), { code: 'not_ready' });
});

test('the manifest keeps local-llm out of the generic agent tier', () => {
    const manifest = read('manifest.json');
    assert.deepEqual(manifest.capabilities.tags, ['local-llm']);
    assert.match(manifest.about, /GPU/, 'the Marketplace shows `about`');
});

test('the test prompt tool describes the loopback admin check, not the Soul Gateway path', () => {
    const tool = read('mcp-config.json').tools.find((entry) => entry.name === 'local_llm_test_prompt');
    assert.doesNotMatch(tool.description, /Soul Gateway|AchillesAgentLib/i);
    assert.match(tool.description, /admin/i);
    assert.match(tool.description, /loopback/i);
});

test('the chat responder caps completions, allows one choice and bounds the runner call', { timeout: 10_000 }, async () => {
    const target = { baseUrl: 'http://127.0.0.1:18080', apiKey: 'k', model: 'gpt-oss-20b' };
    assert.equal(buildRunnerRequest({ messages: [], max_tokens: 100_000 }, target).max_tokens, 8192);
    assert.equal(buildRunnerRequest({ messages: [], max_completion_tokens: 9000 }, target).max_completion_tokens, 8192);
    assert.equal(buildRunnerRequest({ messages: [], max_tokens: 256 }, target).max_tokens, 256);
    const call = async () => target;
    const never = async () => assert.fail('validated before the runner');
    for (const request of [{ messages: [], n: 4 }, { messages: [], max_tokens: -5 }, { messages: [], max_tokens: 1.5 }]) {
        const result = await respond({ request }, { call, fetchImpl: never, out: { write() {} } });
        assert.equal(result.status, 400, JSON.stringify(request));
        assert.equal(result.code, 'invalid_request');
    }
    // A runner that never answers is cut off at the deadline with a clean 504.
    let seenSignal = null;
    const hanging = (url, init) => {
        seenSignal = init.signal;
        return new Promise((resolve, reject) => init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true }));
    };
    const timedOut = await respond({ request: { messages: [] } }, { call, fetchImpl: hanging, out: { write() {} }, timeoutMs: 50 });
    assert.ok(seenSignal instanceof AbortSignal);
    assert.deepEqual({ status: timedOut.status, code: timedOut.code }, { status: 504, code: 'runner_timeout' });
});

test('a chat request without a token limit gets the documented default cap', async () => {
    const target = { baseUrl: 'http://127.0.0.1:18080', apiKey: 'k', model: 'gpt-oss-20b' };
    assert.equal(buildRunnerRequest({ messages: [] }, target).max_tokens, 8192);
    // A caller that sets either limit keeps only its own field.
    const onlyNew = buildRunnerRequest({ messages: [], max_completion_tokens: 64 }, target);
    assert.equal(onlyNew.max_completion_tokens, 64);
    assert.equal('max_tokens' in onlyNew, false);
    // The runner receives the default.
    let sent = null;
    const fetchImpl = async (url, init) => {
        sent = JSON.parse(init.body);
        return new Response(JSON.stringify({ choices: [] }), { status: 200, headers: { 'content-type': 'application/json' } });
    };
    await respond({ request: { messages: [] } }, { call: async () => target, fetchImpl, out: { write() {} } });
    assert.equal(sent.max_tokens, 8192);
});

test('with no GPU in the container, the reason says what to run on the host', async () => {
    const { readGpu } = await import('../src/controller/hardware.mjs');
    const missing = (command, args, options, callback) => callback(Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' }), '', '');
    const gpu = await readGpu({ execFileImpl: missing, nvidiaSmi: '/usr/local/nvidia/bin/nvidia-smi', env: {} });
    assert.equal(gpu.available, false);
    assert.match(gpu.reason, /`ploinky gpu status` shows why/);
    assert.match(gpu.reason, /revoked: run `ploinky gpu grant --agent local-llms\/local-llm`/);
    assert.match(gpu.reason, /not applied yet: run `ploinky start`/);
    // Ploinky tells a manifest-declared agent why it has no GPU (D14); that
    // reason is what Settings, local_llm_overview and local_llm_status show.
    const revoked = 'GPU access for local-llms/local-llm was revoked by the operator; on the host run `ploinky gpu grant --agent local-llms/local-llm`';
    const told = await readGpu({
        execFileImpl: missing,
        nvidiaSmi: '/usr/local/nvidia/bin/nvidia-smi',
        env: { PLOINKY_GPU_STATUS: 'unavailable', PLOINKY_GPU_REASON: revoked },
    });
    assert.equal(told.reason, `No GPU is attached to this agent: ${revoked}`);
});
