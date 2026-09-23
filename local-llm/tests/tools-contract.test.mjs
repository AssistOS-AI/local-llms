import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { TOOL_NAMES, TOOL_OPERATIONS, assertAdmin, authInfoFromEnvelope, handleTool } from '../tools/local_llm_tool.mjs';
import { buildRunnerRequest, respond } from '../src/chatResponder.mjs';

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

test('the manifest requests only the granted CDI device and closes the agent-port relay', () => {
    const manifest = read('manifest.json');
    assert.deepEqual(manifest.llmRuntime, { runtimePolicy: { devices: [{ type: 'cdi', value: 'ploinky.local/gpu=all' }] } });
    assert.equal(manifest.llmRuntime.enabled, undefined);
    assert.equal(manifest.routerAccess.agentPorts, false);
    assert.equal(manifest.containerSecurity, undefined);
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
    assert.deepEqual(buildRunnerRequest({ messages: [], stream: true }, { model: 'gpt-oss:20b' }), { messages: [], stream: true, model: 'gpt-oss:20b' });
});
