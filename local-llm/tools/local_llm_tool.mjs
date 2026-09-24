// One entry point for every local_llm_* MCP tool, dispatched on TOOL_NAME
// (AgentServer does not set it; mcp-config.json does, per tool).
//
// Every tool is admin-only twice: the Router's MCP tag policy (tags
// ["admin"]) and this check on the Router-signed invocation. Like the Router,
// the check requires the admin role and rejects guests; delegated user roles
// win over the actor's, as authInfoFromInvocation defines.

import { stdin, stdout, env, exit } from 'node:process';

import { callController } from '../src/controlSocket.mjs';
import { LocalLlmError, serializeError } from '../src/errors.mjs';

export const TOOL_OPERATIONS = Object.freeze({
    local_llm_overview: { op: 'overview', args: (input) => (input.preview ? { preview: input.preview } : {}) },
    local_llm_status: { op: 'status', args: (input) => ({ sinceSeq: input.sinceSeq ?? 0 }) },
    local_llm_run: {
        op: 'run',
        args: (input) => ({
            requestId: input.requestId,
            modelId: input.modelId,
            runnerId: input.runnerId,
            params: input.params,
            replace: input.replace === true,
        }),
    },
    local_llm_stop: { op: 'stop', args: () => ({}) },
    local_llm_download_cancel: { op: 'cancelDownload', args: () => ({}) },
    local_llm_weights_delete: {
        op: 'deleteWeights',
        // Weights are named by format or by a runner that reads that format.
        args: (input) => ({
            modelId: input.modelId,
            ...(input.runnerId !== undefined ? { runnerId: input.runnerId } : {}),
            ...(input.format !== undefined ? { format: input.format } : {}),
        }),
    },
    local_llm_model_add: { op: 'addModel', args: (input) => ({ model: input.model }) },
    local_llm_model_update: { op: 'updateModel', args: (input) => ({ model: input.model }) },
    local_llm_model_remove: { op: 'removeModel', args: (input) => ({ modelId: input.modelId }) },
});

export const TOOL_NAMES = Object.freeze([...Object.keys(TOOL_OPERATIONS), 'local_llm_test_prompt']);

async function loadInvocationAuth() {
    const candidates = [env.PLOINKY_INVOCATION_AUTH_MODULE, '/Agent/lib/invocation-auth.mjs'].filter(Boolean);
    for (const candidate of candidates) {
        try {
            return await import(candidate);
        } catch {}
    }
    throw new LocalLlmError('auth_unavailable', 'The verified invocation helper is unavailable.');
}

/** Resolve the caller from the Router-signed invocation in the envelope. */
export async function authInfoFromEnvelope(envelope, { loadAuth = loadInvocationAuth } = {}) {
    const grant = envelope?.metadata?.invocation;
    if (!grant || typeof grant !== 'object') return null;
    const { authInfoFromInvocation } = await loadAuth();
    return authInfoFromInvocation(grant, { invocationToken: envelope?.metadata?.invocationToken || '' });
}

export function assertAdmin(authInfo) {
    const roles = Array.isArray(authInfo?.user?.roles) ? authInfo.user.roles.map(String) : [];
    if (!roles.includes('admin') || roles.includes('guest')) {
        throw new LocalLlmError('admin_required', 'Administrator access is required.');
    }
}

export async function handleTool(name, input, { authInfo, call = callController, testPrompt } = {}) {
    if (!TOOL_NAMES.includes(name)) throw new LocalLlmError('unknown_tool', `Unknown tool ${name}.`);
    assertAdmin(authInfo);
    if (name === 'local_llm_test_prompt') {
        const run = testPrompt || (await import('../src/testPrompt.mjs')).runTestPrompt;
        return run({ prompt: input.prompt, maxTokens: input.maxTokens, call });
    }
    const operation = TOOL_OPERATIONS[name];
    return call(operation.op, operation.args(input || {}));
}

async function main() {
    const chunks = [];
    for await (const chunk of stdin) chunks.push(chunk);
    let envelope = {};
    try {
        envelope = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
    } catch {
        envelope = {};
    }
    const name = env.TOOL_NAME || envelope.tool || envelope.name;
    const input = envelope.input ?? envelope.arguments ?? {};
    try {
        const authInfo = await authInfoFromEnvelope(envelope);
        const result = await handleTool(name, input, { authInfo });
        stdout.write(JSON.stringify(result ?? {}));
    } catch (error) {
        const { code, message, details } = serializeError(error);
        stdout.write(JSON.stringify({ ok: false, error: code, message, ...(details ? { details } : {}) }));
        exit(1);
    }
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
