// endpoints.chatCompletions responder (D10). AgentServer runs this command
// per request, after the Router verified the caller, with
// { endpoint, request, metadata } on stdin. The request is forwarded to the
// active runner with its per-start key; streaming responses are piped through
// unchanged. With no model ready the answer is a 503 failure envelope.

import { stdin, stdout, exit } from 'node:process';

import { callController } from './controlSocket.mjs';
import { serializeError } from './errors.mjs';

const FORWARDED_FIELDS = new Set([
    'messages', 'stream', 'stream_options', 'max_tokens', 'max_completion_tokens', 'temperature', 'top_p',
    'stop', 'seed', 'presence_penalty', 'frequency_penalty', 'n', 'tools', 'tool_choice', 'response_format',
    'reasoning_effort', 'chat_template_kwargs', 'logprobs', 'top_logprobs',
]);

export function buildRunnerRequest(request, target) {
    const body = {};
    for (const [key, value] of Object.entries(request || {})) {
        if (FORWARDED_FIELDS.has(key)) body[key] = value;
    }
    body.model = target.model;
    return body;
}

// A failure envelope chooses the HTTP status. For a streaming request the
// SSE headers are already sent, and AgentServer reads the envelope from the
// last stderr line instead of stdout.
function failure(status, code, message, { streaming = false } = {}) {
    const type = status === 503 ? 'service_unavailable' : status === 400 ? 'invalid_request_error' : 'server_error';
    const envelope = JSON.stringify({ ok: false, error: code, message, status, type });
    if (streaming) process.stderr.write(`${envelope}\n`);
    else stdout.write(envelope);
    exit(1);
}

export async function respond(payload, { call = callController, fetchImpl = globalThis.fetch, out = stdout } = {}) {
    const request = payload?.request;
    if (!request || !Array.isArray(request.messages)) {
        return { status: 400, code: 'invalid_request', message: 'messages must be an array' };
    }
    let target;
    try {
        target = await call('chatTarget', {}, { timeoutMs: 10_000 });
    } catch (error) {
        const { code, message } = serializeError(error);
        return { status: code === 'not_ready' ? 503 : 502, code, message };
    }
    const headers = { 'content-type': 'application/json' };
    if (target.apiKey) headers.authorization = `Bearer ${target.apiKey}`;
    const response = await fetchImpl(`${target.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(buildRunnerRequest(request, target)),
    });
    if (!response.ok) {
        const text = (await response.text().catch(() => '')).slice(0, 500);
        return { status: response.status >= 500 ? 502 : response.status, code: 'runner_error', message: `The runner answered HTTP ${response.status}: ${text}` };
    }
    if (request.stream === true) {
        for await (const chunk of response.body) out.write(chunk);
        return null;
    }
    out.write(JSON.stringify(await response.json()));
    return null;
}

async function main() {
    const chunks = [];
    for await (const chunk of stdin) chunks.push(chunk);
    let payload;
    try {
        payload = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
    } catch {
        failure(400, 'invalid_request', 'The request body is not valid JSON.');
        return;
    }
    const streaming = payload?.request?.stream === true;
    try {
        const problem = await respond(payload);
        if (problem) failure(problem.status, problem.code, problem.message, { streaming });
    } catch (error) {
        failure(502, 'runner_unreachable', `The local model runner is unreachable: ${error.message}`, { streaming });
    }
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
