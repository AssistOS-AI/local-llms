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

// Runner-reported speed for the Settings test prompt: llama.cpp adds
// `timings` to a response and to the last stream chunk; Ollama's /v1 gives
// token counts in `usage` only.
export function completionStats(body) {
    if (!body || typeof body !== 'object') return null;
    const timings = body.timings && typeof body.timings === 'object' ? body.timings : null;
    const usage = body.usage && typeof body.usage === 'object' ? body.usage : null;
    if (!timings && !usage) return null;
    const count = (value) => (Number.isFinite(value) && value >= 0 ? value : null);
    return {
        promptTokens: count(timings?.prompt_n) ?? count(usage?.prompt_tokens),
        completionTokens: count(timings?.predicted_n) ?? count(usage?.completion_tokens),
        promptTokensPerSecond: count(timings?.prompt_per_second),
        generationTokensPerSecond: count(timings?.predicted_per_second),
        source: timings ? 'runner timings' : 'usage',
    };
}

// Watches SSE `data:` lines as they pass through and keeps the last stats.
function createStreamStatsReader() {
    const decoder = new TextDecoder();
    let pending = '';
    let stats = null;
    const scan = (line) => {
        if (!line.startsWith('data:') || !/"(timings|usage)"/.test(line)) return;
        try {
            stats = completionStats(JSON.parse(line.slice(5).trim())) || stats;
        } catch {}
    };
    return {
        push(chunk) {
            pending += typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true });
            let newline;
            while ((newline = pending.indexOf('\n')) >= 0) {
                scan(pending.slice(0, newline).trim());
                pending = pending.slice(newline + 1);
            }
        },
        finish() {
            scan(pending.trim());
            return stats;
        },
    };
}

async function reportStats(call, stats) {
    if (!stats) return;
    try {
        await call('recordCompletion', stats, { timeoutMs: 2000 });
    } catch {
        // Stats are informational; the completion was already delivered.
    }
}

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
        const reader = createStreamStatsReader();
        for await (const chunk of response.body) {
            out.write(chunk);
            reader.push(chunk);
        }
        await reportStats(call, reader.finish());
        return null;
    }
    const body = await response.json();
    out.write(JSON.stringify(body));
    await reportStats(call, completionStats(body));
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
