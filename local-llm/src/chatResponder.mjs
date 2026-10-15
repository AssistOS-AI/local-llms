// endpoints.chatCompletions responder (D10). AgentServer runs this command
// per request, after the Router verified the caller, with
// { endpoint, request, metadata } on stdin. The request is forwarded to the
// active runner with its per-start key; streaming responses are piped through
// unchanged. With no model ready the answer is a 503 failure envelope.

import { stdin, stdout, exit } from 'node:process';

import { serializeError } from 'pworker/local-llm';
import {
    buildRunnerRequest, chatLimits, completionStats, cpuCompletionBudget, createStreamStatsReader, requestLimitProblem,
} from 'pworker/local-llm/chat';

import { callController } from './controlSocket.mjs';

async function reportStats(call, stats) {
    if (!stats) return;
    try {
        await call('recordCompletion', stats, { timeoutMs: 2000 });
    } catch {
        // Stats are informational; the completion was already delivered.
    }
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

// The completion budget and the runner deadline come from the agent's environment through the engine's chat helpers
// (LOCAL_LLM_MAX_COMPLETION_TOKENS, LOCAL_LLM_RUNNER_TIMEOUT_MS; DS001); the caller's own deadline (Soul Gateway: 120 s
// by default) still applies, so a long budget needs that deadline raised too.
export async function respond(payload, {
    call = callController,
    fetchImpl = globalThis.fetch,
    out = stdout,
    limits = chatLimits(),
    cpuMaxCompletionTokens = cpuCompletionBudget(),
    timeoutMs = limits.runnerTimeoutMs,
} = {}) {
    const request = payload?.request;
    if (!request || !Array.isArray(request.messages)) {
        return { status: 400, code: 'invalid_request', message: 'messages must be an array' };
    }
    const limitProblem = requestLimitProblem(request);
    if (limitProblem) return { status: 400, code: 'invalid_request', message: limitProblem };
    let target;
    try {
        target = await call('chatTarget', {}, { timeoutMs: 10_000 });
    } catch (error) {
        const { code, message } = serializeError(error);
        return { status: code === 'not_ready' ? 503 : 502, code, message };
    }
    const headers = { 'content-type': 'application/json' };
    if (target.apiKey) headers.authorization = `Bearer ${target.apiKey}`;
    const signal = AbortSignal.timeout(timeoutMs);
    const timedOut = () => ({
        status: 504, code: 'runner_timeout', message: `The local runner did not finish within ${Math.round(timeoutMs / 1000)} s.`,
    });
    try {
        const response = await fetchImpl(`${target.baseUrl}/v1/chat/completions`, {
            method: 'POST',
            headers,
            body: JSON.stringify(buildRunnerRequest(request, target, {
                maxCompletionTokens: target.profile === 'cpu' ? cpuMaxCompletionTokens : limits.maxCompletionTokens,
            })),
            signal,
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
    } catch (error) {
        if (signal.aborted) return timedOut();
        throw error;
    }
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
