// local_llm_test_prompt: an admin smoke chat against this agent's own runner.
//
// Exception to the workspace inference-routing rule, approved on 2026-09-23
// through the user's delegated monitor and recorded in
// docs/specs/DS001-agent-contract.md (Question #3) and the plan: Soul
// Gateway refuses an agent calling its own discovered model, so this admin
// check cannot take the AchillesAgentLib -> Soul Gateway path. It sends the
// same filtered request the chat responder would, to 127.0.0.1 inside this
// container, with the runner's per-start key. Other agents still reach the
// model only through AchillesAgentLib and Soul Gateway (GATEWAY_MODEL).

import { buildRunnerRequest, completionStats } from './chatResponder.mjs';
import { callController } from './controlSocket.mjs';
import { LocalLlmError } from './errors.mjs';

// Soul Gateway names a discovered agent's model <repo>/<agent>/<model id>;
// AgentServer's /v1/models answers with the single id `default`.
export const GATEWAY_MODEL = 'soul_gateway/local-llms/local-llm/default';
// Bounds of the exception: the tool schema enforces the same input limits,
// and the runner call ends before the tool's 300 s limit.
export const PROMPT_MAX_CHARS = 4000;
export const MAX_TOKENS_LIMIT = 1024;
const DEFAULT_MAX_TOKENS = 256;
const TEXT_MAX_CHARS = 16_000;
const TIMEOUT_MS = 240_000;

function messageText(body) {
    const message = body?.choices?.[0]?.message || {};
    return {
        text: typeof message.content === 'string' ? message.content : '',
        reasoning: typeof message.reasoning_content === 'string' ? message.reasoning_content : '',
    };
}

export async function runTestPrompt({
    prompt,
    maxTokens = DEFAULT_MAX_TOKENS,
    call = callController,
    fetchImpl = globalThis.fetch,
    clock = () => performance.now(),
} = {}) {
    if (typeof prompt !== 'string' || !prompt.trim() || prompt.length > PROMPT_MAX_CHARS) {
        throw new LocalLlmError('invalid_prompt', `prompt must be 1 to ${PROMPT_MAX_CHARS} characters.`);
    }
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > MAX_TOKENS_LIMIT) {
        throw new LocalLlmError('invalid_prompt', `maxTokens must be an integer from 1 to ${MAX_TOKENS_LIMIT}.`);
    }
    // The target comes only from the controller: the ready runner on
    // 127.0.0.1 and its per-start key. Nothing in the input selects a URL.
    const target = await call('chatTarget', {}, { timeoutMs: 10_000 });
    if (new URL(target.baseUrl).hostname !== '127.0.0.1') {
        throw new LocalLlmError('runner_not_local', 'The test prompt only calls the runner on 127.0.0.1.');
    }
    const request = buildRunnerRequest({ messages: [{ role: 'user', content: prompt }], max_tokens: maxTokens }, target);
    const headers = { 'content-type': 'application/json' };
    if (target.apiKey) headers.authorization = `Bearer ${target.apiKey}`;
    const started = clock();
    let response;
    try {
        response = await fetchImpl(`${target.baseUrl}/v1/chat/completions`, {
            method: 'POST',
            headers,
            body: JSON.stringify(request),
            signal: AbortSignal.timeout(TIMEOUT_MS),
        });
    } catch (error) {
        throw new LocalLlmError('runner_unreachable', `The local runner did not answer: ${error.message}`);
    }
    if (!response.ok) {
        const detail = (await response.text().catch(() => '')).slice(0, 300);
        throw new LocalLlmError('runner_error', `The runner answered HTTP ${response.status}: ${detail}`);
    }
    const body = await response.json();
    const elapsedMs = Math.round(clock() - started);
    const runnerStats = completionStats(body);
    // Ollama's OpenAI endpoint reports token counts only; its speed is then an
    // end-to-end estimate from the wall clock, and labelled as such.
    const measured = runnerStats?.source === 'runner timings';
    const stats = runnerStats ? {
        ...runnerStats,
        generationTokensPerSecond: measured
            ? runnerStats.generationTokensPerSecond
            : (runnerStats.completionTokens && elapsedMs > 0 ? runnerStats.completionTokens / (elapsedMs / 1000) : null),
        source: measured ? 'runner timings' : 'token count over wall clock',
    } : null;
    if (stats) {
        try {
            await call('recordCompletion', stats, { timeoutMs: 2000 });
        } catch {}
    }
    const { text, reasoning } = messageText(body);
    return {
        text: text.slice(0, TEXT_MAX_CHARS),
        truncated: text.length > TEXT_MAX_CHARS,
        reasoningChars: reasoning.length,
        finishReason: body?.choices?.[0]?.finish_reason ?? null,
        runnerId: target.runnerId,
        modelId: target.modelId,
        via: 'loopback runner (admin test prompt)',
        gatewayModel: GATEWAY_MODEL,
        elapsedMs,
        promptTokens: stats?.promptTokens ?? null,
        completionTokens: stats?.completionTokens ?? null,
        promptTokensPerSecond: stats?.promptTokensPerSecond ?? null,
        generationTokensPerSecond: stats?.generationTokensPerSecond ?? null,
        statsSource: stats?.source ?? null,
    };
}
