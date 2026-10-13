// local_llm_test_prompt: an admin smoke chat against this agent's own runner.
//
// Exception to the workspace inference-routing rule, approved by the user in
// the implementation session's question dialog on 2026-09-23, with its bounds
// set in a follow-up message in that session, and recorded in
// docs/specs/DS001-agent-contract.md (Question #3) and the plan: Soul
// Gateway refuses an agent calling its own discovered model, so this admin
// check cannot take the AchillesAgentLib -> Soul Gateway path. It sends the
// same filtered request the chat responder would, to 127.0.0.1 inside this
// container, with the runner's per-start key. Other agents still reach the
// model only through AchillesAgentLib and Soul Gateway (GATEWAY_MODEL).

import { buildRunnerRequest, completionStats } from './chatResponder.mjs';
import { callController } from './controlSocket.mjs';
import { LocalLlmError } from './errors.mjs';
import { OBSERVATION_ID_RE } from './promptObservations.mjs';
import { readPromptStream } from './promptStream.mjs';

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
    observationId,
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
    if (observationId !== undefined) {
        if (typeof observationId !== 'string' || !OBSERVATION_ID_RE.test(observationId)) throw new LocalLlmError('invalid_observation', 'Invalid observationId.');
        return runObservedPrompt({ prompt, maxTokens, observationId, call, fetchImpl, clock });
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

async function runObservedPrompt({ prompt, maxTokens, observationId, call, fetchImpl, clock }) {
    const begun = await call('beginPromptObservation', { observationId }, { timeoutMs: 10000 });
    const { target, lease, observation: initial } = begun || {};
    const error = (code) => new LocalLlmError(code, 'The observed prompt did not complete with acknowledged evidence.');
    const positive = (value) => Number.isSafeInteger(value) && value > 0;
    if (!initial || initial.id !== observationId || initial.phase !== 'active' || !positive(initial.beginSequence)
        || !['controllerEpoch', 'deploymentId', 'modelId', 'runnerId'].every((key) => typeof initial[key] === 'string' && initial[key])
        || typeof initial.runnerIdentity?.runnerStartId !== 'string' || !initial.runnerIdentity.runnerStartId
        || !positive(initial.runnerIdentity.pid) || typeof lease !== 'string' || !lease
        || target?.modelId !== initial.modelId || target?.runnerId !== initial.runnerId) throw error('observation_incomplete');
    const bound = (record) => record && ['id', 'controllerEpoch', 'deploymentId', 'modelId', 'runnerId', 'beginSequence']
        .every((key) => record[key] === initial[key]) && record.runnerIdentity?.runnerStartId === initial.runnerIdentity.runnerStartId
        && record.runnerIdentity?.pid === initial.runnerIdentity.pid;
    const abort = new AbortController(); const started = clock(); const deadline = started + TIMEOUT_MS;
    let winner = null; let settledAt = null; let pollTimer = null; let pollInFlight = null; let inference = null; let resolveWinner;
    let firstTokenPending = false; let acknowledgedFirstTokenSequence = null;
    const selected = new Promise((resolve) => { resolveWinner = resolve; });
    const remaining = () => Math.max(0, deadline - clock());
    const select = (outcome) => {
        if (winner) return false;
        winner = outcome;
        settledAt = clock();
        clearTimeout(timer); clearTimeout(pollTimer); pollTimer = null;
        if (outcome.error) abort.abort(outcome.error);
        resolveWinner(outcome);
        return true;
    };
    const timer = setTimeout(() => select({ error: error('prompt_timeout') }), TIMEOUT_MS);
    function bounded(promise, timeoutMs, timeoutCode) {
        if (!(timeoutMs > 0)) return Promise.reject(error(timeoutCode));
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(error(timeoutCode)), timeoutMs);
            Promise.resolve(promise).then(resolve, reject).finally(() => clearTimeout(timeout));
        });
    }
    async function observedCall(operation, args = {}) {
        if (!remaining()) { const failure = error('prompt_timeout'); select({ error: failure }); throw failure; }
        let result;
        try {
            const timeoutMs = Math.min(1000, remaining());
            result = await bounded(Promise.resolve().then(() => call(operation, { observationId, lease, ...args }, { timeoutMs })), timeoutMs, 'observation_control_failed');
        } catch (cause) {
            const failure = error(!remaining() ? 'prompt_timeout' : cause?.code === 'controller_error' ? 'observation_incomplete' : 'observation_control_failed');
            select({ error: failure }); throw failure;
        }
        if (winner) return null;
        if (!remaining()) { const failure = error('prompt_timeout'); select({ error: failure }); throw failure; }
        const record = result?.observation;
        if (result?.recorded !== true || !bound(record) || record.phase !== 'active' || record.outcome !== null
            || record.errorCode !== null || record.cancellation !== null || record.terminalSequence !== null
            || typeof record.firstTokenPresent !== 'boolean'
            || (record.firstTokenPresent ? !positive(record.firstTokenSequence) || record.firstTokenSequence <= initial.beginSequence : record.firstTokenSequence !== null)) {
            const failure = error('observation_incomplete'); select({ error: failure }); throw failure;
        }
        return result.observation;
    }
    function poll() {
        pollInFlight = (async () => {
            try {
                const record = await observedCall('observePromptStop');
                if (!record || winner) return;
                if (record.stopSequence === null) {
                    if (record.stopNotificationSequence !== null) throw error('observation_incomplete');
                } else {
                    if (!positive(record.stopSequence) || !positive(record.stopNotificationSequence)
                        || record.stopSequence <= initial.beginSequence || record.stopNotificationSequence <= record.stopSequence) throw error('observation_incomplete');
                    if (firstTokenPending || (record.firstTokenPresent && record.firstTokenSequence !== acknowledgedFirstTokenSequence)) throw error('observation_incomplete');
                    select({ error: error('prompt_cancelled_by_stop'), cancellation: { mechanism: 'controller-stop-abort-v1',
                        stopSequence: record.stopSequence, notificationSequence: record.stopNotificationSequence } });
                }
            } catch (failure) { select({ error: failure }); }
            if (!winner) pollTimer = setTimeout(poll, 100);
        })();
        return pollInFlight;
    }
    async function infer() {
        try {
            if (new URL(target.baseUrl).hostname !== '127.0.0.1') throw error('runner_not_local');
            const request = buildRunnerRequest({ messages: [{ role: 'user', content: prompt }], max_tokens: maxTokens, stream: true }, target);
            let response;
            try {
                response = await fetchImpl(`${target.baseUrl}/v1/chat/completions`, { method: 'POST',
                    headers: { 'content-type': 'application/json', ...(target.apiKey ? { authorization: `Bearer ${target.apiKey}` } : {}) },
                    body: JSON.stringify(request), signal: abort.signal });
            } catch {
                const failure = error('prompt_transport_failed'); select({ error: failure }); throw failure;
            }
            if (winner) { Promise.resolve(response.body?.cancel()).catch(() => {}); return; }
            if (!response.ok) { Promise.resolve(response.body?.cancel()).catch(() => {}); throw error('runner_error'); }
            await readPromptStream(response, { signal: abort.signal, abort: () => { if (!abort.signal.aborted) abort.abort(); }, onSettlement: select,
                onFirstToken: async () => {
                    if (winner) throw winner.error || error('observation_incomplete');
                    firstTokenPending = true;
                    try {
                        const record = await observedCall('recordPromptObservation', { event: 'first-token' });
                        if (winner) return;
                        if (!record?.firstTokenPresent || !positive(record.firstTokenSequence) || record.firstTokenSequence <= initial.beginSequence) {
                            const failure = error('observation_incomplete'); select({ error: failure }); throw failure;
                        }
                        acknowledgedFirstTokenSequence = record.firstTokenSequence;
                    } finally {
                        firstTokenPending = false;
                    }
                } });
        } catch (failure) { select({ error: failure instanceof LocalLlmError ? failure : error('prompt_stream_invalid') }); }
    }
    await poll();
    if (!winner) inference = infer();
    const outcome = await selected;
    const elapsedMs = Math.max(0, Math.round(settledAt - started)); const finalDeadline = settledAt + 2000;
    let terminalAcknowledged = false;
    try {
        await bounded(Promise.allSettled([pollInFlight, inference]), finalDeadline - clock(), 'observation_incomplete');
        const timeoutMs = finalDeadline - clock();
        if (!(timeoutMs > 0)) throw error('observation_incomplete');
        const args = { observationId, lease, outcome: outcome.error ? 'failed' : 'success',
            ...(outcome.error ? { errorCode: outcome.error.code } : {}), ...(outcome.cancellation ? { cancellation: outcome.cancellation } : {}) };
        const finished = await bounded(Promise.resolve().then(() => call('finishPromptObservation', args, { timeoutMs })), timeoutMs, 'observation_incomplete');
        const record = finished?.observation;
        const sameCancellation = outcome.cancellation
            ? record?.cancellation && Object.keys(record.cancellation).length === 3
                && ['mechanism', 'stopSequence', 'notificationSequence'].every((key) => record.cancellation[key] === outcome.cancellation[key])
            : record?.cancellation === null;
        const expectedOutcome = !outcome.error ? 'success' : record?.stopSequence !== null ? 'interrupted-after-stop' : 'failed';
        if (finished?.recorded !== true || !bound(record) || record.phase !== 'terminal' || record.outcome !== expectedOutcome
            || record.errorCode !== (outcome.error?.code ?? null) || !positive(record.terminalSequence)
            || record.terminalSequence <= initial.beginSequence
            || !sameCancellation || (acknowledgedFirstTokenSequence !== null && record.firstTokenSequence !== acknowledgedFirstTokenSequence)
            || (record.stopSequence !== null && (!positive(record.stopSequence) || record.stopSequence <= initial.beginSequence || record.terminalSequence <= record.stopSequence))
            || (record.stopNotificationSequence !== null && (!positive(record.stopNotificationSequence)
                || !positive(record.stopSequence) || record.stopNotificationSequence <= record.stopSequence || record.terminalSequence <= record.stopNotificationSequence))
            || (outcome.cancellation && (record.stopSequence !== outcome.cancellation.stopSequence
                || record.stopNotificationSequence !== outcome.cancellation.notificationSequence || record.terminalSequence <= record.stopNotificationSequence))) throw error('observation_incomplete');
        terminalAcknowledged = true;
        if (outcome.error) throw outcome.error;
        const body = outcome.body; const stats = completionStats(body.statistics);
        return { text: body.text, truncated: body.truncated, reasoningChars: body.reasoningChars, finishReason: body.finishReason,
            runnerId: target.runnerId, modelId: target.modelId, via: 'loopback runner (admin test prompt)', gatewayModel: GATEWAY_MODEL,
            elapsedMs, promptTokens: stats?.promptTokens ?? null, completionTokens: stats?.completionTokens ?? null,
            promptTokensPerSecond: stats?.promptTokensPerSecond ?? null, generationTokensPerSecond: stats?.generationTokensPerSecond ?? null,
            statsSource: stats?.source ?? null, observation: record };
    } catch (failure) {
        if (terminalAcknowledged) throw failure;
        throw error('observation_incomplete');
    } finally { clearTimeout(timer); clearTimeout(pollTimer); }
}
