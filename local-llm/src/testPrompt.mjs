// local_llm_test_prompt: an admin smoke chat that reaches the model the way
// other agents do (D10): AchillesAgentLib -> the workspace Soul Gateway ->
// Router -> this agent's endpoints.chatCompletions responder -> the runner.
// It never calls the runner directly, so request-time inference stays on the
// AchillesAgentLib path. Speed comes from the runner's own timings, which the
// chat responder reports to the controller.

import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { callController } from './controlSocket.mjs';
import { LocalLlmError } from './errors.mjs';

// Soul Gateway names a discovered agent's model <repo>/<agent>/<model id>;
// AgentServer's /v1/models answers with the single id `default`.
export const GATEWAY_MODEL = 'soul_gateway/local-llms/local-llm/default';
const DEFAULT_MAX_TOKENS = 256;

async function loadLLMAgent(env = process.env) {
    const root = env.PLOINKY_AGENTLIB_DIR || '/opt/ploinky-agentlib';
    const module = await import(pathToFileURL(path.join(root, 'LLMAgents', 'index.mjs')).href);
    if (typeof module.LLMAgent !== 'function') {
        throw new LocalLlmError('agentlib_unavailable', `AchillesAgentLib at ${root} has no LLMAgent.`);
    }
    return module.LLMAgent;
}

export async function runTestPrompt({
    prompt,
    maxTokens = DEFAULT_MAX_TOKENS,
    call = callController,
    loadAgent = loadLLMAgent,
    clock = () => performance.now(),
    now = () => Date.now(),
} = {}) {
    if (typeof prompt !== 'string' || !prompt.trim()) {
        throw new LocalLlmError('invalid_prompt', 'prompt must be a non-empty string.');
    }
    // Refuse early, with the controller's actionable message, when nothing runs.
    const target = await call('chatTarget', {}, { timeoutMs: 10_000 });
    const LLMAgent = await loadAgent();
    const agent = new LLMAgent({ name: 'local-llm-test-prompt' });
    const startedAt = now();
    const started = clock();
    let text;
    try {
        text = await agent.complete({ prompt, model: GATEWAY_MODEL, params: { max_tokens: maxTokens } });
    } catch (error) {
        throw new LocalLlmError('test_prompt_failed', `The test prompt failed on its way through Soul Gateway: ${error.message}`);
    }
    const elapsedMs = Math.round(clock() - started);
    const status = await call('status', { sinceSeq: Number.MAX_SAFE_INTEGER });
    const last = status?.lastCompletion;
    const stats = last && Date.parse(last.at) >= startedAt && last.modelId === target.modelId ? last : null;
    return {
        text: typeof text === 'string' ? text : String(text ?? ''),
        via: GATEWAY_MODEL,
        runnerId: target.runnerId,
        modelId: target.modelId,
        elapsedMs,
        promptTokens: stats?.promptTokens ?? null,
        completionTokens: stats?.completionTokens ?? null,
        promptTokensPerSecond: stats?.promptTokensPerSecond ?? null,
        generationTokensPerSecond: stats?.generationTokensPerSecond ?? null,
        statsSource: stats?.source ?? null,
    };
}
