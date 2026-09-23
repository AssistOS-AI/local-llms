import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
    AGENT_SERVER_KILL_MS,
    AGENT_SERVER_SHUTDOWN_MS,
    DRAIN_DEADLINE_MS,
    DRAIN_QUEUE_WAIT_MS,
    DRAIN_RUNNER_GRACE_MS,
} from '../src/drainBudget.mjs';

test('the worst-case drain leaves at least 5 s under the deadline and lets AgentServer finish', () => {
    const worstCase = DRAIN_QUEUE_WAIT_MS + DRAIN_RUNNER_GRACE_MS + AGENT_SERVER_KILL_MS;
    assert.ok(DRAIN_DEADLINE_MS - worstCase >= 5000, `worst case ${worstCase} ms against ${DRAIN_DEADLINE_MS} ms`);
    // Ploinky's targeted restart allows 35 s; AgentServer's task queue allows itself 20 s.
    assert.ok(DRAIN_DEADLINE_MS < 35_000);
    assert.equal(AGENT_SERVER_SHUTDOWN_MS, 20_000);
    assert.ok(AGENT_SERVER_KILL_MS > AGENT_SERVER_SHUTDOWN_MS);
    // The controller and main use the budget instead of their own numbers.
    const main = fs.readFileSync(new URL('../src/main.mjs', import.meta.url), 'utf8');
    const deployments = fs.readFileSync(new URL('../src/controller/deployments.mjs', import.meta.url), 'utf8');
    assert.doesNotMatch(main, /const (DRAIN_DEADLINE_MS|AGENT_SERVER_KILL_MS) = \d/);
    assert.doesNotMatch(deployments, /const DRAIN_QUEUE_WAIT_MS = \d/);
    assert.match(deployments, /DRAIN_RUNNER_GRACE_MS/);
});
