import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
    AGENT_SERVER_KILL_MS,
    AGENT_SERVER_SHUTDOWN_MS,
    DRAIN_DEADLINE_MS,
    DRAIN_QUEUE_WAIT_MS,
    DRAIN_RUNNER_GRACE_MS,
    HARDWARE_QUERY_REAP_MS,
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

test('a stopped hardware query is waited for within the command wait, so the worst case does not grow', () => {
    // The drain stops every hardware query before its command wait, and a
    // killed query is waited for at most HARDWARE_QUERY_REAP_MS: both run at once.
    assert.ok(HARDWARE_QUERY_REAP_MS > 0 && HARDWARE_QUERY_REAP_MS <= DRAIN_QUEUE_WAIT_MS);
    const deployments = fs.readFileSync(new URL('../src/controller/deployments.mjs', import.meta.url), 'utf8');
    const drain = deployments.slice(deployments.indexOf('async function drain()'));
    assert.ok(drain.indexOf('hardwareStop.abort()') > 0
        && drain.indexOf('hardwareStop.abort()') < drain.indexOf('queue.close()'), 'queries stop before the command wait');
    // The query timeout is unchanged and uses the shared reap bound, not a number of its own.
    const hardware = fs.readFileSync(new URL('../src/controller/hardware.mjs', import.meta.url), 'utf8');
    assert.match(hardware, /const QUERY_TIMEOUT_MS = 10_000;/);
    assert.match(hardware, /HARDWARE_QUERY_REAP_MS/);
});
