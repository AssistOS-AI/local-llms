import test from 'node:test';
import assert from 'node:assert/strict';
import { effectiveMemory, observeMemoryBudget } from '../src/controller/ploinkyBudget.mjs';
import { GIB, gpuHarness, waitForLaunch } from './ploinkyGpuFixture.mjs';

async function until(predicate) {
    const deadline = Date.now() + 3000;
    while (!predicate() && Date.now() < deadline) await new Promise((resolve) => setTimeout(resolve, 2));
    assert.ok(predicate());
}

test('finite caps never fabricate physical capacity or availability from an unknown raw reading', () => {
    const budget = observeMemoryBudget({ maxText: String(16 * GIB), currentText: String(2 * GIB) });
    for (const field of ['totalBytes', 'availableBytes']) {
        for (const value of [undefined, null, NaN, -1, 'malformed']) {
            const raw = { totalBytes: 128 * GIB, availableBytes: 120 * GIB, [field]: value };
            assert.equal(effectiveMemory(raw, budget), raw);
            assert.equal(Object.hasOwn(raw, 'budgetBytes'), false);
        }
    }
});

for (const field of ['totalBytes', 'availableBytes']) {
    test(`finite-budget dedicated offload guard stops after two unknown ${field} samples`, async (t) => {
        let calls = 0;
        const h = gpuHarness(t, {
            env: {}, cgroupMemory: { maxBytes: 16 * GIB, currentBytes: 2 * GIB }, guardSampleMs: 10,
            readMemory: () => { calls += 1; return { totalBytes: 128 * GIB, availableBytes: 120 * GIB, [field]: null }; },
        });
        assert.equal((await h.run({ maxModelLen: 512, cpuOffloadGb: 0.5 })).accepted, true);
        const launch = await waitForLaunch(h);
        await until(() => launch.stopped === 'SIGKILL');
        assert.equal(calls, 2);
        await until(() => h.controller.state.deployment?.phase === 'error');
        assert.match(h.controller.state.deployment.error, /budget_unreadable/);
    });
}

test('finite-budget CPU guard retains immediate unreadable MemAvailable handling', async (t) => {
    let calls = 0;
    const h = gpuHarness(t, {
        env: {}, profile: 'cpu', gpu: { available: false, state: 'absent', reason: 'No GPU.' }, qualificationDataProvider: null,
        cgroupMemory: { maxBytes: 16 * GIB, currentBytes: 2 * GIB }, guardSampleMs: 10,
        readMemory: () => { calls += 1; return { totalBytes: 128 * GIB, availableBytes: null }; },
    });
    assert.equal((await h.run({}, { runnerId: 'llama.cpp', modelId: 'qwen2.5-0.5b-instruct-q4_k_m' })).accepted, true);
    const launch = await waitForLaunch(h);
    await until(() => launch.stopped === 'SIGKILL');
    assert.equal(calls, 1);
    await until(() => h.controller.state.deployment?.phase === 'error');
    assert.match(h.controller.state.deployment.error, /host memory cannot be read/);
});
