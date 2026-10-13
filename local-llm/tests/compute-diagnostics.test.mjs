import assert from 'node:assert/strict';
import test from 'node:test';
import { cpuFallbackExplanation, computeDiagnostic } from '../src/controller/computeDiagnostics.mjs';
import { hardwareCardsHtml, measurementHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';
import { readGpu } from '../src/controller/hardware.mjs';

test('CPU explanations distinguish unknown GPU, observed Vulkan and observed Venus without performance claims', () => {
    for (const [diagnostic, name] of [[null, 'GPU'], [computeDiagnostic('vulkan_query_unreadable', { backend: 'vulkan', message: 'query failed' }), 'Vulkan GPU'],
        [computeDiagnostic('vulkan_memory_invalid', { backend: 'vulkan', driver: 'venus', message: 'memory invalid' }), 'Apple Venus']]) {
        const decision = { cause: 'unusable', diagnostic };
        const explanation = cpuFallbackExplanation(decision, { gpuRecovered: true });
        assert.ok(explanation.summary.includes(name)); assert.doesNotMatch(explanation.summary, /NVIDIA|slower/);
        assert.match(explanation.restartHint, /restart local-llm/);
        const html = hardwareCardsHtml({}, null, { profile: 'cpu', decision: { ...decision, explanation } });
        assert.ok(html.includes(explanation.summary));
    }
});

test('Ploinky explanations are bounded, sanitized and attributed without guessing Apple identity', async () => {
    const gpu = await readGpu({ env: { PLOINKY_GPU_STATUS: 'unavailable', PLOINKY_GPU_REASON: 'wrong provider\n\u202efor machine' },
        execFileImpl(command, args, options, callback) { callback(Object.assign(new Error('missing'), { code: 'ENOENT' }), '', ''); } });
    assert.equal(gpu.diagnostic.code, 'gpu_access_unavailable'); assert.equal(gpu.diagnostic.driver, null);
    assert.match(gpu.diagnostic.message, /^Ploinky reports:/); assert.doesNotMatch(gpu.diagnostic.message, /[\n\u202e]/);
});

test('historical records render escaped text with no URL activation or object coercion', () => {
    const html = measurementHtml({ state: 'historical', applicability: 'unknown', records: [{ id: 'test', measuredAt: 'today', evidence: { reference: '<script>https://example.test</script>' }, reasons: [] }] });
    assert.match(html, /&lt;script&gt;/); assert.doesNotMatch(html, /<script>|<a|\[object Object\]/);
});

test('an unavailable Ploinky status without explanatory text retains attribution and an unknown cause', async () => {
    for (const reason of [undefined, '', '\n']) {
        const gpu = await readGpu({ env: { PLOINKY_GPU_STATUS: 'unavailable', PLOINKY_GPU_REASON: reason },
            execFileImpl(command, args, options, callback) { callback(Object.assign(new Error('missing'), { code: 'ENOENT' }), '', ''); } });
        assert.equal(gpu.diagnostic.code, 'gpu_access_unavailable'); assert.equal(gpu.diagnostic.source, 'ploinky-environment');
        assert.equal(gpu.diagnostic.backend, null); assert.equal(gpu.diagnostic.driver, null);
        assert.match(gpu.diagnostic.message, /^Ploinky reports: GPU access is unavailable; the cause is unknown/);
        assert.doesNotMatch(gpu.diagnostic.message, /Apple|denied|wrong provider/);
    }
});
