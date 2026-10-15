import assert from 'node:assert/strict';
import test from 'node:test';

import { computeDiagnostic } from 'pworker/local-llm/internal/controller/computeDiagnostics.mjs';

import { cpuFallbackExplanation } from '../src/engineOptions.mjs';
import { hardwareCardsHtml, measurementHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';

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

test('historical records render escaped text with no URL activation or object coercion', () => {
    const html = measurementHtml({ state: 'historical', applicability: 'unknown', records: [{ id: 'test', measuredAt: 'today', evidence: { reference: '<script>https://example.test</script>' }, reasons: [] }] });
    assert.match(html, /&lt;script&gt;/); assert.doesNotMatch(html, /<script>|<a|\[object Object\]/);
});

