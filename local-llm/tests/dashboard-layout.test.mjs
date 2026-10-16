import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { LocalLlmDashboard } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.js';
import { hardwareSummaryHtml, hardwareWarningsHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-resources-view.js';
import { helpHtml, initAddModelDialog } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-help.js';
import { estimateHtml, statusCardHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';

test('the Runner selector label is bold and uses primary text without changing other labels', async () => {
    const base = new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/', import.meta.url);
    const html = await readFile(new URL('local-llm-dashboard.html', base), 'utf8');
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    assert.match(html, /<span class="form-label local-llm-runner-label">Runner<\/span>/);
    assert.equal((html.match(/local-llm-runner-label/g) || []).length, 1);
    const titleSize = css.match(/\.settings-section-title \{[^}]*font-size: ([^;]+);/)?.[1];
    const runnerSize = css.match(/\.form-label\.local-llm-runner-label \{[^}]*font-size: ([^;]+);/)?.[1];
    assert.ok(titleSize);
    assert.equal(runnerSize, titleSize, 'Runner label matches the model title size');
    assert.match(css, /\.local-llm-field \.form-label\.local-llm-runner-label \{[^}]*font-weight: var\(--font-bold\);[^}]*color: var\(--text\)/);
});

test('approved design keeps underlined tabs, quiet actions, flat detail and compact rows', async () => {
    const base = new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/', import.meta.url);
    const html = await readFile(new URL('local-llm-dashboard.html', base), 'utf8');
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    assert.match(css, /\.settings-tabs \.settings-tab \{[^}]*border-bottom: 2px solid transparent/);
    assert.match(css, /\.settings-tabs \.settings-tab\.active \{[^}]*border-bottom-color: var\(--accent\)/);
    assert.match(css, /\.local-llm-dashboard \{[^}]*align-content: start;[^}]*background: var\(--surface\)/);
    assert.match(html, /class="local-llm-quiet local-llm-refresh"/);
    assert.match(html, /class="local-llm-quiet"[\s\S]*data-llm-add/);
    assert.match(html, /<aside class="local-llm-detail local-llm-surface"/);
    assert.doesNotMatch(html, /settings-card[^\"]*local-llm-detail/);
    assert.match(html, /local-llm-run-footer[\s\S]*Weights download only when you run/);
    assert.match(css, /@media \(max-width: 620px\)[\s\S]*\.local-llm-models-layout \{ grid-template-columns: minmax\(0, 1fr\)/);
});

test('Playground, Logs and Runners have scoped surfaces', async () => {
    const base = new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/', import.meta.url);
    const html = await readFile(new URL('local-llm-dashboard.html', base), 'utf8');
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    assert.match(html, /local-llm-prompt-form local-llm-surface/);
    assert.match(html, /<\/form>\s*<section class="local-llm-surface local-llm-response"[\s\S]*data-llm-prompt-result aria-live="polite"/);
    assert.match(html, /local-llm-surface local-llm-logs-panel/);
    assert.match(css, /\.settings-card\.local-llm-surface \{[^}]*border: 1px solid var\(--border\);[^}]*background: var\(--surface-soft\)/);
    assert.match(css, /\.local-llm-logs-panel > \.local-llm-log \{[^}]*background: var\(--surface-soft\)/);
    assert.match(css, /\.local-llm-runners-panel \.local-llm-runner-item \{[^}]*border: 1px solid var\(--border\)/);
    assert.match(css, /@media \(max-width: 900px\) \{\s*local-llm-dashboard \.local-llm-playground-layout \{\s*grid-template-columns: minmax\(0, 1fr\)/);
});

test('Logs fills the available height and Models separates its list, selection and run form', async () => {
    const base = new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/', import.meta.url);
    const html = await readFile(new URL('local-llm-dashboard.html', base), 'utf8');
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    assert.match(html, /class="local-llm-models-list local-llm-surface"/);
    assert.match(html, /class="local-llm-detail local-llm-surface"/);
    assert.match(css, /\.local-llm-model-list > li \+ li \{[^}]*border-top: 1px solid var\(--border\)/);
    assert.match(css, /\.local-llm-model-list > li:last-child \{[^}]*padding-bottom: 4px;[^}]*border-bottom: 1px solid var\(--border\)/);
    assert.match(css, /\.local-llm-detail > \.local-llm-run-form \{[^}]*border-top: 1px solid var\(--border\)/);
    assert.match(css, /\.local-llm-dashboard:has\(> \.local-llm-logs-panel:not\(\[hidden\]\)\) \{[^}]*height: 100%;[^}]*min-height: 0/);
    assert.match(css, /\.local-llm-logs-panel \{[^}]*flex: 1 1 0;[^}]*grid-template-rows: auto minmax\(0, 1fr\)/);
    assert.match(css, /\.local-llm-log \{[^}]*height: 100%;[^}]*overflow: auto/);
    assert.doesNotMatch(css, /height: max\(18rem/);
    assert.match(css, /\.local-llm-models-layout \{\s*grid-template-columns: clamp\(17rem, 30%, 25rem\) minmax\(0, 1fr\)/);
    assert.match(css, /\.local-llm-models-layout \{[^}]*align-items: stretch/);
    assert.match(css, /\.local-llm-models-list \{[^}]*align-content: start/);
    assert.match(css, /\.local-llm-models-layout \{ grid-template-columns: minmax\(0, 1fr\); align-items: start/);
    assert.match(html, /Choose a model, configure its runner and check memory requirements/);
    assert.doesNotMatch(html, /Select a model to see its runners, settings and estimate/);
});

test('default selection prefers the in-use model, preserves user choice and handles catalog changes', () => {
    const p = new LocalLlmDashboard({}, () => {});
    const first = { id: 'first' };
    const active = { id: 'active' };
    p.overview = { models: [first, active], deployment: { modelId: 'active', phase: 'ready' } };
    p.busy = true;
    assert.equal(p.ensureModelSelection(), active);
    assert.equal(p.selectedModelId, 'active');
    assert.equal(p.detailMode, 'model');
    p.selectedModelId = 'first';
    assert.equal(p.ensureModelSelection(), null);
    assert.equal(p.selectedModelId, 'first');
    p.overview.models = [active];
    assert.equal(p.ensureModelSelection(), active);
    p.selectedModelId = '';
    p.overview = { models: [first, active] };
    assert.equal(p.ensureModelSelection(), first);
    p.overview.models = [];
    assert.equal(p.ensureModelSelection(), null);
    assert.equal(p.selectedModelId, '');
    assert.equal(p.detailMode, 'empty');
});

test('dashboard uses the modal title without a duplicate intro and preserves live feedback', async () => {
    const base = new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/', import.meta.url);
    const html = await readFile(new URL('local-llm-dashboard.html', base), 'utf8');
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    assert.doesNotMatch(html, /local-llm-intro|local-llm-page-title|Manage models running in your workspace/);
    assert.match(html, /id="localLlmStatus" role="status" aria-live="polite"/);
    assert.match(css, /#localLlmStatus:empty \{\s*display: none;/);
    assert.doesNotMatch(css, /local-llm-intro|local-llm-page-title/);
    assert.doesNotMatch(html, /<h3[^>]*>(Models|Playground|Runner log|Runners)<\/h3>/);
    assert.match(html, /id="localLlmResponseTitle" class="settings-section-title">Response<\/h3>/);
    for (const tab of ['Models', 'Playground', 'Logs', 'Runners']) {
        assert.match(html, new RegExp(`aria-labelledby="localLlmTab${tab}"`));
    }
});

test('runtime and hardware share two aligned columns that stack on narrow screens', async () => {
    const base = new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/', import.meta.url);
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    const html = await readFile(new URL('local-llm-dashboard.html', base), 'utf8');
    assert.ok(html.indexOf('class="local-llm-resources"') < html.indexOf('data-llm-deployment'));
    assert.match(css, /\.local-llm-summary \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);[^}]*background: var\(--surface-soft\);[^}]*border: 1px solid var\(--border\)/);
    assert.match(css, /\.local-llm-stat-status \{[^}]*padding: 0;[^}]*background: transparent;[^}]*border: 0/);
    assert.match(css, /\.local-llm-stat-status > \.local-llm-stat-title \{[^}]*grid-row: 1/);
    assert.match(css, /\.local-llm-stat-status > \.local-llm-stat-actions \{[^}]*grid-column: 2; grid-row: 1; align-self: start/);
    assert.match(css, /@media \(max-width: 900px\)[\s\S]*\.local-llm-summary \{\s*grid-template-columns: minmax\(0, 1fr\)/);
    assert.match(css, /\.local-llm-resources \{[^}]*display: grid;[^}]*gap: 4px/);
    const running = statusCardHtml({ phase: 'ready', deployment: { modelId: 'test', runnerId: 'llama.cpp' }, runnerReport: { device: 'CPU' } });
    assert.match(running, /local-llm-field-heading">Running model<button[^>]*aria-label="Runtime details"/);
});

test('admitted-run explanations use help, while admission refusals remain inline', () => {
    const allowed = estimateHtml({ admission: { status: 'ok', warnings: ['Sizing is estimated'], estimate: { basis: 'pinned weights' } } });
    assert.match(allowed, /1 notice/);
    assert.match(allowed, /aria-label="Runtime notices"/);
    assert.match(allowed, /id="localLlmNoticesHelp" popover[^>]*>Sizing is estimated/);
    const refused = estimateHtml({ admission: { status: 'incompatible', warnings: ['Unsupported format'] } });
    assert.match(refused, /Unsupported format<\/div>/);
    assert.doesNotMatch(refused, /localLlmNoticesHelp/);
});

test('resource summary respects container headroom and uses no invented values', () => {
    const hardware = { cores: 4, memory: { availableBytes: 8 * 2 ** 30 }, disk: { freeBytes: 2e9 } };
    const summary = hardwareSummaryHtml(hardware, null, { profile: 'cpu', limits: { availableBytes: 2 ** 30 } });
    assert.match(summary, /CPU · 4 cores/);
    assert.match(summary, /1,024 MiB/);
    assert.doesNotMatch(summary, /8,192/);
    assert.match(summary, /2\.0 GB/);
    assert.doesNotMatch(hardwareSummaryHtml(), /CPU|0 MiB|0 B/);
    assert.match(hardwareSummaryHtml(), /GPU/);
    assert.match(hardwareSummaryHtml({ gpu: { available: true, name: '<GPU>' } }), /&lt;GPU&gt;/);
    assert.match(hardwareSummaryHtml({ gpu: { available: true, name: 'old' } }, { available: true, name: 'fresh' }), /fresh/);
});

test('resource help holds reserves and measurements without an expandable details panel', async () => {
    const html = await readFile(new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.html', import.meta.url), 'utf8');
    assert.doesNotMatch(html, /<details class="local-llm-resources"|data-llm-resource-expand|data-llm-hardware>/);
    assert.match(html, /class="local-llm-resource-peek" data-llm-hardware-summary/);
    const summary = hardwareSummaryHtml({ memory: { totalBytes: 8 * 2 ** 30 }, disk: { freeBytes: 1e9 } }, null, {
        profile: 'cpu', limits: { availableBytes: 2 ** 30, floorBytes: 100 * 2 ** 20, budget: { cpus: 2 } },
        decision: { explanation: { summary: '<compute reason>' } },
    });
    assert.match(summary, /aria-label="About Compute"/);
    assert.match(summary, /id="localLlmResource-memory" popover[\s\S]*keeps 100 MiB free for the workspace[\s\S]*<\/span>/);
    assert.match(summary, /aria-label="Memory in use"/);
    assert.match(summary, /aria-label="About Limits from Ploinky"/);
    assert.match(summary, /&lt;compute reason&gt;/);
    assert.doesNotMatch(summary, /<compute reason>/);
    const apu = hardwareSummaryHtml({}, null, { profile: 'apu', limits: { warnings: ['one', 'two'] } });
    const ids = [...apu.matchAll(/\sid="([^"]+)"/g)].map(match => match[1]);
    assert.equal(new Set(ids).size, ids.length);
});

test('compact memory uses correctly converted available / total decimal units and short labels', () => {
    const memory = { availableBytes: 3833 * 2 ** 20, totalBytes: 7906 * 2 ** 20 };
    const html = hardwareSummaryHtml({ cores: 5, memory }, null, { profile: 'cpu' });
    assert.match(html, /data-llm-resource="memory"[^]*?class="settings-card-meta">Memory<\/span><strong>4\.0 \/ 8\.3 GB<\/strong>/);
    assert.match(html, /class="settings-card-meta">CPU<\/span><strong>5 cores<\/strong>/);
    assert.match(hardwareSummaryHtml({ cores: 1 }, null, { profile: 'cpu' }), />CPU<\/span><strong>1 core<\/strong>/);
    assert.match(hardwareSummaryHtml({}, null, { profile: 'cpu' }), />CPU<\/span><strong>—<\/strong>/);
    assert.doesNotMatch(html, />Compute<\/span>|>Memory available<\/span>/);
    assert.match(html, /3,833 MiB of 7,906 MiB/);
    assert.match(html, /Summary: available \/ total memory/);
    const limited = hardwareSummaryHtml({ memory }, null, { profile: 'cpu', limits: { availableBytes: 2e9, poolBytes: 4e9 } });
    assert.match(limited, /<strong>2\.0 \/ 4\.0 GB<\/strong>/);
    const smaller = hardwareSummaryHtml({ memory: { availableBytes: 512e6, totalBytes: 900e6 } }, null, { profile: 'cpu' });
    assert.match(smaller, /<strong>512 \/ 900 MB<\/strong>/);
    const unknown = hardwareSummaryHtml({ memory: { totalBytes: 8e9 } }, null, { profile: 'cpu' });
    assert.match(unknown, /<strong>— \/ 8\.0 GB<\/strong>/);
    const zero = hardwareSummaryHtml({ memory: { availableBytes: 0, totalBytes: 8e9 } }, null, { profile: 'cpu' });
    assert.match(zero, /<strong>0\.0 \/ 8\.0 GB<\/strong>/);
    const shared = hardwareSummaryHtml({ memory, gpu: { available: true, memoryModel: 'unified' } });
    assert.match(shared, /data-llm-resource="unified"[^]*?<strong>4\.0 \/ 8\.3 GB<\/strong>/);
    assert.doesNotMatch(shared, /data-llm-resource="ram"/);
    const dedicated = hardwareSummaryHtml({ gpu: { available: true, totalBytes: 8e9, usedBytes: 3e9 } });
    assert.match(dedicated, />VRAM<\/span><strong>5\.0 \/ 8\.0 GB<\/strong>/);
});

test('compact disk shows free / total in one decimal unit across hardware profiles', () => {
    const disk = { freeBytes: 8.3e12, totalBytes: 127e12 };
    for (const profile of ['cpu', 'apu', 'vulkan', 'venus', 'dedicated', null]) {
        const html = hardwareSummaryHtml({ disk }, null, { profile });
        assert.match(html, />Disk<\/span><strong>8\.3 \/ 127\.0 TB<\/strong>/);
        assert.match(html, /Summary: free \/ total storage/);
    }
    assert.match(hardwareSummaryHtml({ disk: { freeBytes: 500e9, totalBytes: 2e12 } }), /<strong>0\.5 \/ 2\.0 TB<\/strong>/);
    assert.match(hardwareSummaryHtml({ disk: { freeBytes: 8e12 } }), /<strong>8\.0 \/ — TB<\/strong>/);
    assert.match(hardwareSummaryHtml({ disk: { totalBytes: 8e12 } }), /<strong>— \/ 8\.0 TB<\/strong>/);
    assert.match(hardwareSummaryHtml({ disk: { freeBytes: 0, totalBytes: 8e12 } }), /<strong>0\.0 \/ 8\.0 TB<\/strong>/);
});

test('safety and experimental warnings remain outside resource help', async () => {
    assert.match(hardwareWarningsHtml({ profile: 'apu', limits: { warnings: ['<guard>'] } }), /experimental and unqualified[\s\S]*&lt;guard&gt;/);
    assert.match(hardwareWarningsHtml({ profile: 'venus', limits: {} }), /headroom is unknown/);
    assert.match(hardwareWarningsHtml({ profile: 'vulkan' }), /unverified on hardware/);
    assert.match(hardwareWarningsHtml({ profile: 'cpu', profileDecision: { cause: 'revoked', reason: 'Access revoked' } }), /Access revoked/);
    const html = await readFile(new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.html', import.meta.url), 'utf8');
    assert.match(html, /data-llm-deployment><\/div>\s*<div data-llm-hardware-warnings/);
    assert.match(html, /<dialog[^>]*id="localLlmAddDialog"[\s\S]*data-llm-form="add"/);
    assert.doesNotMatch(html, />Cancel<\/button>/);
    assert.ok(html.indexOf('data-llm-detail') < html.indexOf('<dialog'));
    assert.match(html, /<details[^>]*data-add-metadata/);
});

test('help uses native accessible popovers and escapes externally supplied descriptions', () => {
    const html = helpHtml('help', 'About <field>', '<img src=x onerror=1>');
    assert.match(html, /popovertarget="help"/);
    assert.match(html, /aria-label="About &lt;field&gt;"/);
    assert.match(html, /id="help" popover/);
    assert.doesNotMatch(html, /<img/);
    assert.match(html, /&lt;img/);
});

function target() {
    const listeners = new Map();
    return {
        listeners, open: false,
        addEventListener(name, handler) { listeners.set(name, handler); },
        removeEventListener(name) { listeners.delete(name); },
        fire(name, event) { listeners.get(name)?.(event); },
        setAttribute() {}, focus() {},
        showModal() { this.open = true; }, close() { this.open = false; },
        getBoundingClientRect: () => ({ left: 10, right: 100, top: 10, bottom: 100 }),
    };
}

test('add popup preserves parent selection and draft; Escape cannot dismiss a pending add', () => {
    const p = new LocalLlmDashboard({ querySelectorAll: () => [] }, () => {});
    p.addDialog = target();
    p.addButton = target();
    p.addForm = target();
    p.selectedModelId = 'selected';
    p.detailMode = 'model';
    p.addForm.draft = 'owner/draft';
    const dispose = initAddModelDialog({ dialog: p.addDialog, trigger: p.addButton, form: p.addForm, isBusy: () => p.busy, onClose: () => p.cancelAddModel() });
    p.openAddModel();
    assert.equal(p.addDialog.open, true);
    assert.equal(p.detailMode, 'model');
    p.busy = true;
    const event = { preventDefault() {}, stopPropagation() {} };
    p.addDialog.fire('cancel', event);
    assert.equal(p.addDialog.open, true);
    p.busy = false;
    p.addDialog.fire('cancel', event);
    assert.equal(p.addDialog.open, false);
    assert.equal(p.selectedModelId, 'selected');
    assert.equal(p.addForm.draft, 'owner/draft');
    p.openAddModel();
    p.addDialog.fire('click', { target: p.addDialog, clientX: 0, clientY: 0 });
    assert.equal(p.addDialog.open, false);
    const metadata = { open: false };
    p.addForm.fire('invalid', { target: { closest: () => metadata } });
    assert.equal(metadata.open, true);
    p.openAddModel();
    dispose();
    assert.equal(p.addDialog.open, false);
    assert.equal(p.addDialog.listeners.size, 0);
});

test('busy operations use Explorer loader tokens and release them once on success, failure and disposal', async t => {
    const previous = globalThis.window;
    const shown = [];
    const hidden = [];
    globalThis.window = { webSkel: {
        showLoading() { const token = {}; shown.push(token); return token; },
        hideLoading(token) { hidden.push(token); },
    } };
    t.after(() => {
        if (previous === undefined) delete globalThis.window;
        else globalThis.window = previous;
    });
    const p = new LocalLlmDashboard({ querySelectorAll: () => [], querySelector: () => null }, () => {});
    p.statusLine = { textContent: '', classList: { toggle() {} } };
    let finish;
    const pending = p.withBusy('Waiting…', () => new Promise(resolve => { finish = resolve; }));
    assert.equal(shown.length, 1);
    assert.equal(await p.withBusy('Duplicate', async () => 'duplicate'), undefined);
    finish('done');
    assert.equal(await pending, 'done');
    assert.deepEqual(hidden, [shown[0]]);
    await p.withBusy('Failing…', async () => { throw new Error('Action failed'); });
    assert.equal(p.statusLine.textContent, 'Action failed');
    assert.deepEqual(hidden, shown);
    const disposing = p.withBusy('Waiting…', () => new Promise(resolve => { finish = resolve; }));
    p.afterUnload();
    assert.deepEqual(hidden, shown);
    finish('late');
    await disposing;
    assert.deepEqual(hidden, shown);
    assert.equal(p.busyLoader, null);
    assert.equal(await p.withBusy('Closed', async () => 'closed'), undefined);
    const css = await readFile(new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.css', import.meta.url), 'utf8');
    assert.match(css, /#localLlmStatus\.loading,[\s\S]*clip-path: inset\(50%\)/);
    assert.match(css, /#localLlmStatus\.loading::before,[\s\S]*content: none/);
});

test('success feedback uses an escaped Explorer toast, expires and cannot clear a newer error', t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const previous = globalThis.assistOS;
    const toasts = [];
    globalThis.assistOS = { showToast: (...args) => toasts.push(args) };
    t.after(() => {
        if (previous === undefined) delete globalThis.assistOS;
        else globalThis.assistOS = previous;
    });
    const p = new LocalLlmDashboard({ querySelectorAll: () => [] }, () => {});
    p.statusLine = { textContent: '', classList: { toggle() {} } };
    p.setStatus('<model> added.', 'success');
    assert.deepEqual(toasts, [['&lt;model&gt; added.', 'success', 5000]]);
    assert.equal(p.statusLine.textContent, '<model> added.');
    t.mock.timers.tick(5000);
    assert.equal(p.statusLine.textContent, '');
    p.setStatus('Run request accepted.', 'success');
    p.setStatus('Runner failed.', 'error');
    t.mock.timers.tick(5000);
    assert.equal(p.statusLine.textContent, 'Runner failed.');
    p.setStatus('Stopped.', 'success');
    p.afterUnload();
    p.setStatus('Late success.', 'success');
    assert.equal(toasts.length, 3);
    assert.equal(p.statusTimer, null);
    t.mock.timers.tick(5000);
    assert.equal(p.statusLine.textContent, 'Stopped.');
});

test('popup errors stay inside the popup and closed presenters ignore late feedback', () => {
    const p = new LocalLlmDashboard({}, () => {});
    const line = () => ({ textContent: '', classList: { toggle() {} } });
    p.statusLine = line();
    p.addStatusLine = line();
    p.addDialog = { open: true };
    p.setStatus('Lookup failed', 'error');
    assert.equal(p.statusLine.textContent, '');
    assert.equal(p.addStatusLine.textContent, 'Lookup failed');
    p.closed = true;
    p.setStatus('Late error', 'error');
    assert.equal(p.addStatusLine.textContent, 'Lookup failed');
});
