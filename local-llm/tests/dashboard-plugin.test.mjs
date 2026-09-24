import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import * as dashboardModule from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.js';
import {
    detailInfoHtml,
    estimateHtml,
    hardwareCardsHtml,
    modelsTableHtml,
    splitRunFields,
    statusCardHtml,
} from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';
import { fieldsFromSchema } from '../IDE-plugins/local-llm-settings/local-llm-settings-model.js';
import { RUNNERS } from '../src/runners/index.mjs';

const ROOT = new URL('..', import.meta.url);
const BUTTON = new URL('IDE-plugins/local-llm-tool-button/', ROOT);
const DASHBOARD = new URL('components/local-llm-dashboard/', BUTTON);
const readText = (url) => fs.readFileSync(url, 'utf8');
const readJson = (url) => JSON.parse(readText(url));

const OVERVIEW_MODELS = [
    {
        id: 'gpt-oss-20b', displayName: 'gpt-oss-20b', architecture: 'moe', seed: true, license: 'Apache-2.0', totalParams: '21B', activeParams: '3.6B',
        runners: {
            'llama.cpp': { size: 12e9, download: { state: 'complete', total: 12e9 }, admission: { status: 'ok' } },
            ollama: { size: 13.8e9, download: { state: 'absent' }, admission: { status: 'ok' } },
        },
    },
    {
        id: 'qwen3-0.6b', displayName: 'Qwen3 0.6B <b>', architecture: 'dense', seed: false, license: 'Apache-2.0',
        runners: { 'llama.cpp': { size: 6.4e8, download: { state: 'complete', total: 6.4e8 }, admission: { status: 'ok' } } },
    },
];
const RUNNER_LIST = [
    { id: 'llama.cpp', supported: true, installed: true },
    { id: 'ollama', supported: true, installed: true },
    { id: 'vllm', supported: false, installed: false },
];

test('the toolbar plugin sits next to Soul Gateway, is admin-only and opens the dashboard in component mode', () => {
    const config = readJson(new URL('config.json', BUTTON));
    assert.equal(config.id, 'local-llm-tool-button');
    assert.equal(config.component, 'local-llm-tool-button');
    assert.equal(config.presenter, 'LocalLlmToolButton');
    assert.equal(config.type, 'embedded');
    assert.deepEqual(config.location, ['file-exp:toolbar']);
    assert.equal(config.locationOrder, 295);
    assert.equal(config.adminOnly, true);
    assert.deepEqual(config.toolbarModal, { mode: 'component', component: 'local-llm-dashboard', title: 'Local LLMs' });
    const dashboard = config.dependencies.find((entry) => entry.component === 'local-llm-dashboard');
    assert.deepEqual(dashboard, { component: 'local-llm-dashboard', presenter: 'LocalLlmDashboard', type: 'embedded' });
    for (const name of ['local-llm-dashboard.html', 'local-llm-dashboard.css', 'local-llm-dashboard.js']) {
        assert.ok(fs.existsSync(new URL(name, DASHBOARD)), name);
    }
    for (const name of ['local-llm-tool-button.html', 'local-llm-tool-button.css', 'local-llm-tool-button.js', 'icon.svg']) {
        assert.ok(fs.existsSync(new URL(name, BUTTON)), name);
    }
    assert.equal(typeof dashboardModule.LocalLlmDashboard, 'function');
});

test('the dashboard follows Explorer style rules and works in both themes through Explorer tokens', () => {
    const presenter = readText(new URL('local-llm-dashboard.js', DASHBOARD));
    const template = readText(new URL('local-llm-dashboard.html', DASHBOARD));
    const styles = readText(new URL('local-llm-dashboard.css', DASHBOARD));
    const buttonStyles = readText(new URL('local-llm-tool-button.css', BUTTON));
    assert.match(presenter, /constructor\(element, invalidate\)[\s\S]*?this\.invalidate\(\);/);
    assert.match(presenter, /webSkelPresenter\?\.setOptions/);
    assert.match(template, /class="settings-tabs"[\s\S]*?role="tablist"/);
    for (const tab of ['models', 'playground', 'logs']) {
        assert.match(template, new RegExp(`data-llm-tab="${tab}"[\\s\\S]*?role="tab"`));
        assert.match(template, new RegExp(`data-llm-panel="${tab}"[\\s\\S]*?role="tabpanel"`));
    }
    assert.match(template, /<custom-select id="localLlmRunner"/);
    assert.doesNotMatch(template, /<select\b/);
    assert.doesNotMatch(presenter, /<select\b/);
    assert.doesNotMatch(presenter, /window\.confirm|window\.prompt|window\.alert/);
    assert.doesNotMatch(template, /\$\$/, 'WebSkel treats $$name as a template variable');
    assert.doesNotMatch(template, /\/explorer\/assets\/icons\//);
    for (const css of [styles, buttonStyles]) {
        assert.doesNotMatch(css, /#[0-9a-f]{3,8}\b/i);
        assert.doesNotMatch(css, /(?:color|background|border(?:-color|-radius)?|box-shadow):/);
        assert.doesNotMatch(css, /min-width:\s*(?:[1-9]\d*px|min\()/);
        assert.doesNotMatch(css, /\/\*/, 'WebSkel mis-scopes the rule after a CSS comment');
    }
    assert.match(styles, /var\(--(?:space|font)-/);
    assert.match(styles, /@media \(max-width: 720px\)/);
    const scoped = (css, pattern) => {
        for (const rule of css.replace(/@media[^{]*\{/g, '').split('}')) {
            const selector = rule.split('{')[0].trim();
            if (!selector) continue;
            for (const part of selector.split(',')) assert.match(part.trim(), pattern, part.trim());
        }
    };
    scoped(styles, /^local-llm-dashboard\b/);
    scoped(buttonStyles, /^(?:\.app-toolbar-slot )?local-llm-tool-button\b/);
});

test('the Models table has no buttons; the detail panel holds the one Run and the model actions', () => {
    const table = modelsTableHtml(OVERVIEW_MODELS, { selectedId: 'qwen3-0.6b', activeModelId: 'gpt-oss-20b' });
    assert.doesNotMatch(table, /<button/);
    assert.doesNotMatch(table, />\s*Run/);
    assert.match(table, /data-model-id="qwen3-0.6b"[^>]*aria-selected="true"/);
    assert.match(table, /class="local-llm-model-row active"/);
    assert.match(table, /status-badge success">in use/);
    assert.match(table, /Qwen3 0\.6B &lt;b&gt;/, 'names are escaped');
    assert.match(table, /<td class="settings-card-meta">—<\/td>/, 'no Ollama source shows a dash');

    const template = readText(new URL('local-llm-dashboard.html', DASHBOARD));
    assert.equal((template.match(/data-run-submit/g) || []).length, 1);
    assert.equal((template.match(/>\s*Run\s*</g) || []).length, 1);

    const catalog = detailInfoHtml(OVERVIEW_MODELS[0], { runners: RUNNER_LIST });
    assert.doesNotMatch(catalog, /removeModel/, 'catalog models cannot be removed');
    assert.match(catalog, /data-local-action="deleteWeights gpt-oss-20b llama\.cpp"/);
    assert.doesNotMatch(catalog, /deleteWeights gpt-oss-20b ollama/, 'nothing to delete for Ollama');
    const added = detailInfoHtml(OVERVIEW_MODELS[1], { runners: RUNNER_LIST });
    assert.match(added, /data-local-action="removeModel qwen3-0\.6b"\s+disabled title="Delete its weights first"/,
        'the server removes a model only without weights on disk');
    assert.match(added, /Delete its weights first to remove it\./);
    const emptyAdded = { ...OVERVIEW_MODELS[1], runners: { 'llama.cpp': { size: 1, download: { state: 'absent' }, admission: { status: 'ok' } } } };
    const removable = detailInfoHtml(emptyAdded, { runners: RUNNER_LIST });
    assert.match(removable, /data-local-action="removeModel qwen3-0\.6b"\s*>Remove/);
    assert.doesNotMatch(removable, /Delete its weights first/);
    assert.doesNotMatch(added, />\s*Run\s*</);
    assert.equal(detailInfoHtml(null), '');
    assert.equal(modelsTableHtml([]), '<div class="settings-empty-state">No models in the catalog.</div>');
});

test('basic run settings are the ones that decide the fit; everything else is under a closed Advanced', () => {
    const llama = fieldsFromSchema(RUNNERS['llama.cpp'].paramSchema, {});
    const dense = splitRunFields(llama, 'llama.cpp', { architecture: 'dense' });
    assert.deepEqual(dense.basic.map((field) => field.name), ['ctxSize']);
    assert.equal(dense.basic.length + dense.advanced.length, llama.length);
    const moe = splitRunFields(llama, 'llama.cpp', { architecture: 'moe' });
    assert.deepEqual(moe.basic.map((field) => field.name), ['ctxSize', 'nCpuMoe']);
    assert.ok(!moe.advanced.some((field) => ['ctxSize', 'nCpuMoe'].includes(field.name)));
    const ollama = splitRunFields(fieldsFromSchema(RUNNERS.ollama.paramSchema, {}), 'ollama', { architecture: 'moe' });
    assert.deepEqual(ollama.basic.map((field) => field.name), ['numCtx']);
    assert.deepEqual(splitRunFields([], 'vllm', {}), { basic: [], advanced: [] });
    const template = readText(new URL('local-llm-dashboard.html', DASHBOARD));
    assert.match(template, /<details class="local-llm-advanced" data-run-advanced>/, 'Advanced starts closed');
});

test('the summary cards show live GPU use, and the status card names what runs and how to stop it', () => {
    const hardware = {
        gpu: { available: true, name: 'RTX', usedBytes: 13 * 2 ** 20, totalBytes: 6144 * 2 ** 20, processes: [] },
        memory: { availableBytes: 24000 * 2 ** 20, totalBytes: 31301 * 2 ** 20 },
        disk: { freeBytes: 290e9, totalBytes: 532e9 },
    };
    const idle = hardwareCardsHtml(hardware);
    assert.match(idle, /13 MiB of 6,144 MiB/);
    assert.match(idle, /<meter class="local-llm-meter"/);
    const busy = hardwareCardsHtml(hardware, { available: true, usedBytes: 2593 * 2 ** 20, totalBytes: 6144 * 2 ** 20, processes: [{ name: 'llama-server' }] });
    assert.match(busy, /2,593 MiB of 6,144 MiB/, 'a fresher status poll wins');
    assert.match(busy, /in use by llama-server/);
    assert.match(hardwareCardsHtml({ gpu: { available: false, reason: 'revoked' } }), /Not available[\s\S]*revoked/);

    const none = statusCardHtml({ phase: 'idle', deployment: { modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', phase: 'idle' } });
    assert.match(none, />None</);
    assert.match(none, /Last run: gpt-oss-20b on llama\.cpp/);
    assert.match(none, /data-llm-stop/);
    const downloading = statusCardHtml({ phase: 'downloading', deployment: { modelId: 'm', runnerId: 'ollama', download: { bytes: 5, total: 10, rate: 1e6, etaSeconds: 5 } } });
    assert.match(downloading, /<progress class="local-llm-progress" max="100" value="50"/);
    assert.match(downloading, /data-llm-cancel/);
    const ready = statusCardHtml({
        phase: 'ready',
        deployment: { modelId: 'qwen3-0.6b', runnerId: 'llama.cpp' },
        runnerReport: { device: 'CUDA0', offloaded: { layers: 29, of: 29 }, totalMiB: 2438 },
        lastCompletion: { completionTokens: 7, generationTokensPerSecond: 223.4, source: 'runner timings' },
    });
    assert.match(ready, /qwen3-0\.6b on llama\.cpp/);
    assert.match(ready, /status-badge success">ready/);
    assert.match(ready, /device CUDA0 · 29\/29 layers on the GPU · buffers 2,438 MiB/);
    assert.match(ready, /7 tokens at 223\.4 tokens\/s/);

    const estimate = estimateHtml({ admission: { status: 'ok', estimate: { gpuBytes: 2051 * 2 ** 20, ramBytes: 768 * 2 ** 20, basis: 'file size heuristic' } } }, hardware);
    assert.match(estimate, /GPU about 2,051 MiB of 6,144 MiB/);
    assert.match(estimate, /RAM about 768 MiB of 24,000 MiB available/);
    assert.match(estimateHtml({ error: 'Context size must be a whole number.' }), /settings-status error/);
});

// Fake timers and a fake local-llm client for the dashboard's polling.
function pollingHarness(t, callTool, { document } = {}) {
    const saved = {
        setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout, window: globalThis.window, document: globalThis.document,
    };
    const pending = new Map();
    let nextId = 1;
    globalThis.setTimeout = (fn, ms) => { const id = nextId++; pending.set(id, { fn, ms }); return id; };
    globalThis.clearTimeout = (id) => { pending.delete(id); };
    globalThis.window = { webSkel: { appServices: { getClient: () => ({ callTool }) } } };
    if (document) globalThis.document = document;
    t.after(() => Object.assign(globalThis, saved));
    const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise((resolve) => saved.setTimeout(resolve, 0)); };
    const fireAll = async () => {
        const due = [...pending.values()];
        pending.clear();
        for (const timer of due) timer.fn();
        await settle();
    };
    const presenter = new dashboardModule.LocalLlmDashboard({ isConnected: true, querySelector: () => null, querySelectorAll: () => [] }, () => {});
    return { presenter, pending, fireAll, settle };
}

const statusText = (phase) => ({ content: [{ type: 'text', text: JSON.stringify({
    phase, deployment: { phase, modelId: 'm', runnerId: 'llama.cpp' }, logs: [], nextSeq: 0,
}) }] });

test('each Send, Stop or Cancel keeps a single polling chain', async (t) => {
    let statusCalls = 0;
    const h = pollingHarness(t, async (tool) => {
        if (tool === 'local_llm_status') { statusCalls += 1; return statusText('ready'); }
        if (tool === 'local_llm_test_prompt') return { content: [{ type: 'text', text: JSON.stringify({ text: 'PONG', elapsedMs: 5 }) }] };
        return { content: [{ type: 'text', text: '{}' }] };
    });
    const p = h.presenter;
    p.status = { phase: 'ready' };
    p.promptForm = { reportValidity: () => true, elements: { prompt: { value: 'hi' }, maxTokens: { value: '16' } } };
    p.promptResult = { innerHTML: '' };
    p.startPollingIfActive();
    await h.fireAll();
    assert.equal(h.pending.size, 1);
    for (let send = 1; send <= 3; send += 1) {
        await p.submitPrompt({ preventDefault() {} });
        await h.settle();
        assert.equal(h.pending.size, 1, `after Send #${send}`);
    }
    await p.stopDeployment();
    await h.settle();
    await p.cancelDownload();
    await h.settle();
    assert.equal(h.pending.size, 1, 'after Stop and Cancel');
    const before = statusCalls;
    await h.fireAll();
    assert.equal(statusCalls - before, 1, 'one status call per tick');
    p.closed = true;
    await h.fireAll();
    assert.equal(h.pending.size, 0);
});

test('a failed status call backs off and keeps polling while the dashboard is open', async (t) => {
    let calls = 0;
    const h = pollingHarness(t, async () => {
        calls += 1;
        if (calls === 2) throw new Error('transient router 502');
        return statusText('downloading');
    });
    const p = h.presenter;
    p.status = { phase: 'downloading' };
    p.startPollingIfActive();
    await h.fireAll();
    assert.equal(calls, 1);
    await h.fireAll();
    assert.equal(calls, 2);
    assert.equal(h.pending.size, 1, 'still polling after the failure');
    assert.ok([...h.pending.values()][0].ms > 1500, 'with a backoff');
    await h.fireAll();
    assert.equal(calls, 3);
    assert.equal(h.pending.size, 1);
    assert.equal([...h.pending.values()][0].ms, 1500, 'back to the normal interval after a success');
});

function deferred() {
    let resolve;
    const promise = new Promise((ok) => { resolve = ok; });
    return { promise, resolve };
}

test('at most one status request is in flight; a poll asked for meanwhile runs once after it', { timeout: 5000 }, async (t) => {
    const replies = [];
    let inFlight = 0;
    let maxInFlight = 0;
    const h = pollingHarness(t, async () => {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        const reply = deferred();
        replies.push(reply);
        try {
            return await reply.promise;
        } finally {
            inFlight -= 1;
        }
    });
    const p = h.presenter;
    p.status = { phase: 'downloading' };
    const first = p.poll();
    await h.settle();
    assert.equal(replies.length, 1);
    const second = p.poll();
    p.startPollingIfActive(true);
    await h.settle();
    assert.equal(replies.length, 1);
    assert.equal(h.pending.size, 0);
    await second;
    replies[0].resolve(statusText('downloading'));
    await first;
    await h.settle();
    assert.equal(h.pending.size, 1);
    assert.equal([...h.pending.values()][0].ms, 0, 'the follow-up runs at once');
    await h.fireAll();
    assert.equal(replies.length, 2);
    assert.equal(maxInFlight, 1);
    replies[1].resolve(statusText('ready'));
    await h.settle();
});

test('while open and visible, the overview refreshes on a timer; hidden or closed, it stops', async (t) => {
    const document = { hidden: false, addEventListener() {}, removeEventListener() {} };
    let overviews = 0;
    const h = pollingHarness(t, async (tool) => {
        if (tool === 'local_llm_overview') overviews += 1;
        return { content: [{ type: 'text', text: JSON.stringify({ hardware: {}, models: [], runners: [], deployment: null }) }] };
    }, { document });
    const p = h.presenter;
    p.scheduleOverviewRefresh();
    assert.deepEqual([...h.pending.values()].map((timer) => timer.ms), [dashboardModule.OVERVIEW_REFRESH_MS]);
    await h.fireAll();
    assert.equal(overviews, 1);
    assert.equal(h.pending.size, 1, 'rescheduled');
    document.hidden = true;
    await h.fireAll();
    assert.equal(overviews, 1, 'no refresh while the tab is hidden');
    assert.equal(h.pending.size, 0, 'and no timer until it is visible again');
    document.hidden = false;
    p.onVisibilityChange();
    assert.deepEqual([...h.pending.values()].map((timer) => timer.ms), [0], 'refreshes at once when visible again');
    await h.fireAll();
    assert.equal(overviews, 2);
    p.afterUnload();
    assert.equal(h.pending.size, 0, 'unload stops the timer');
    await h.fireAll();
    assert.equal(overviews, 2);
});
