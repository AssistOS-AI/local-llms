import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import * as dashboardModule from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.js';
import {
    detailInfoHtml,
    estimateHtml,
    hardwareCardsHtml,
    installMessage,
    lookupResultsHtml,
    modelsTableHtml,
    runnersIntroText,
    runnersPanelHtml,
    splitRunFields,
    statusCardHtml,
    tableRunners,
} from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';
import { fieldsFromSchema, rememberRunners, runnerOptions, stripBidi } from '../IDE-plugins/local-llm-settings/local-llm-settings-model.js';
import { RUNNERS } from '../src/runners/index.mjs';

const ROOT = new URL('..', import.meta.url);
const BUTTON = new URL('IDE-plugins/local-llm-tool-button/', ROOT);
const DASHBOARD = new URL('components/local-llm-dashboard/', BUTTON);
const readText = (url) => fs.readFileSync(url, 'utf8');
const readJson = (url) => JSON.parse(readText(url));

const OVERVIEW_MODELS = [
    {
        id: 'gpt-oss-20b', displayName: 'gpt-oss-20b', architecture: 'moe', seed: true, license: 'Apache-2.0', totalParams: '21B', activeParams: '3.6B',
        weights: {
            gguf: { label: 'GGUF file', size: 12e9, download: { state: 'complete', total: 12e9 }, runners: ['llama.cpp', 'fake-gguf'] },
            ollama: { label: 'Ollama tag', size: 13.8e9, download: { state: 'absent' }, runners: ['ollama'] },
        },
        runners: {
            'llama.cpp': { format: 'gguf', size: 12e9, download: { state: 'complete', total: 12e9 }, admission: { status: 'ok' } },
            'fake-gguf': { format: 'gguf', size: 12e9, download: { state: 'complete', total: 12e9 }, admission: { status: 'ok' } },
            ollama: { format: 'ollama', size: 13.8e9, download: { state: 'absent' }, admission: { status: 'ok' } },
        },
    },
    {
        id: 'qwen3-0.6b', displayName: 'Qwen3 0.6B <b>', architecture: 'dense', seed: false, license: 'Apache-2.0',
        weights: { gguf: { label: 'GGUF file', size: 6.4e8, download: { state: 'complete', total: 6.4e8 }, runners: ['llama.cpp', 'fake-gguf'] } },
        runners: {
            'llama.cpp': { format: 'gguf', size: 6.4e8, download: { state: 'complete', total: 6.4e8 }, admission: { status: 'ok' } },
            'fake-gguf': { format: 'gguf', size: 6.4e8, download: { state: 'complete', total: 6.4e8 }, admission: { status: 'ok' } },
        },
    },
];
// A third supported runner the dashboard has never heard of.
const RUNNER_LIST = [
    { id: 'llama.cpp', displayName: 'llama.cpp', supported: true, installed: true },
    { id: 'fake-gguf', displayName: 'Fake <GGUF>', supported: true, installed: true },
    { id: 'ollama', displayName: 'Ollama', supported: true, installed: true },
    { id: 'vllm', displayName: 'vLLM', supported: false, installed: false },
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
    const table = modelsTableHtml(OVERVIEW_MODELS, { selectedId: 'qwen3-0.6b', activeModelId: 'gpt-oss-20b', runners: RUNNER_LIST });
    // One column per supported runner, named by the runner itself.
    assert.deepEqual([...table.matchAll(/<th scope="col">([^<]*)<\/th>/g)].map((match) => match[1]),
        ['Model', 'Licence', 'llama.cpp', 'Fake &lt;GGUF&gt;', 'Ollama']);
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
    // Weights are listed once per format, whatever the number of runners reading them.
    assert.equal((catalog.match(/data-local-action="deleteWeights /g) || []).length, 1);
    assert.match(catalog, /data-local-action="deleteWeights gpt-oss-20b gguf"/);
    assert.match(catalog, /GGUF file[\s\S]*used by llama\.cpp, Fake &lt;GGUF&gt;/);
    assert.doesNotMatch(catalog, /deleteWeights gpt-oss-20b ollama/, 'nothing to delete for Ollama');
    assert.match(catalog, /Fake &lt;GGUF&gt;[\s\S]*Fits this machine/, 'every supported runner shows its fit');
    const added = detailInfoHtml(OVERVIEW_MODELS[1], { runners: RUNNER_LIST });
    assert.match(added, /data-local-action="removeModel qwen3-0\.6b"\s+disabled title="Delete its weights first"/,
        'the server removes a model only without weights on disk');
    assert.match(added, /Delete its weights first to remove it\./);
    const emptyAdded = {
        ...OVERVIEW_MODELS[1],
        weights: { gguf: { label: 'GGUF file', size: 1, download: { state: 'absent' }, runners: ['llama.cpp'] } },
        runners: { 'llama.cpp': { format: 'gguf', size: 1, download: { state: 'absent' }, admission: { status: 'ok' } } },
    };
    const removable = detailInfoHtml(emptyAdded, { runners: RUNNER_LIST });
    assert.match(removable, /data-local-action="removeModel qwen3-0\.6b"\s*>Remove/);
    assert.doesNotMatch(removable, /Delete its weights first/);
    assert.doesNotMatch(added, />\s*Run\s*</);
    assert.equal(detailInfoHtml(null), '');
    assert.equal(modelsTableHtml([]), '<div class="settings-empty-state">No models in the catalog.</div>');
});

test('the Runners tab offers Install and Uninstall only for on-demand runners, with size, licence and progress', () => {
    const licence = { name: 'AGPL-3.0', url: 'https://github.com/example/tabby/blob/main/LICENSE', source: 'https://github.com/example/tabby',
        notice: 'Installed from upstream, not redistributed.', requiresAcceptance: true };
    const runners = [
        { id: 'llama.cpp', displayName: 'llama.cpp', supported: true, installed: true, version: 'b11159' },
        { id: 'vllm', displayName: 'vLLM', supported: true, installed: false, version: null,
            install: { version: '0.30.0', totalBytes: 4.2e9, licence: { name: 'Apache-2.0', url: 'https://x.example/L' }, installed: false, runnable: false, state: null } },
        { id: 'tabby', displayName: 'Tabby <API>', supported: true, installed: true, version: 'abc1234',
            install: { version: 'abc1234', totalBytes: 3e9, licence, installed: true, runnable: true,
                state: { phase: 'installed', licence: { acceptedBy: 'admin@example.com', acceptedAt: '2026-09-24T12:00:00.000Z' }, rebuild: { seconds: 42.5 } } } },
        { id: 'busy', displayName: 'Busy', supported: true, installed: false,
            install: { version: '1', totalBytes: 1e9, licence: { name: 'MIT', url: 'https://x.example/M' }, installed: false, runnable: false, installing: true,
                state: { phase: 'downloading', download: { bytes: 2.5e8, total: 1e9, rate: 5e6 } } } },
    ];
    const idle = runnersPanelHtml(runners.filter((runner) => runner.id !== 'busy'));
    assert.match(idle, /data-local-action="installRunner vllm"/);
    const html = runnersPanelHtml(runners);
    assert.doesNotMatch(html, /installRunner llama\.cpp|uninstallRunner llama\.cpp/, 'runners in the image are not installed on demand');
    assert.doesNotMatch(html, /data-local-action="installRunner/, 'no Install anywhere while one install runs');
    assert.match(html, /4\.2 GB/);
    assert.match(html, /data-local-action="uninstallRunner tabby"/);
    assert.match(html, /Tabby &lt;API&gt;/);
    assert.match(html, /AGPL-3\.0[\s\S]*accepted by admin@example\.com/);
    assert.match(html, /href="https:\/\/github\.com\/example\/tabby"/);
    assert.match(html, /<progress[^>]*value="25"/);
});

test('the Runners tab shows the platform reason for a runner this image lacks, never the version probe', async () => {
    const platform = 'ik_llama.cpp is not available on this platform: this image does not include it.';
    const probe = 'Executable not found at /opt/ik_llama.cpp/llama-server';
    const lacking = { id: 'ik_llama.cpp', displayName: 'ik_llama.cpp', supported: false, installed: false, unsupportedReason: platform, reason: platform };
    // The card reads `unsupportedReason` first, even when `reason` still carries the probe text.
    const html = runnersPanelHtml([{ ...lacking, reason: probe }]);
    assert.match(html, /not available on this platform: this image does not include it/);
    assert.doesNotMatch(html, /Executable not found/);
    // A runner that is supported but not installed keeps the probe's reason.
    assert.match(runnersPanelHtml([{ id: 'ollama', displayName: 'Ollama', supported: true, installed: false, reason: 'Executable not found at /opt/ollama/bin/ollama' }]),
        /Executable not found at \/opt\/ollama\/bin\/ollama/);
    // Run refuses with the same words.
    const statuses = [];
    const presenter = new dashboardModule.LocalLlmDashboard({ isConnected: true, querySelector: () => null, querySelectorAll: () => [] }, () => {});
    presenter.overview = { runners: [{ ...lacking, reason: probe }], models: [{ id: 'm', runners: {} }] };
    presenter.runModelId = 'm';
    presenter.runnerSelect = { value: 'ik_llama.cpp' };
    presenter.runForm = { reportValidity: () => true };
    presenter.statusLine = { textContent: '', classList: { toggle(state, on) { if (on) statuses.push(state); } } };
    await presenter.submitRun({ preventDefault() {} });
    assert.equal(presenter.statusLine.textContent, platform);
    assert.deepEqual(statuses, ['error']);
});

test('the Runners intro lists the runners this image contains', () => {
    const amd64 = [
        { id: 'llama.cpp', displayName: 'llama.cpp', supported: true, installed: true },
        { id: 'ik_llama.cpp', displayName: 'ik_llama.cpp', supported: true, installed: true },
        { id: 'ollama', displayName: 'Ollama', supported: true, installed: true },
        { id: 'vllm', displayName: 'vLLM', supported: true, installed: false, install: { version: '0.30.0', totalBytes: 4.2e9 } },
    ];
    const withInstall = runnersIntroText(amd64);
    assert.match(withInstall, /^In this image: llama\.cpp, ik_llama\.cpp, Ollama\. Other runners are installed here, only when you press Install/);
    assert.doesNotMatch(withInstall, /vLLM/);
    // A runner the image lacks is not "in this image", and with nothing installable the second sentence says so.
    const arm64 = [
        { id: 'llama.cpp', displayName: 'llama.cpp', supported: true, installed: true },
        { id: 'ik_llama.cpp', displayName: 'ik_llama.cpp', supported: false, installed: false, unsupportedReason: 'not available on this platform' },
    ];
    assert.equal(runnersIntroText(arm64), 'In this image: llama.cpp. No other runner can be installed on this image.');
    assert.equal(runnersIntroText([]), '');
    assert.equal(runnersIntroText(undefined), '');
    // The static paragraph is gone; the presenter fills the placeholder from the overview.
    const html = readText(new URL('local-llm-dashboard.html', DASHBOARD));
    assert.match(html, /<p class="settings-section-description" data-llm-runners-intro><\/p>/);
    assert.doesNotMatch(html, /part of the agent/);
    const presenter = new dashboardModule.LocalLlmDashboard({ isConnected: true, querySelector: () => null, querySelectorAll: () => [] }, () => {});
    presenter.runnersIntro = { textContent: '' };
    presenter.runnersRegion = { innerHTML: '' };
    presenter.overview = { runners: arm64 };
    presenter.renderRunners();
    assert.equal(presenter.runnersIntro.textContent, 'In this image: llama.cpp. No other runner can be installed on this image.');
    assert.match(presenter.runnersRegion.innerHTML, /not available on this platform/);
});

test('basic run settings are the ones that decide the fit; everything else is under a closed Advanced', () => {
    const llama = fieldsFromSchema(RUNNERS['llama.cpp'].paramSchema, {});
    const dense = splitRunFields(llama, RUNNERS['llama.cpp'], { architecture: 'dense' });
    assert.deepEqual(dense.basic.map((field) => field.name), ['ctxSize']);
    assert.equal(dense.basic.length + dense.advanced.length, llama.length);
    const moe = splitRunFields(llama, RUNNERS['llama.cpp'], { architecture: 'moe' });
    assert.deepEqual(moe.basic.map((field) => field.name), ['ctxSize', 'nCpuMoe']);
    assert.ok(!moe.advanced.some((field) => ['ctxSize', 'nCpuMoe'].includes(field.name)));
    const ollama = splitRunFields(fieldsFromSchema(RUNNERS.ollama.paramSchema, {}), RUNNERS.ollama, { architecture: 'moe' });
    assert.deepEqual(ollama.basic.map((field) => field.name), ['numCtx']);
    assert.deepEqual(splitRunFields([], RUNNERS.vllm, {}), { basic: [], advanced: [] });
    // A runner the dashboard has never heard of names its own basic fields.
    const custom = splitRunFields(fieldsFromSchema({ properties: { a: { type: 'integer' }, b: { type: 'integer' } } }, {}),
        { id: 'fake', basicParams: ['b'], moeParams: [] }, { architecture: 'dense' });
    assert.deepEqual(custom.basic.map((field) => field.name), ['b']);
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

test('on unified memory one card shows the shared pool, and the run form one shared-memory estimate', () => {
    const hardware = {
        gpu: { available: true, memoryModel: 'unified', name: 'NVIDIA GB10', totalBytes: null, usedBytes: null, freeBytes: null,
            processes: [{ name: '/opt/llama.cpp/llama-server', usedBytes: 11875 * 2 ** 20 }] },
        memory: { totalBytes: 122502 * 2 ** 20, availableBytes: 96000 * 2 ** 20, freeBytes: 80000 * 2 ** 20, cachedBytes: 14000 * 2 ** 20 },
        disk: { freeBytes: 434e9, totalBytes: 916e9 },
    };
    const cards = hardwareCardsHtml(hardware);
    assert.match(cards, /Unified memory available/);
    assert.match(cards, /96,000 MiB of 122,502 MiB/);
    assert.match(cards, /NVIDIA GB10 shares system memory · GPU in use by \/opt\/llama\.cpp\/llama-server · page cache 14,000 MiB/);
    const total = 122502 * 2 ** 20;
    assert.ok(cards.includes(`<meter class="local-llm-meter" min="0" max="${total}" value="${total - 96000 * 2 ** 20}"`));
    assert.doesNotMatch(cards, /GPU memory|RAM available|—|NaN|null/, 'no GPU or RAM card, and no missing value reaches a card');
    assert.match(cards, /Disk free/);
    const estimate = estimateHtml({ admission: { status: 'ok', estimate: { unifiedBytes: 84 * 2 ** 30, basis: 'measured envelope' } } }, hardware);
    assert.match(estimate, /Shared memory about 86,016 MiB of 122,502 MiB, 96,000 MiB available now/);
    assert.doesNotMatch(estimate, /GPU about|RAM about/);
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

// LM Studio (runners plan I9, Phase R7): off unless this deployment's operator
// turned it on; proprietary, for internal use only, with its Terms shown and
// accepted before anything downloads.
const LMSTUDIO_LICENCE = {
    name: 'LM Studio Terms of Use, version August 23, 2026', url: 'https://lmstudio.ai/app-terms', requiresAcceptance: true, proprietary: true,
    notice: 'Internal use only: the Terms license it "solely for Your personal and / or internal business purposes".',
};
const REASON = 'LM Studio is not enabled on this deployment (internal use only). This deployment\'s operator enables it with "ploinky var LOCAL_LLM_LMSTUDIO internal-use" and a restart of local-llm.';

test('a runner the operator left off says why, offers no Install, and gets no column or Run option', () => {
    const off = { id: 'lmstudio', displayName: 'LM Studio (internal use only)', supported: true, installed: false, enabled: false, disabledReason: REASON,
        install: { version: '0.0.25-1', totalBytes: 1105623572, licence: LMSTUDIO_LICENCE, installed: false, runnable: false, state: null } };
    const html = runnersPanelHtml([off]);
    assert.doesNotMatch(html, /data-local-action="installRunner lmstudio"/);
    assert.match(html, /not enabled on this deployment/);
    assert.match(html, /ploinky var LOCAL_LLM_LMSTUDIO internal-use/);
    assert.match(html, /href="https:\/\/lmstudio\.ai\/app-terms"/, 'the Terms stay visible');
    // Installed earlier, then turned off: it can still be uninstalled to free the disk.
    const installedOff = { ...off, installed: true, install: { ...off.install, installed: true, runnable: true } };
    assert.match(runnersPanelHtml([installedOff]), /data-local-action="uninstallRunner lmstudio"/);
    // On, it installs like any on-demand runner.
    assert.match(runnersPanelHtml([{ ...off, enabled: true, disabledReason: undefined }]), /data-local-action="installRunner lmstudio"/);
    assert.deepEqual(tableRunners([...RUNNER_LIST, off]).map((runner) => runner.id), ['llama.cpp', 'fake-gguf', 'ollama']);
    const model = { runners: { 'llama.cpp': { admission: { status: 'ok' }, download: { state: 'complete' } }, lmstudio: { admission: { status: 'incompatible' } } } };
    // The dashboard learns runner names from the overview.
    rememberRunners([off]);
    const options = runnerOptions({ runners: [RUNNER_LIST[0], off] }, model);
    assert.equal(options.find((option) => option.value === 'lmstudio').label, 'LM Studio (internal use only) · not enabled on this deployment');
});

test('installing a proprietary runner shows its Terms, the internal-use rule and that the acceptance is recorded', () => {
    const runner = { id: 'lmstudio', displayName: 'LM Studio (internal use only)',
        install: { version: '0.0.25-1', totalBytes: 1105623572, licence: LMSTUDIO_LICENCE } };
    const message = installMessage(runner);
    assert.match(message, /^Install LM Studio \(internal use only\) 0\.0\.25-1\? It downloads 1\.1 GB of pinned files/);
    assert.match(message, /proprietary software, for internal use only/);
    assert.match(message, /LM Studio Terms of Use, version August 23, 2026 \(https:\/\/lmstudio\.ai\/app-terms\)/);
    assert.match(message, /internal business purposes/);
    assert.match(message, /accepts these terms on this workspace's behalf, and your acceptance is recorded/);
    // A notice that already says so is not followed by the same sentence again (seen live in L3).
    const said = installMessage({ ...runner, install: { ...runner.install, licence: { ...LMSTUDIO_LICENCE,
        notice: 'Installing downloads it and accepts the Terms on this workspace\'s behalf, and your acceptance is recorded.' } } });
    assert.equal(said.match(/acceptance is recorded/g).length, 1, said);
    // An open-source licence that needs acceptance keeps its wording; one that does not just names it.
    const agpl = installMessage({ id: 'tabbyapi', displayName: 'TabbyAPI', install: { version: 'f07131c', totalBytes: 4e9,
        licence: { name: 'AGPL-3.0', url: 'https://x.example/L', source: 'https://x.example/S', notice: 'AGPL notice.', requiresAcceptance: true } } });
    assert.match(agpl, /It is licensed under AGPL-3\.0 \(https:\/\/x\.example\/L; source: https:\/\/x\.example\/S\)\. AGPL notice\./);
    assert.doesNotMatch(agpl, /proprietary/);
    assert.match(installMessage({ id: 'vllm', install: { version: '0.30.0', totalBytes: 4e9, licence: { name: 'Apache-2.0' } } }), /Licence: Apache-2\.0\.$/);
});

// The cpu profile (DS005): no NVIDIA GPU is usable, so the dashboard shows the CPU, its memory and the disk.
const CPU_MIB = 1024 * 1024;
const CPU_REASON = 'No GPU is attached to this agent: GPU not applied to this Box yet';
const CPU_HARDWARE = {
    gpu: { available: false, state: 'absent', reason: CPU_REASON },
    memory: { totalBytes: 6036128 * 1024, availableBytes: 3122576 * 1024 },
    disk: { freeBytes: 100_000 * CPU_MIB, totalBytes: 400_000 * CPU_MIB },
    cpus: 4,
    cores: 4,
};
const CPU_LIMITS = { floorBytes: 618099508, hostReserveBytes: 1536 * CPU_MIB, poolBytes: 6036128 * 1024, availableBytes: 3122576 * 1024 };

test('on the cpu profile the cards show the CPU, memory with its floor and disk, and why no NVIDIA GPU is used', async () => {
    const html = hardwareCardsHtml(CPU_HARDWARE, null, { profile: 'cpu', decision: { cause: 'absent', reason: `${CPU_REASON}.` }, limits: CPU_LIMITS });
    assert.deepEqual([...html.matchAll(/data-llm-card="([a-z]+)"/g)].map((match) => match[1]), ['compute', 'memory', 'disk']);
    assert.match(html, />CPU · 4 cores</);
    assert.match(html, /No NVIDIA GPU is used: no NVIDIA GPU is attached\. No GPU is attached to this agent: GPU not applied to this Box yet\./);
    assert.match(html, />3,049 MiB of 5,895 MiB</);
    assert.match(html, /keeps 589 MiB free for the workspace/);
    assert.match(html, /<meter[^>]*aria-label="Memory in use"/);
    assert.match(html, /Disk free/);
    assert.doesNotMatch(html, /GPU memory|RAM available|Unified memory|NaN|null|undefined/);
    // Every cause leads its own wording.
    const lead = (cause) => hardwareCardsHtml(CPU_HARDWARE, null, { profile: 'cpu', decision: { cause, reason: 'r' }, limits: CPU_LIMITS });
    assert.match(lead('unusable'), /the NVIDIA GPU cannot be used\. r\./);
    assert.match(lead('mismatch'), /this image&#39;s CUDA runners were not built for this GPU\. r\./);
    assert.match(lead('unreadable-timeout'), /the NVIDIA GPU could not be read for 60 s\. r\./);
    assert.match(lead(undefined), /No NVIDIA GPU is used: no NVIDIA GPU is used\./);
    // Without the controller's limits the card falls back to the snapshot, and no missing value reaches it.
    const bare = hardwareCardsHtml({ memory: {}, disk: {} }, null, { profile: 'cpu' });
    assert.doesNotMatch(bare, /NaN|null|undefined/);
    assert.match(bare, />CPU</);
    assert.match(hardwareCardsHtml({ ...CPU_HARDWARE, cores: 1 }, null, { profile: 'cpu', limits: CPU_LIMITS }), />CPU · 1 core</);
    assert.match(hardwareCardsHtml({ ...CPU_HARDWARE, cores: undefined }, null, { profile: 'cpu', limits: CPU_LIMITS }), />CPU · 4 cores</, 'the logical CPUs when the cores are not reported');
    assert.match(hardwareCardsHtml(CPU_HARDWARE, null, { profile: 'cpu', limits: { ...CPU_LIMITS, poolBytes: 4096 * CPU_MIB, availableBytes: 2048 * CPU_MIB } }), />2,048 MiB of 4,096 MiB</, 'a container limit caps the pool');
    // Without options, or on another profile, the cards are exactly what they were.
    const unified = { gpu: { available: true, memoryModel: 'unified', name: 'NVIDIA GB10', processes: [] }, memory: CPU_HARDWARE.memory, disk: CPU_HARDWARE.disk };
    for (const hardware of [CPU_HARDWARE, unified]) {
        assert.equal(hardwareCardsHtml(hardware, null, { profile: 'dedicated', decision: null, limits: null }), hardwareCardsHtml(hardware, null));
        assert.equal(hardwareCardsHtml(hardware, null, {}), hardwareCardsHtml(hardware));
    }
    assert.match(hardwareCardsHtml(CPU_HARDWARE), /data-llm-card="gpu"[\s\S]*Not available/);
    // The run form's estimate says what stays free; the status card claims no GPU layers when there are none.
    const estimate = estimateHtml({ admission: { status: 'ok', estimate: { ramBytes: 1413 * CPU_MIB, floorBytes: 618099508 } } }, CPU_HARDWARE);
    assert.match(estimate, /RAM about 1,413 MiB of 3,049 MiB available, keeps 589 MiB free/);
    assert.doesNotMatch(estimateHtml({ admission: { status: 'ok', estimate: { ramBytes: 1413 * CPU_MIB } } }, CPU_HARDWARE), /keeps/);
    // Under a container memory limit the estimate is compared with what the limit leaves, not with the host's figure.
    const limited = { ...CPU_LIMITS, poolBytes: 4096 * CPU_MIB, availableBytes: 2048 * CPU_MIB };
    const capped = estimateHtml({ admission: { status: 'ok', estimate: { ramBytes: 1413 * CPU_MIB, floorBytes: 618099508 } } }, CPU_HARDWARE, { limits: limited });
    assert.match(capped, /RAM about 1,413 MiB of 2,048 MiB available, keeps 589 MiB free/);
    assert.doesNotMatch(capped, /3,049 MiB/);
    assert.match(capped, /<meter[^>]*max="2147483648"[^>]*aria-label="Estimated RAM"/);
    // Without limits, or with limits that carry no figure (unified memory's), the host's figure stays.
    const host = estimateHtml({ admission: { status: 'ok', estimate: { ramBytes: 1413 * CPU_MIB, floorBytes: 618099508 } } }, CPU_HARDWARE);
    assert.equal(estimateHtml({ admission: { status: 'ok', estimate: { ramBytes: 1413 * CPU_MIB, floorBytes: 618099508 } } }, CPU_HARDWARE, { limits: { floorBytes: 1, hostReserveBytes: 2 } }), host);
    assert.equal(estimateHtml({ admission: { status: 'ok', estimate: { ramBytes: 1413 * CPU_MIB, floorBytes: 618099508 } } }, CPU_HARDWARE, { limits: null }), host);
    // The presenter hands the overview's limits to the run form's estimate.
    const form = new dashboardModule.LocalLlmDashboard({ isConnected: true, querySelector: () => null, querySelectorAll: () => [] }, () => {});
    const region = { innerHTML: '' };
    form.runForm = { querySelector: (selector) => (selector === '[data-run-estimate]' ? region : null) };
    form.overview = { hardware: CPU_HARDWARE, limits: limited };
    form.renderEstimate({ admission: { status: 'ok', estimate: { ramBytes: 1413 * CPU_MIB, floorBytes: 618099508 } } });
    assert.match(region.innerHTML, /of 2,048 MiB available/);
    form.overview = { hardware: CPU_HARDWARE };
    form.renderEstimate({ admission: { status: 'ok', estimate: { ramBytes: 1413 * CPU_MIB, floorBytes: 618099508 } } });
    assert.match(region.innerHTML, /of 3,049 MiB available/);
    const running = (offloaded) => statusCardHtml({ phase: 'ready', deployment: { modelId: 'm', runnerId: 'llama.cpp', phase: 'ready' },
        runnerReport: { device: 'CPU (armv8.2_2)', offloaded, totalMiB: 634 } });
    assert.match(running({ layers: 0, of: 25 }), /device CPU \(armv8\.2_2\) · buffers 634 MiB/);
    assert.doesNotMatch(running({ layers: 0, of: 25 }), /layers on the GPU/);
    assert.match(running({ layers: 25, of: 25 }), /25\/25 layers on the GPU/);
    // The presenter hands the overview's profile, decision and limits to the cards.
    const presenter = new dashboardModule.LocalLlmDashboard({ isConnected: true, querySelector: () => null, querySelectorAll: () => [] }, () => {});
    presenter.hardware = { innerHTML: '' };
    presenter.overview = { profile: 'cpu', profileDecision: { cause: 'absent', reason: CPU_REASON }, limits: CPU_LIMITS, hardware: CPU_HARDWARE };
    presenter.renderHardware();
    assert.match(presenter.hardware.innerHTML, /data-llm-card="compute"[\s\S]*CPU · 4 cores[\s\S]*keeps 589 MiB free/);
    presenter.overview = { hardware: CPU_HARDWARE };
    presenter.renderHardware();
    assert.match(presenter.hardware.innerHTML, /data-llm-card="gpu"/, 'no profile: the cards of today');
});

test('a runner with no policy for the profile gets no column and no Install; while undecided every runner keeps its column', () => {
    const reason = 'Ollama needs an NVIDIA GPU in this release; on this machine models run on the CPU with the runners listed in the Runners tab.';
    const cpuRunners = [
        { id: 'llama.cpp', displayName: 'llama.cpp', supported: true, installed: true, paramSchema: { type: 'object' } },
        { id: 'ollama', displayName: 'Ollama', supported: true, installed: true, paramSchema: null, profileUnsupportedReason: reason },
        { id: 'vllm', displayName: 'vLLM', supported: true, installed: false, paramSchema: null, profileUnsupportedReason: 'vLLM needs an NVIDIA GPU in this release.',
            install: { version: '0.30.0', totalBytes: 4.2e9, licence: { name: 'Apache-2.0', url: 'https://x.example/L' }, installed: false, runnable: false, state: null } },
        { id: 'tabby', displayName: 'Tabby', supported: true, installed: true, paramSchema: null, profileUnsupportedReason: 'Tabby needs an NVIDIA GPU in this release.',
            install: { version: '1', totalBytes: 3e9, licence: { name: 'AGPL-3.0', url: 'https://x.example/T' }, installed: true, runnable: true, state: { phase: 'installed' } } },
    ];
    assert.deepEqual(tableRunners(cpuRunners).map((runner) => runner.id), ['llama.cpp']);
    // Undecided: no runner carries the reason, so every runner that can run keeps its column.
    const undecided = cpuRunners.map(({ profileUnsupportedReason: _reason, ...runner }) => runner);
    assert.deepEqual(tableRunners(undecided).map((runner) => runner.id), ['llama.cpp', 'ollama', 'vllm', 'tabby']);
    const html = runnersPanelHtml(cpuRunners);
    assert.match(html, /Ollama needs an NVIDIA GPU in this release/);
    assert.doesNotMatch(html, /data-local-action="installRunner/, 'no Install for a runner with no policy for the profile');
    assert.match(html, /data-local-action="uninstallRunner tabby"/, 'an installed one can still be uninstalled to free the disk');
    assert.match(runnersPanelHtml(undecided), /data-local-action="installRunner vllm"/);
    // The platform's reason still comes first for a runner the image lacks.
    const lacking = runnersPanelHtml([{ ...cpuRunners[1], supported: false, unsupportedReason: 'Ollama is not available on this platform: this image does not include it.' }]);
    assert.match(lacking, /not available on this platform/);
    assert.doesNotMatch(lacking, /needs an NVIDIA GPU/);
    // The Runners intro names the runners that run models on the CPU.
    // (Tabby and vLLM are installed on demand, so they are not "in this image".)
    // Install is offered only for a runner with a policy for the profile, and the intro promises it only then: here
    // both on-demand runners are refused on the CPU, so it says nothing can be installed.
    assert.equal(runnersIntroText(cpuRunners, { profile: 'cpu' }),
        'In this image: llama.cpp, Ollama. On this machine models run on the CPU, with: llama.cpp. No other runner can be installed while models run on the CPU.');
    assert.doesNotMatch(runnersIntroText(cpuRunners, { profile: 'cpu' }), /press Install/);
    // The arm64 image on the CPU: llama.cpp in the image, the rest absent or refused.
    const arm64 = [cpuRunners[0], { id: 'ik_llama.cpp', displayName: 'ik_llama.cpp', supported: false, unsupportedReason: 'not available on this platform', paramSchema: null }, cpuRunners[2]];
    assert.equal(runnersIntroText(arm64, { profile: 'cpu' }),
        'In this image: llama.cpp. On this machine models run on the CPU, with: llama.cpp. No other runner can be installed while models run on the CPU.');
    // A runner that is installable and has a policy keeps the promise.
    const { profileUnsupportedReason: _refused, ...installable } = cpuRunners[2];
    assert.match(runnersIntroText([cpuRunners[0], installable], { profile: 'cpu' }), /^In this image: llama\.cpp\. On this machine models run on the CPU, with: llama\.cpp\. Other runners are installed here, only when you press Install/);
    // With no on-demand runner at all, the SPEC's wording stands.
    assert.match(runnersIntroText([cpuRunners[0]], { profile: 'cpu' }), /No other runner can be installed on this image\.$/);
    // Other profiles carry no refusal, so their intro is what it was.
    const noReasons = cpuRunners.map(({ profileUnsupportedReason: _reason, ...runner }) => runner);
    assert.match(runnersIntroText(noReasons, { profile: 'dedicated' }), /Other runners are installed here, only when you press Install/);
    assert.doesNotMatch(runnersIntroText(noReasons, { profile: 'dedicated' }), /run models on the CPU/);
    // The run form says why such a runner cannot run.
    const presenter = new dashboardModule.LocalLlmDashboard({ isConnected: true, querySelector: () => null, querySelectorAll: () => [] }, () => {});
    const noteFor = (runner) => {
        const note = { textContent: '' };
        const find = (selector) => (selector === '[data-run-runner-note]' ? note : selector === '[data-run-basic]' || selector === '[data-run-params-advanced]' ? { innerHTML: '' } : null);
        presenter.overview = { runners: [runner], models: [{ id: 'm', runners: { [runner.id]: { download: { state: 'absent' }, admission: { status: 'incompatible' } } } }] };
        presenter.runForm = { querySelector: find, querySelectorAll: () => [] };
        presenter.runModelId = 'm';
        presenter.runnerSelect = { value: runner.id };
        return { note, run: () => presenter.renderRunFields() };
    };
    return (async () => {
        const probe = noteFor(cpuRunners[1]);
        await probe.run();
        assert.match(probe.note.textContent, /Ollama needs an NVIDIA GPU in this release/);
    })();
});

// ------------------------------------------------------------ the Add model lookup

const SHA = 'a'.repeat(64);
const GGUF_LOOKUP = {
    provider: 'huggingface', repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', revision: 'main', commit: 'c'.repeat(40), format: 'gguf',
    gated: false, license: 'apache-2.0', truncated: false, sizing: null,
    files: [
        { file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf', size: 491_400_032, sha256: SHA, gitOid: null, quantization: 'Q4_K_M', shards: null },
        { file: 'Q6_K/qwen-big-00001-of-00002.gguf', size: 1_100_000_000, sha256: null, gitOid: null, quantization: 'Q6_K',
            shards: [{ file: 'Q6_K/qwen-big-00001-of-00002.gguf', size: 600_000_000, sha256: SHA }, { file: 'Q6_K/qwen-big-00002-of-00002.gguf', size: 500_000_000, sha256: SHA }] },
        { file: 'mystery.gguf', size: 12_000, sha256: SHA, gitOid: null, quantization: null, shards: null },
    ],
};
const SIZING = { contextLength: 32768, architecture: 'dense', memory: { layers: 24, kvBytesPerToken: 12288 }, source: 'gguf-header', readBytes: 4096, notes: [] };
const tagsOf = (html) => [...html.matchAll(/<\/?([a-zA-Z][a-zA-Z0-9-]*)/g)].map((match) => match[1]);
// The attribute names of every tag; a quoted value is consumed whole, so text inside one is never read as a name.
const attributesOf = (html) => [...html.matchAll(/<[a-zA-Z][a-zA-Z0-9-]*((?:\s+[a-zA-Z_:][-a-zA-Z0-9_:.]*(?:="[^"]*")?)*)\s*>/g)]
    .flatMap((match) => [...match[1].matchAll(/\s+([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:="[^"]*")?/g)].map((attribute) => attribute[1]));

test('the lookup results list one row per GGUF file or split set, with size, quantization and gated status', () => {
    const html = lookupResultsHtml(GGUF_LOOKUP, { selectedFile: 'Q6_K/qwen-big-00001-of-00002.gguf' });
    const rows = html.split('<li class="local-llm-lookup-row">').slice(1);
    assert.equal(rows.length, 3, 'one row per file or split set, never one per shard');
    const rowText = (row) => row.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    assert.equal(rowText(rows[0]), 'qwen2.5-0.5b-instruct-q4_k_m.gguf Q4_K_M · 491 MB');
    assert.equal(rowText(rows[1]), 'Q6_K/qwen-big-00001-of-00002.gguf Q6_K · 1.1 GB · 2 parts');
    assert.equal(rowText(rows[2]), 'mystery.gguf quantization unknown · 12.0 KB');
    // One radio per row, named by its index, checked only for the selected file.
    assert.deepEqual([...html.matchAll(/data-local-action="pickLookupFile (\d+)"/g)].map((match) => match[1]), ['0', '1', '2']);
    assert.deepEqual([...html.matchAll(/<input type="radio" name="lookupFile" value="(\d)"[^>]*?( checked)?>/g)].map((match) => [match[1], Boolean(match[2])]),
        [['0', false], ['1', true], ['2', false]]);
    // The repository's status and licence, and the commit the files are at.
    assert.match(html, /Qwen\/Qwen2\.5-0\.5B-Instruct-GGUF at commit ccccccccccc/);
    assert.match(html, /<span class="status-badge">Not gated<\/span>/);
    assert.match(html, /<span class="status-badge">Licence apache-2\.0<\/span>/);
    assert.doesNotMatch(html, />gated<\/span>/, 'no per-row marker when the repository is open');
    assert.doesNotMatch(html, /HF_TOKEN/);
    // A gated repository says so on the summary and on every row, and says how to set the token.
    for (const [gated, label] of [['auto', 'Gated \\(access is granted automatically\\)'], ['manual', 'Gated \\(access needs approval\\)']]) {
        const locked = lookupResultsHtml({ ...GGUF_LOOKUP, gated });
        assert.match(locked, new RegExp(`<span class="status-badge">${label}</span>`));
        assert.equal((locked.match(/<span class="status-badge">gated<\/span>/g) || []).length, 3, gated);
        assert.match(locked, /ploinky var HF_TOKEN &lt;token&gt;/);
    }
    assert.match(lookupResultsHtml({ ...GGUF_LOOKUP, gated: null }), /Gated status unknown/);
    assert.match(lookupResultsHtml({ ...GGUF_LOOKUP, license: null }), /^(?![\s\S]*Licence )/);
    // The sizing read for the picked file, a cut-short list, an empty one, and a snapshot's summary.
    const sized = lookupResultsHtml({ ...GGUF_LOOKUP, sizing: SIZING, truncated: true });
    assert.match(sized, /Read from the GGUF header: 24 layers · context 32,768 tokens · KV cache 12,288 bytes per token \(f16\) · dense/);
    assert.match(sized, /The list is cut short/);
    assert.match(lookupResultsHtml({ ...GGUF_LOOKUP, sizing: { ...SIZING, memory: { layers: 24, kvBytesPerToken: null }, notes: ['x.ssm.state_size marks a hybrid model'] } }),
        /KV cache size not read, so the estimate uses its default[^<]*\(x\.ssm\.state_size marks a hybrid model\)/);
    assert.match(lookupResultsHtml({ ...GGUF_LOOKUP, files: [] }), /No GGUF file in this repository can be added/);
    assert.doesNotMatch(lookupResultsHtml({ ...GGUF_LOOKUP, files: [] }), /<input/);
    const snapshot = lookupResultsHtml({
        ...GGUF_LOOKUP, format: 'hf', sizing: { ...SIZING, source: 'config.json' },
        files: [{ file: 'config.json', size: 700, sha256: null, gitOid: '1'.repeat(40), quantization: null, shards: null },
            { file: 'model.safetensors', size: 1_000_000_000, sha256: SHA, gitOid: null, quantization: null, shards: null }],
    });
    assert.match(snapshot, /2 files, 1 of them safetensors · 1\.0 GB in all/);
    assert.match(snapshot, /Read from config\.json: 24 layers/);
    assert.doesNotMatch(snapshot, /<input|pickLookupFile/);
    assert.match(lookupResultsHtml({ ...GGUF_LOOKUP, format: 'exl3', sizing: null, files: [] }), /No config\.json to size the model from/);
    // A message and an error are plain text.
    assert.equal(lookupResultsHtml({ message: 'Looking up…' }).trim(), '<div class="settings-card-meta">Looking up…</div>');
    assert.equal(lookupResultsHtml({ error: 'not_found: nothing' }).trim(), '<div class="settings-status error">not_found: nothing</div>');
});

test('text from Hugging Face or from a model file cannot add markup or attributes to the lookup results', () => {
    const evil = '"><img src=x onerror=alert(1)><script>alert(2)</script>';
    const hostile = lookupResultsHtml({
        repo: `${evil}/r`, commit: evil, format: 'gguf', gated: evil, license: evil, truncated: true,
        files: [
            { file: `${evil}.gguf`, size: evil, quantization: evil, shards: [evil, evil] },
            { file: "x' onclick='alert(3).gguf", size: Number.NaN, quantization: '<b>Q4</b>', shards: null },
        ],
        sizing: { source: evil, architecture: evil, contextLength: evil, memory: { layers: evil, kvBytesPerToken: evil }, notes: [evil, 7, { toString: () => evil }] },
    }, { selectedFile: `${evil}.gguf` });
    const allowed = new Set(['div', 'span', 'ul', 'li', 'label', 'input']);
    assert.deepEqual([...new Set(tagsOf(hostile))].filter((tag) => !allowed.has(tag)), [], 'no element beyond the ones the view writes');
    const attributes = new Set(['class', 'role', 'aria-label', 'type', 'name', 'value', 'data-local-action', 'checked']);
    assert.deepEqual([...new Set(attributesOf(hostile))].filter((name) => !attributes.has(name)), [], 'no attribute beyond the ones the view writes');
    assert.doesNotMatch(hostile, /<img|<script|<b>/);
    // The only attribute values that come from data are numbers (the row index).
    assert.deepEqual([...hostile.matchAll(/data-local-action="([^"]*)"/g)].map((match) => match[1]), ['pickLookupFile 0', 'pickLookupFile 1']);
    // An error and a message are escaped too, and so are snapshot results.
    assert.equal(lookupResultsHtml({ error: evil }).trim(), `<div class="settings-status error">${evil.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')}</div>`);
    assert.doesNotMatch(lookupResultsHtml({ message: evil }), /<img|<script/);
    const snapshot = lookupResultsHtml({ repo: evil, commit: evil, format: 'hf', gated: evil, license: evil, files: [{ file: evil, size: evil }], sizing: { source: evil, notes: [evil] } });
    assert.deepEqual([...new Set(tagsOf(snapshot))].filter((tag) => !allowed.has(tag)), []);
    assert.doesNotMatch(snapshot, /<img|<script/);
    // A result with no data at all still renders, with no "undefined" or "null" in it.
    for (const empty of [undefined, {}, { files: null }, { files: [null, {}] }]) {
        const html = lookupResultsHtml(empty);
        assert.doesNotMatch(html, /undefined|null|NaN/, JSON.stringify(empty));
    }
});

test('the Add model form offers the four sources, a Look up button, a results region and a sizing label', () => {
    const html = readText(new URL('local-llm-dashboard.html', DASHBOARD));
    const form = /<form class="local-llm-add-form"[\s\S]*?<\/form>/.exec(html)[0];
    const fieldsOf = (kind) => [...form.matchAll(/data-source-field="([^"]*)"[\s\S]*?<\/(?:label|div)>/g)]
        .filter((match) => match[1].split(/\s+/).includes(kind))
        .map((match) => /name="([^"]+)"/.exec(match[0])?.[1] ?? (/data-llm-lookup-button/.test(match[0]) ? 'lookup' : '?'));
    assert.deepEqual(fieldsOf('huggingface'), ['repo', 'revision', 'lookup', 'file', 'quantization']);
    assert.deepEqual(fieldsOf('hf'), ['repo', 'revision', 'lookup']);
    assert.deepEqual(fieldsOf('exl3'), ['repo', 'revision', 'lookup']);
    assert.deepEqual(fieldsOf('ollama'), ['tag']);
    for (const token of [...form.matchAll(/data-source-field="([^"]*)"/g)].flatMap((match) => match[1].split(/\s+/))) {
        assert.ok(['huggingface', 'hf', 'exl3', 'ollama'].includes(token), token);
    }
    assert.match(form, /<button type="button" class="gray-button" data-llm-lookup-button data-local-action="lookupModel">Look up<\/button>/);
    assert.match(form, /<div class="local-llm-lookup" data-llm-lookup aria-live="polite"><\/div>/);
    assert.match(form, /<p class="settings-card-meta" data-llm-sizing-label hidden><\/p>/);
    // The presenter has the two actions and the table the page wires them to.
    const presenter = readText(new URL('local-llm-dashboard.js', DASHBOARD));
    for (const action of ['lookupModel', 'pickLookupFile']) assert.match(presenter, new RegExp(`\\n    (?:async )?${action}\\(`));
    assert.equal(typeof dashboardModule.LocalLlmDashboard.prototype.lookupModel, 'function');
    assert.equal(typeof dashboardModule.LocalLlmDashboard.prototype.pickLookupFile, 'function');
});

// An Add model form made of plain objects: each control keeps a value, the lookup region and label keep their text.
function addFormHarness(t, callTool, { kind = 'huggingface', values = {} } = {}) {
    const h = pollingHarness(t, callTool);
    const p = h.presenter;
    const names = ['id', 'displayName', 'license', 'repo', 'revision', 'file', 'quantization', 'tag', 'contextLength', 'layers', 'kvBytesPerToken'];
    const controls = Object.fromEntries(names.map((name) => [name, { name, value: values[name] ?? '' }]));
    const attributes = { 'data-options': encodeURIComponent(JSON.stringify([{ value: 'dense', label: 'Dense' }, { value: 'moe', label: 'Mixture of experts' }])) };
    const architecture = {
        value: 'dense', presenterReadyPromise: Promise.resolve(), getAttribute: (name) => attributes[name], setAttribute: (name, value) => { attributes[name] = value; },
    };
    const region = { innerHTML: '' };
    const label = { textContent: '', hidden: true };
    const button = { disabled: false };
    p.addForm = { elements: controls, reportValidity: () => true, reset() { for (const control of Object.values(controls)) control.value = ''; } };
    p.lookupRegion = region;
    p.lookupButton = button;
    p.sizingLabel = label;
    p.sourceKind = { value: kind };
    p.element = { querySelector: (selector) => (selector === '#localLlmArchitecture' ? architecture : null), querySelectorAll: () => [] };
    p.readAddForm = () => Object.fromEntries(Object.entries(controls).map(([name, control]) => [name, control.value]));
    p.loadOverview = async () => {};
    p.selectModel = async () => {};
    return { ...h, p, controls, region, label, button, architecture };
}
const asTool = (value) => ({ content: [{ type: 'text', text: JSON.stringify(value) }] });

test('Look up lists a repository, picking a file fills the form from its header, and Add records the sizing source', async (t) => {
    const calls = [];
    let busyDuring = null;
    const h = addFormHarness(t, async (tool, args) => {
        calls.push([tool, args]);
        if (tool === 'local_llm_model_lookup') {
            busyDuring = h.p.busy;
            return asTool(args.file ? { ...GGUF_LOOKUP, sizing: SIZING } : GGUF_LOOKUP);
        }
        return asTool({ model: { id: 'x' } });
    });
    const { p, controls } = h;
    controls.repo.value = ' Qwen/Qwen2.5-0.5B-Instruct-GGUF ';
    await p.lookupModel();
    assert.deepEqual(calls, [['local_llm_model_lookup', { repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', format: 'gguf' }]], 'the revision is left out when empty');
    assert.equal(busyDuring, false, 'a lookup never makes the dashboard busy, so Stop and Cancel stay usable');
    assert.equal((h.region.innerHTML.match(/<li class="local-llm-lookup-row">/g) || []).length, 3);
    assert.equal(controls.file.value, '', 'nothing is picked yet');
    assert.equal(h.label.hidden, true);
    assert.equal(h.button.disabled, false);
    // Picking the first row reads that file's header and fills the form.
    await p.pickLookupFile(null, '0');
    assert.deepEqual(calls[1], ['local_llm_model_lookup', { repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', format: 'gguf', file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf' }]);
    assert.deepEqual(
        Object.fromEntries(['id', 'displayName', 'license', 'file', 'quantization', 'contextLength', 'layers', 'kvBytesPerToken'].map((name) => [name, controls[name].value])),
        {
            id: 'qwen2.5-0.5b-instruct-gguf-q4_k_m', displayName: 'Qwen2.5-0.5B-Instruct-GGUF Q4_K_M', license: 'apache-2.0', file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf',
            quantization: 'Q4_K_M', contextLength: '32768', layers: '24', kvBytesPerToken: '12288',
        },
    );
    assert.equal(h.architecture.value, 'dense');
    assert.equal(h.label.textContent, 'Sizing read from the GGUF header; checked again after download.');
    assert.equal(h.label.hidden, false);
    assert.match(h.region.innerHTML, /value="0"[^>]*checked/, 'the picked row stays selected');
    assert.match(h.region.innerHTML, /Read from the GGUF header: 24 layers/);
    // An id, name or licence the admin already typed is kept.
    controls.id.value = 'my-own-id';
    controls.displayName.value = 'Mine';
    controls.license.value = 'Custom';
    await p.pickLookupFile(null, '2');
    assert.deepEqual([controls.id.value, controls.displayName.value, controls.license.value, controls.file.value, controls.quantization.value],
        ['my-own-id', 'Mine', 'Custom', 'mystery.gguf', '']);
    await p.pickLookupFile(null, '0');
    // Add sends the looked-up sizing as such.
    await p.submitAddModel({ preventDefault() {} });
    const add = calls.filter(([tool]) => tool === 'local_llm_model_add').at(-1)[1].model;
    assert.equal(add.sizingSource, 'gguf-header');
    // Add pins the commit the lookup read (the form's revision was empty, which would have meant the branch as it is by then).
    assert.deepEqual([add.contextLength, add.memory, add.sources.gguf.file, add.sources.gguf.revision],
        [32768, { layers: 24, kvBytesPerToken: 12288 }, 'qwen2.5-0.5b-instruct-q4_k_m.gguf', GGUF_LOOKUP.commit]);
    assert.equal(p.lookupSizing, null, 'the form starts again after Add');
    assert.equal(h.region.innerHTML, '');
    assert.equal(h.label.hidden, true);

    // A value changed by hand after the lookup is recorded as manual; so is a repository changed after it.
    const afterLookup = async (change) => {
        const sent = [];
        const form = addFormHarness(t, async (tool, args) => {
            sent.push([tool, args]);
            return asTool(tool === 'local_llm_model_lookup' ? { ...GGUF_LOOKUP, sizing: SIZING } : { model: {} });
        });
        form.controls.repo.value = GGUF_LOOKUP.repo;
        await form.p.lookupModel();
        await form.p.pickLookupFile(null, '0');
        change(form.controls);
        await form.p.submitAddModel({ preventDefault() {} });
        return sent.filter(([tool]) => tool === 'local_llm_model_add').at(-1)[1].model;
    };
    const edited = await afterLookup((controls) => { controls.layers.value = '28'; });
    assert.deepEqual([edited.sizingSource, edited.memory.layers], ['manual', 28]);
    assert.equal((await afterLookup((controls) => { controls.repo.value = 'someone/else'; })).sizingSource, 'manual');
    assert.equal((await afterLookup((controls) => { controls.file.value = 'other.gguf'; })).sizingSource, 'manual');
    assert.equal((await afterLookup(() => {})).sizingSource, 'gguf-header', 'control: untouched');
    // Typed from scratch with no lookup: today's entry, with no sizingSource.
    const typed = [];
    const scratch = addFormHarness(t, async (tool, args) => { typed.push([tool, args]); return asTool({ model: {} }); }, { values: { id: 'typed', repo: 'a/b', file: 'm.gguf', layers: '12' } });
    await scratch.p.submitAddModel({ preventDefault() {} });
    assert.equal('sizingSource' in typed.at(-1)[1].model, false);
    assert.deepEqual(typed.at(-1)[1].model.memory, { layers: 12 });
});

test('Look up for a safetensors or EXL3 repository sizes it from config.json; an empty repository, a failure and a stale answer are handled', async (t) => {
    const snapshot = {
        ...GGUF_LOOKUP, format: 'hf', license: 'mit', sizing: { ...SIZING, source: 'config.json' },
        files: [{ file: 'config.json', size: 700, sha256: null, gitOid: '1'.repeat(40), quantization: null, shards: null }, { file: 'model.safetensors', size: 1e9, sha256: SHA, gitOid: null, quantization: null, shards: null }],
    };
    for (const kind of ['hf', 'exl3']) {
        const calls = [];
        const h = addFormHarness(t, async (tool, args) => { calls.push([tool, args]); return asTool(tool === 'local_llm_model_lookup' ? { ...snapshot, format: kind, repo: args.repo, revision: args.revision } : { model: {} }); }, { kind });
        h.controls.repo.value = 'owner/repo';
        h.controls.revision.value = 'dev';
        await h.p.lookupModel();
        assert.deepEqual(calls, [['local_llm_model_lookup', { repo: 'owner/repo', format: kind, revision: 'dev' }]]);
        // The sizing arrives with the list: no file to pick.
        assert.deepEqual([h.controls.contextLength.value, h.controls.layers.value, h.controls.kvBytesPerToken.value, h.controls.license.value], ['32768', '24', '12288', 'mit']);
        assert.equal(h.label.textContent, 'Sizing read from config.json, checked against its pinned digest.');
        assert.equal(h.controls.file.value, '');
        await h.p.submitAddModel({ preventDefault() {} });
        const { model } = calls.at(-1)[1];
        assert.deepEqual(Object.keys(model.sources), [kind]);
        assert.equal(model.sources[kind].type, 'hf-snapshot');
        assert.deepEqual([model.sizingSource, model.contextLength, model.memory], ['config.json', 32768, { layers: 24, kvBytesPerToken: 12288 }]);
        assert.equal(model.sources[kind].revision, GGUF_LOOKUP.commit, 'a snapshot is pinned at the commit whose config.json was read');
    }
    // No config.json in the repository: the label says the defaults apply, and no sizing source is recorded.
    const bare = addFormHarness(t, async () => asTool({ ...snapshot, sizing: null }), { kind: 'hf' });
    bare.controls.repo.value = 'owner/repo';
    await bare.p.lookupModel();
    assert.match(bare.label.textContent, /No config\.json was found/);
    // Nothing was sized, so no sizing source; the commit the lookup read is still remembered for Add.
    assert.deepEqual([bare.p.lookupSizing.source, bare.p.lookupSizing.commit, bare.p.lookupSizing.values], [null, GGUF_LOOKUP.commit, { contextLength: '', layers: '', kvBytesPerToken: '' }]);
    // An Ollama tag has no lookup, and an empty repository is asked for before any request.
    const calls = [];
    const none = addFormHarness(t, async (tool, args) => { calls.push(tool); return asTool({}); }, { kind: 'ollama' });
    await none.p.lookupModel();
    assert.deepEqual(calls, []);
    const empty = addFormHarness(t, async (tool) => { calls.push(tool); return asTool({}); });
    await empty.p.lookupModel();
    assert.deepEqual(calls, []);
    assert.match(empty.region.innerHTML, /Enter a repository such as owner\/name first\./);
    // A failure is shown as text, leaves the form alone, and the button works again.
    const failing = addFormHarness(t, async () => { throw new Error('access_denied: <img src=x onerror=1> set HF_TOKEN'); });
    failing.controls.repo.value = 'owner/private';
    await failing.p.lookupModel();
    assert.match(failing.region.innerHTML, /^<div class="settings-status error">access_denied: &lt;img src=x onerror=1&gt; set HF_TOKEN<\/div>$/);
    assert.equal(failing.button.disabled, false);
    assert.equal(failing.p.lookupPending, false);
    // A failure while a file's header is read keeps the list, and reports through the status line.
    let phase = 0;
    const partial = addFormHarness(t, async (tool, args) => { phase += 1; if (args.file) throw new Error('invalid_gguf: bad header'); return asTool(GGUF_LOOKUP); });
    partial.controls.repo.value = 'owner/repo';
    await partial.p.lookupModel();
    await partial.p.pickLookupFile(null, '0');
    assert.equal(phase, 2);
    assert.equal((partial.region.innerHTML.match(/<li class="local-llm-lookup-row">/g) || []).length, 3, 'the list is still there');
    assert.equal(partial.controls.file.value, '', 'nothing was filled in');
    assert.equal(partial.label.hidden, true);
    // Two lookups in flight: the answer of the older one is ignored.
    const waiting = [];
    const racing = addFormHarness(t, (tool, args) => new Promise((resolve) => waiting.push({ args, resolve })));
    racing.controls.repo.value = 'owner/first';
    const first = racing.p.lookupModel();
    racing.p.resetLookup();
    racing.controls.repo.value = 'owner/second';
    const second = racing.p.lookupModel();
    waiting[0].resolve(asTool({ ...GGUF_LOOKUP, repo: 'owner/first' }));
    await first;
    assert.equal(racing.region.innerHTML.includes('owner/first'), false, 'a superseded answer is dropped');
    waiting[1].resolve(asTool({ ...GGUF_LOOKUP, repo: 'owner/second' }));
    await second;
    assert.equal(racing.region.innerHTML.includes('owner/second'), true);
    // While one lookup is pending another is not started.
    const pending = [];
    const once = addFormHarness(t, (tool, args) => new Promise((resolve) => pending.push(resolve)));
    once.controls.repo.value = 'owner/repo';
    const running = once.p.lookupModel();
    await once.p.lookupModel();
    assert.equal(pending.length, 1);
    assert.equal(once.button.disabled, true, 'the button is disabled while it waits');
    pending[0](asTool(GGUF_LOOKUP));
    await running;
    assert.equal(once.button.disabled, false);
});

test('Add pins the commit the lookup read, while the form still names what it read, and otherwise the revision as typed', async (t) => {
    const COMMIT_A = 'a'.repeat(40);
    const lookupAt = (overrides = {}) => async (tool, args) => asTool(tool === 'local_llm_model_lookup'
        ? { ...GGUF_LOOKUP, commit: COMMIT_A, revision: args.revision || 'main', ...(args.file ? { sizing: SIZING } : {}), ...overrides } : { model: {} });
    const addAfter = async (change, { revision = '', kind = 'huggingface', overrides } = {}) => {
        const sent = [];
        const inner = lookupAt(overrides);
        const form = addFormHarness(t, async (tool, args) => { sent.push([tool, args]); return inner(tool, args); }, { kind });
        form.controls.repo.value = GGUF_LOOKUP.repo;
        form.controls.revision.value = revision;
        await form.p.lookupModel();
        if (kind === 'huggingface') await form.p.pickLookupFile(null, '0');
        change(form.controls);
        await form.p.submitAddModel({ preventDefault() {} });
        const entry = sent.filter(([tool]) => tool === 'local_llm_model_add').at(-1)[1].model;
        return { entry, source: entry.sources.gguf ?? entry.sources.hf ?? entry.sources.exl3, form };
    };
    // The branch is named (or left empty): Add gets the commit, so it pins what was sized even if the branch moved on.
    for (const revision of ['', 'main']) assert.equal((await addAfter(() => {}, { revision })).source.revision, COMMIT_A, `revision '${revision}'`);
    assert.equal((await addAfter(() => {}, { revision: 'dev' })).source.revision, COMMIT_A);
    assert.equal((await addAfter(() => {}, { kind: 'hf' })).source.revision, COMMIT_A, 'a snapshot too');
    assert.equal((await addAfter(() => {}, { kind: 'exl3' })).source.revision, COMMIT_A);
    // Values changed by hand are manual, but the files are the ones that were looked up, so the commit stays.
    const edited = await addAfter((controls) => { controls.layers.value = '28'; });
    assert.deepEqual([edited.entry.sizingSource, edited.source.revision], ['manual', COMMIT_A]);
    // Another repository, revision or file is not what was looked up: the revision is pinned as typed, as it always was.
    const other = await addAfter((controls) => { controls.repo.value = 'someone/else'; }, { revision: 'dev' });
    assert.deepEqual([other.entry.sizingSource, other.source.revision], ['manual', 'dev']);
    const moved = await addAfter((controls) => { controls.revision.value = 'v2'; });
    assert.equal(moved.source.revision, 'v2');
    const otherFile = await addAfter((controls) => { controls.file.value = 'other.gguf'; });
    assert.equal(otherFile.source.revision, 'main');
    // A lookup that named no commit (an older agent) leaves the typed revision alone; an Ollama tag has no revision.
    assert.equal((await addAfter(() => {}, { revision: 'dev', overrides: { commit: undefined } })).source.revision, 'dev');
    assert.equal((await addAfter(() => {}, { revision: 'dev', overrides: { commit: 'not-a-commit' } })).source.revision, 'dev');
    const tag = [];
    const ollama = addFormHarness(t, async (tool, args) => { tag.push([tool, args]); return asTool({ model: {} }); }, { kind: 'ollama', values: { id: 'tagged', tag: 'qwen2.5:0.5b' } });
    await ollama.p.submitAddModel({ preventDefault() {} });
    assert.deepEqual(tag.at(-1)[1].model.sources, { ollama: { type: 'ollama', tag: 'qwen2.5:0.5b' } });
});

test('a failed file pick puts the selection back on the file in the form, a long suggested name fits the form, and a listing with no usable header sizing leaves the form alone', async (t) => {
    const calls = [];
    let failPick = false;
    const h = addFormHarness(t, async (tool, args) => {
        calls.push([tool, args]);
        if (args.file && failPick) throw new Error('invalid_gguf: bad header');
        return asTool(args.file ? { ...GGUF_LOOKUP, sizing: SIZING } : GGUF_LOOKUP);
    });
    const { p, controls } = h;
    const checkedRows = () => [...h.region.innerHTML.matchAll(/<input type="radio" name="lookupFile" value="(\d)"[^>]*? checked>/g)].map((match) => match[1]);
    controls.repo.value = GGUF_LOOKUP.repo;
    await p.lookupModel();
    await p.pickLookupFile(null, '0');
    assert.deepEqual(checkedRows(), ['0']);
    assert.equal(controls.file.value, 'qwen2.5-0.5b-instruct-q4_k_m.gguf');
    // The next pick fails: the form keeps file 0, so the list says file 0 and not the row that was tried.
    failPick = true;
    await p.pickLookupFile(null, '2');
    assert.deepEqual(checkedRows(), ['0'], 'the selection is the file in the form');
    assert.equal(controls.file.value, 'qwen2.5-0.5b-instruct-q4_k_m.gguf');
    assert.equal(h.label.hidden, true);
    // With no file in the form the selection is cleared; with a file the list does not hold, none is selected either.
    controls.file.value = '';
    await p.pickLookupFile(null, '1');
    assert.deepEqual(checkedRows(), []);
    controls.file.value = 'typed-by-hand.gguf';
    await p.pickLookupFile(null, '1');
    assert.deepEqual(checkedRows(), []);
    // The display name suggested from a long repository name and quantization is cut to the form's 120 characters.
    failPick = false;
    const long = addFormHarness(t, async (tool, args) => asTool({ ...GGUF_LOOKUP, repo: `owner/${'n'.repeat(150)}`, ...(args.file ? { sizing: SIZING } : {}) }));
    long.controls.repo.value = 'owner/x';
    await long.p.lookupModel();
    await long.p.pickLookupFile(null, '0');
    assert.equal(long.controls.displayName.value.length, 120);
    assert.ok(long.controls.displayName.value.startsWith('nnnn'));
    const fits = addFormHarness(t, async (tool, args) => asTool({ ...GGUF_LOOKUP, repo: `owner/${'n'.repeat(106)}`, ...(args.file ? { sizing: SIZING } : {}) }));
    fits.controls.repo.value = 'owner/x';
    await fits.p.lookupModel();
    await fits.p.pickLookupFile(null, '0');
    assert.equal(fits.controls.displayName.value, `${'n'.repeat(106)} Q4_K_M`, 'a name within the limit is untouched');
    assert.ok(fits.controls.id.value.length <= 64);
    // A header this agent could not size (no usable architecture): nothing is read, so typed values and the architecture stay,
    // values an earlier pick put there are cleared, and the label says the estimate will use its defaults.
    const unsized = { ...SIZING, contextLength: null, architecture: 'dense', memory: { layers: null, kvBytesPerToken: null }, notes: ['general.architecture is missing'] };
    let next = SIZING;
    const pick = addFormHarness(t, async (tool, args) => asTool(args.file ? { ...GGUF_LOOKUP, sizing: next } : GGUF_LOOKUP));
    pick.controls.repo.value = GGUF_LOOKUP.repo;
    await pick.p.lookupModel();
    await pick.p.pickLookupFile(null, '0');
    assert.deepEqual([pick.controls.contextLength.value, pick.controls.layers.value, pick.controls.kvBytesPerToken.value], ['32768', '24', '12288']);
    next = unsized;
    await pick.p.pickLookupFile(null, '2');
    assert.deepEqual([pick.controls.contextLength.value, pick.controls.layers.value, pick.controls.kvBytesPerToken.value], ['', '', ''], 'the first pick\'s values are not kept for a file that has none');
    assert.match(pick.label.textContent, /^Nothing could be read from the GGUF header, so the memory estimate uses its defaults\. general\.architecture is missing\.$/);
    pick.controls.layers.value = '30';
    await pick.p.pickLookupFile(null, '1');
    assert.equal(pick.controls.layers.value, '30', 'a value the admin typed is not erased by a header that has none');
    assert.equal(pick.controls.contextLength.value, '');
    pick.architecture.value = 'moe';
    await pick.p.pickLookupFile(null, '2');
    assert.equal(pick.architecture.value, 'moe', 'an unsizable header does not change the architecture on the form');
    assert.equal(pick.p.lookupSizing.architecture, 'moe');
    assert.equal(pick.p.lookupSizing.commit, GGUF_LOOKUP.commit);
});

test('bidirectional control characters in a licence never reach the page or the form', async (t) => {
    const evil = 'MIT\u202Egpl\u2066x\u2069\u202C';
    assert.equal(stripBidi(evil), 'MITgplx');
    assert.equal(stripBidi('\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069'), '');
    assert.equal(stripBidi('plain apache-2.0 é 日本'), 'plain apache-2.0 é 日本');
    assert.equal(stripBidi(undefined), '');
    const html = lookupResultsHtml({ ...GGUF_LOOKUP, license: evil });
    assert.match(html, /<span class="status-badge">Licence MITgplx<\/span>/);
    assert.doesNotMatch(html, /[\u202A-\u202E\u2066-\u2069]/);
    const h = addFormHarness(t, async (tool, args) => asTool({ ...GGUF_LOOKUP, license: evil, ...(args.file ? { sizing: SIZING } : {}) }));
    h.controls.repo.value = GGUF_LOOKUP.repo;
    await h.p.lookupModel();
    await h.p.pickLookupFile(null, '0');
    assert.equal(h.controls.license.value, 'MITgplx');
    assert.doesNotMatch(h.region.innerHTML, /[\u202A-\u202E\u2066-\u2069]/);
    // A licence of nothing but controls leaves the field empty.
    const empty = addFormHarness(t, async (tool, args) => asTool({ ...GGUF_LOOKUP, license: '\u202E\u2069', ...(args.file ? { sizing: SIZING } : {}) }));
    empty.controls.repo.value = GGUF_LOOKUP.repo;
    await empty.p.lookupModel();
    await empty.p.pickLookupFile(null, '0');
    assert.equal(empty.controls.license.value, '');
});

test('bidirectional control characters in a sizing note never reach the page', () => {
    // A note can repeat a key name from the model's own header or config.json.
    const noted = { ...SIZING, notes: ['qwen2‮.ssm.conv_kernel⁦ marks a recurrent layer⁩'] };
    for (const result of [{ ...GGUF_LOOKUP, sizing: noted }, { ...GGUF_LOOKUP, format: 'hf', files: [], sizing: { ...noted, source: 'config.json' } }]) {
        const html = lookupResultsHtml(result);
        assert.match(html, /qwen2\.ssm\.conv_kernel marks a recurrent layer/);
        assert.doesNotMatch(html, /[‪-‮⁦-⁩]/);
    }
});

test('the source fields follow the four kinds, and the lookup is forgotten when the source changes', () => {
    const fields = [
        { dataset: { sourceField: 'huggingface hf exl3' }, hidden: false, querySelector: () => ({ name: 'repo', required: false }) },
        { dataset: { sourceField: 'huggingface' }, hidden: false, querySelector: () => ({ name: 'file', required: false }) },
        { dataset: { sourceField: 'ollama' }, hidden: false, querySelector: () => ({ name: 'tag', required: false }) },
    ];
    const inputs = fields.map((field) => field.querySelector());
    fields.forEach((field, index) => { field.querySelector = () => inputs[index]; });
    const presenter = new dashboardModule.LocalLlmDashboard({ isConnected: true, querySelector: () => null, querySelectorAll: () => fields }, () => {});
    const state = () => fields.map((field, index) => [field.hidden, inputs[index].required]);
    for (const [kind, expected] of [
        ['huggingface', [[false, true], [false, true], [true, false]]],
        ['hf', [[false, true], [true, false], [true, false]]],
        ['exl3', [[false, true], [true, false], [true, false]]],
        ['ollama', [[true, false], [true, false], [false, true]]],
        ['anything else', [[false, true], [false, true], [true, false]]],
    ]) {
        presenter.sourceKind = { value: kind };
        presenter.updateSourceFields();
        assert.deepEqual(state(), expected, kind);
    }
    presenter.lookup = { args: {}, result: {} };
    presenter.lookupSizing = { source: 'gguf-header' };
    presenter.lookupRegion = { innerHTML: 'x' };
    presenter.sizingLabel = { textContent: 'y', hidden: false };
    presenter.resetLookup();
    assert.deepEqual([presenter.lookup, presenter.lookupRegion.innerHTML, presenter.sizingLabel.textContent, presenter.sizingLabel.hidden], [null, '', '', true]);
    assert.deepEqual(presenter.lookupSizing, { source: 'gguf-header' }, 'what was filled in is kept for the sizing source, unless the form is cancelled or added');
    presenter.resetLookup({ forget: true });
    assert.equal(presenter.lookupSizing, null);
});

