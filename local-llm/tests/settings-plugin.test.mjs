import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import * as presenterModule from '../IDE-plugins/local-llm-settings/local-llm-settings.js';
import {
    confirmMessage,
    fieldsFromSchema,
    mergeLogs,
    modelEntryFromForm,
    newRequestId,
    paramsFromForm,
    parseToolResult,
    runnerOptions,
    shouldPoll,
} from '../IDE-plugins/local-llm-settings/local-llm-settings-model.js';
import { RUNNERS } from '../src/runners/index.mjs';

const ROOT = new URL('..', import.meta.url);
const PLUGIN = new URL('IDE-plugins/local-llm-settings/', ROOT);
const readText = (name) => fs.readFileSync(new URL(name, PLUGIN), 'utf8');
const readJson = (url) => JSON.parse(fs.readFileSync(url, 'utf8'));

test('the manifest contributes one admin-only workspace settings entry that matches the plugin', () => {
    const manifest = readJson(new URL('manifest.json', ROOT));
    assert.deepEqual(manifest.ideSettings, [{
        key: 'local-llm-settings',
        label: 'Local LLMs',
        scope: 'workspace',
        pluginKey: 'local-llm/local-llm-settings',
        settingsComponent: 'local-llm-settings',
        adminOnly: true,
    }]);
    const config = readJson(new URL('config.json', PLUGIN));
    assert.equal(config.id, 'local-llm-settings');
    assert.equal(config.component, 'local-llm-settings');
    assert.equal(config.settings, 'local-llm-settings');
    assert.equal(config.presenter, 'LocalLlmSettings');
    assert.equal(config.type, 'global');
    assert.deepEqual(config.location, []);
    for (const extension of ['html', 'css', 'js']) {
        assert.ok(fs.existsSync(new URL(`local-llm-settings.${extension}`, PLUGIN)), extension);
    }
    assert.ok(fs.existsSync(new URL('icon.svg', PLUGIN)));
});

test('the presenter module exports only LocalLlmSettings, which the settings loader picks', () => {
    assert.deepEqual(Object.keys(presenterModule), ['LocalLlmSettings']);
    const first = Object.keys(presenterModule).find((key) => typeof presenterModule[key] === 'function');
    assert.equal(first, 'LocalLlmSettings');
});

// The settings entry is now a launcher: it opens the Local LLMs dashboard in
// Explorer's full-screen panel and closes itself.
test('the settings launcher keeps Explorer settings style rules and has no dialog of its own to style', () => {
    const presenter = readText('local-llm-settings.js');
    const template = readText('local-llm-settings.html');
    const styles = readText('local-llm-settings.css');
    assert.match(presenter, /constructor\(element, invalidate[\s\S]*?this\.invalidate\(\);/);
    assert.match(template, /class="close"[\s\S]*?\/explorer\/shared\/assets\/icons\/x-mark\.svg/);
    assert.match(template, /data-local-action="openDashboard"/);
    assert.doesNotMatch(template, /\/explorer\/assets\/icons\//, 'icons that moved to /explorer/shared/assets/icons answer 404');
    assert.doesNotMatch(template, /\$\$/, 'WebSkel treats $$name as a template variable');
    assert.doesNotMatch(styles, /#[0-9a-f]{3,8}\b/i);
    assert.doesNotMatch(styles, /(?:color|background|border(?:-color|-radius)?|box-shadow):/);
    assert.doesNotMatch(styles, /\/\*/, 'WebSkel mis-scopes the rule after a CSS comment');
    for (const rule of styles.replace(/@media[^{]*\{/g, '').split('}')) {
        const selector = rule.split('{')[0].trim();
        if (!selector) continue;
        for (const part of selector.split(',')) {
            assert.match(part.trim(), /^(?:dialog\.modal\.local-llm-settings-dialog|local-llm-settings\b)/, part.trim());
        }
    }
});

function launcherHarness({ openExpandedModal, registered = false, registerRuntimeComponent } = {}) {
    const calls = [];
    const ui = {
        ...(openExpandedModal === null ? {} : { openExpandedModal: openExpandedModal || ((descriptor) => { calls.push(['open', descriptor]); }) }),
        closeModal: (element) => calls.push(['close', element === host]),
    };
    const statusLine = { textContent: '', classList: { toggle: (name, on) => { if (on) calls.push(['status', name]); } } };
    const host = { querySelector: () => statusLine };
    const saved = { customElements: globalThis.customElements, fetch: globalThis.fetch };
    globalThis.customElements = { get: () => (registered ? class {} : undefined) };
    globalThis.fetch = async (url) => { calls.push(['fetch', String(url).split('/').slice(-1)[0]]); return { ok: true, text: async () => 'x' }; };
    const presenter = new presenterModule.LocalLlmSettings(host, () => {}, {
        ui: () => ui,
        webSkel: () => ({ name: 'webSkel' }),
        baseUrl: new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard', import.meta.url).href,
        loadRegistration: async () => ({ registerRuntimeComponent: registerRuntimeComponent || (async (webSkel, definition) => {
            calls.push(['register', definition.name, definition.presenterClassName, typeof definition.presenterModule?.LocalLlmDashboard]);
        }) }),
    });
    presenter.statusLine = statusLine;
    const restore = () => Object.assign(globalThis, saved);
    return { presenter, calls, statusLine, restore };
}

test('the settings launcher registers the dashboard when needed, opens it full screen and closes itself', async (t) => {
    const fresh = launcherHarness();
    t.after(fresh.restore);
    assert.equal(await fresh.presenter.openDashboard(), true);
    assert.deepEqual(fresh.calls.filter(([kind]) => kind === 'register'), [['register', 'local-llm-dashboard', 'LocalLlmDashboard', 'function']]);
    assert.deepEqual(fresh.calls.filter(([kind]) => kind === 'fetch').map(([, name]) => name).sort(), ['local-llm-dashboard.css', 'local-llm-dashboard.html']);
    assert.deepEqual(fresh.calls.find(([kind]) => kind === 'open')[1], { mode: 'component', component: 'local-llm-dashboard', title: 'Local LLMs' });
    assert.deepEqual(fresh.calls.at(-1), ['close', true]);
    fresh.restore();

    const already = launcherHarness({ registered: true });
    t.after(already.restore);
    assert.equal(await already.presenter.openDashboard(), true);
    assert.equal(already.calls.some(([kind]) => kind === 'register' || kind === 'fetch'), false, 'no second registration');
    already.restore();

    const old = launcherHarness({ openExpandedModal: null });
    t.after(old.restore);
    assert.equal(await old.presenter.openDashboard(), false);
    assert.match(old.statusLine.textContent, /toolbar/);
    assert.equal(old.calls.some(([kind]) => kind === 'close'), false, 'stays open to show why');
    old.restore();

    const failing = launcherHarness({ registerRuntimeComponent: async () => { throw new Error('boom'); } });
    t.after(failing.restore);
    assert.equal(await failing.presenter.openDashboard(), false);
    assert.match(failing.statusLine.textContent, /could not be loaded: boom/);
    failing.restore();
});

test('the parameter form is generated from the runner schema and read back as typed values', () => {
    const schema = RUNNERS['llama.cpp'].paramSchema;
    const fields = fieldsFromSchema(schema, { ctxSize: 16384, nCpuMoe: 17, threads: null, chatTemplateKwargs: { reasoning_effort: 'low' } });
    const byName = Object.fromEntries(fields.map((field) => [field.name, field]));
    assert.equal(byName.ctxSize.kind, 'number');
    assert.equal(byName.ctxSize.value, 16384);
    assert.equal(byName.flashAttn.kind, 'enum');
    assert.deepEqual(byName.flashAttn.options.map((option) => option.value), ['auto', 'on', 'off']);
    assert.equal(byName.threads.nullable, true);
    const { params, errors } = paramsFromForm(fields, { ctxSize: '8192', nCpuMoe: '17', threads: '', flashAttn: 'on', nGpuLayers: '99' });
    assert.deepEqual(errors, []);
    assert.equal(params.ctxSize, 8192);
    assert.equal(params.threads, null);
    assert.equal(params.flashAttn, 'on');
    if (byName.chatTemplateKwargs) assert.deepEqual(params.chatTemplateKwargs, { reasoning_effort: 'low' });
    assert.match(paramsFromForm(fields, { ctxSize: '16384; rm -rf /' }).errors[0], /whole number/);
    assert.match(paramsFromForm(fields, { ctxSize: '100' }).errors[0], /between 512 and 131072/);

    const ollama = fieldsFromSchema(RUNNERS.ollama.paramSchema, {});
    const flash = ollama.find((field) => field.name === 'flashAttention');
    assert.equal(flash.kind, 'boolean');
    assert.deepEqual(flash.options.map((option) => option.value), ['', 'true', 'false']);
    assert.equal(paramsFromForm(ollama, { flashAttention: '' }).params.flashAttention, null);
    assert.equal(paramsFromForm(ollama, { flashAttention: 'false' }).params.flashAttention, false);
});

test('runner options show every runner with its state, and a Run gets a fresh valid request id', () => {
    const overview = {
        runners: [
            { id: 'llama.cpp', supported: true, installed: true, version: 'b11125' },
            { id: 'ollama', supported: true, installed: true, version: '0.34.3' },
            { id: 'vllm', supported: false, installed: false, reason: 'Not supported' },
        ],
    };
    const model = { runners: {
        'llama.cpp': { admission: { status: 'ok' }, download: { state: 'complete' } },
        ollama: { admission: { status: 'insufficient-now' }, download: { state: 'absent' } },
        vllm: { admission: { status: 'incompatible' } },
    } };
    assert.deepEqual(runnerOptions(overview, model).map((option) => option.label), [
        'llama.cpp b11125 · ready to run',
        'Ollama 0.34.3 · not now',
        'vLLM · not supported',
    ]);
    const first = newRequestId();
    assert.match(first, /^[A-Za-z0-9_-]{8,128}$/);
    assert.notEqual(first, newRequestId());
    assert.equal(shouldPoll('downloading'), true);
    assert.equal(shouldPoll('ready'), true);
    for (const settled of ['idle', 'error', 'paused', '', undefined]) assert.equal(shouldPoll(settled), false);
});

test('the Add model form builds a registry entry for a GGUF file or an Ollama tag', () => {
    assert.deepEqual(modelEntryFromForm({
        id: 'Qwen3.6-35B-A3B', displayName: 'Qwen3.6 35B A3B', architecture: 'moe', license: 'Apache-2.0', sourceKind: 'huggingface',
        repo: 'unsloth/Qwen3.6-35B-A3B-GGUF', file: 'Qwen3.6-35B-A3B-UD-Q3_K_M.gguf', revision: '', quantization: 'UD-Q3_K_M', tag: 'ignored',
    }), {
        id: 'qwen3.6-35b-a3b',
        displayName: 'Qwen3.6 35B A3B',
        license: 'Apache-2.0',
        architecture: 'moe',
        sources: { 'llama.cpp': { type: 'huggingface', repo: 'unsloth/Qwen3.6-35B-A3B-GGUF', file: 'Qwen3.6-35B-A3B-UD-Q3_K_M.gguf', revision: 'main', quantization: 'UD-Q3_K_M' } },
    });
    assert.deepEqual(modelEntryFromForm({ id: 'granite', sourceKind: 'ollama', tag: 'granite4:tiny-h', repo: 'ignored' }).sources, {
        ollama: { type: 'ollama', tag: 'granite4:tiny-h' },
    });
});

test('tool results are parsed, and tool failures surface their message', () => {
    assert.deepEqual(parseToolResult({ content: [{ type: 'text', text: '{"phase":"idle"}' }] }), { phase: 'idle' });
    assert.throws(() => parseToolResult({ isError: true, content: [{ type: 'text', text: 'MCP error -32603: not_ready: No local model is ready.' }] }),
        /^Error: not_ready: No local model is ready\.$/);
    assert.throws(() => parseToolResult({ content: [{ type: 'text', text: '{"ok":false,"error":"busy","message":"Busy."}' }] }), /Busy\./);
    const merged = mergeLogs([{ seq: 1, line: 'a' }, { seq: 2, line: 'b' }], [{ seq: 2, line: 'b' }, { seq: 3, line: 'c' }], 2);
    assert.deepEqual(merged.map((entry) => entry.seq), [2, 3]);
});

test('confirmation text cannot break out of the modal attribute', () => {
    assert.equal(confirmMessage('Remove "Evil" <img src=x onerror=alert(1)>?'), "Remove 'Evil' img src=x onerror=alert(1)?");
    const presenter = fs.readFileSync(new URL('IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.js', ROOT), 'utf8');
    const confirms = presenter.match(/showModal\('confirm-action-modal', \{\s*message: ([a-zA-Z]+)\(/g) || [];
    assert.equal(confirms.length, 3);
    assert.ok(confirms.every((call) => call.endsWith('confirmMessage(')));
});
