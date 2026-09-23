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

// Explorer's settings style rules (explorer/tests/unit/settingsModalPluginSettings.test.js,
// the DPU Data Sources assertions), applied to this plugin because Explorer's
// own test reads only the DPU files.
test('the settings plugin follows Explorer settings style rules', () => {
    const presenter = readText('local-llm-settings.js');
    const template = readText('local-llm-settings.html');
    const styles = readText('local-llm-settings.css');
    assert.match(presenter, /constructor\(element, invalidate\)[\s\S]*?this\.invalidate\(\);/);
    assert.match(presenter, /webSkelPresenter\?\.setOptions/);
    assert.match(template, /class="settings-tabs"[\s\S]*?role="tablist"/);
    assert.match(template, /data-llm-tab="models"[\s\S]*?role="tab"/);
    assert.match(template, /data-llm-panel="models"[\s\S]*?role="tabpanel"/);
    assert.match(template, /<custom-select id="localLlmRunner"/);
    assert.match(presenter, /assistOS\.UI\.showModal\('confirm-action-modal'/);
    assert.doesNotMatch(presenter, /window\.confirm|window\.prompt|window\.alert/);
    assert.match(template, /class="close"[\s\S]*?\/explorer\/assets\/icons\/x-mark\.svg/);
    assert.match(template, /data-local-action="toggleFullscreen"/);
    assert.match(template, /\/explorer\/assets\/icons\/fullscreen\.svg/);
    assert.match(template, /class="settings-card settings-card-static local-llm-editor"/);
    assert.doesNotMatch(template, /<select\b/);
    assert.doesNotMatch(presenter, /<select\b/);
    assert.doesNotMatch(template, /\$\$/, 'WebSkel treats $$name as a template variable');
    assert.doesNotMatch(styles, /min-width:\s*(?:[1-9]\d*px|min\()/);
    assert.doesNotMatch(styles, /#[0-9a-f]{3,8}\b/i);
    assert.doesNotMatch(styles, /\.(?:general-button|gray-button|form-input|modal-actions)\s*\{/);
    assert.doesNotMatch(styles, /(?:color|background|border(?:-color|-radius)?|box-shadow):/);
    assert.match(styles, /var\(--(?:space|text|surface|border|radius)-/);
    assert.match(styles, /local-llm-settings-dialog\s*\{[\s\S]*?height:\s*min\(720px,/);
    assert.match(styles, /local-llm-settings-dialog\.is-fullscreen/);
    assert.match(presenter, /classList\.toggle\('is-fullscreen'/);
    assert.match(styles, /@media \(max-width: 720px\)/);
    // Every rule is scoped to the component tag or its dialog.
    for (const rule of styles.replace(/@media[^{]*\{/g, '').split('}')) {
        const selector = rule.split('{')[0].trim();
        if (!selector) continue;
        for (const part of selector.split(',')) {
            assert.match(part.trim(), /^(?:dialog\.modal\.local-llm-settings-dialog|local-llm-settings\b)/, part.trim());
        }
    }
});

test('the settings modal toggles fullscreen on its host dialog', () => {
    const classes = new Set();
    const dialog = {
        classList: {
            contains: (name) => classes.has(name),
            toggle: (name, enabled) => (enabled ? classes.add(name) : classes.delete(name)),
        },
    };
    const aria = new Map();
    const modal = new presenterModule.LocalLlmSettings({ closest: () => dialog }, () => {});
    modal.fullscreenButton = { setAttribute: (name, value) => aria.set(name, value) };
    modal.toggleFullscreen();
    assert.equal(classes.has('is-fullscreen'), true);
    assert.equal(aria.get('aria-pressed'), 'true');
    modal.toggleFullscreen();
    assert.equal(classes.has('is-fullscreen'), false);
    assert.equal(aria.get('aria-pressed'), 'false');
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
    const presenter = readText('local-llm-settings.js');
    const confirms = presenter.match(/showModal\('confirm-action-modal', \{\s*message: ([a-zA-Z]+)\(/g) || [];
    assert.equal(confirms.length, 3);
    assert.ok(confirms.every((call) => call.endsWith('confirmMessage(')));
});
