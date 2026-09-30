import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import * as presenterModule from '../IDE-plugins/local-llm-settings/local-llm-settings.js';
import {
    confirmMessage,
    fieldsFromSchema,
    lookupCommitFor,
    lookupFormatFor,
    mergeLogs,
    modelEntryFromForm,
    newRequestId,
    paramsFromForm,
    parseToolResult,
    rememberRunners,
    runnerLabel,
    runnerOptions,
    shouldPoll,
    sizingSourceFor,
    stripBidi,
    suggestModelId,
} from '../IDE-plugins/local-llm-settings/local-llm-settings-model.js';
import { validateModel } from '../src/controller/catalog.mjs';
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

test('a field that takes a named value or a number is one text field that accepts either', () => {
    const schema = { type: 'object', properties: {
        gpu: { type: ['string', 'number'], enum: ['off', 'max'], minimum: 0, maximum: 1, default: 'max', title: 'GPU offload', description: 'x' },
    } };
    const [field] = fieldsFromSchema(schema, {});
    assert.equal(field.kind, 'enumOrNumber');
    assert.deepEqual(field.options.map((option) => option.value), ['off', 'max']);
    assert.equal(field.min, 0);
    assert.equal(field.max, 1);
    assert.deepEqual(paramsFromForm([field], { gpu: 'off' }), { params: { gpu: 'off' }, errors: [] });
    assert.deepEqual(paramsFromForm([field], { gpu: '0.5' }), { params: { gpu: 0.5 }, errors: [] });
    assert.match(paramsFromForm([field], { gpu: '2' }).errors[0], /between 0 and 1/);
    assert.match(paramsFromForm([field], { gpu: 'half' }).errors[0], /off, max or a number/);
});

test('runner labels come from the runners the agent reports', () => {
    rememberRunners([{ id: 'ik_llama.cpp', displayName: 'ik_llama.cpp' }, { id: 'future', displayName: 'Future <runner>' }]);
    assert.equal(runnerLabel('future'), 'Future <runner>');
    assert.equal(runnerLabel('ollama'), 'Ollama');
    assert.equal(runnerLabel('unknown-id'), 'unknown-id');
    // Intended change (runners plan I9): LM Studio is a runner again. It has no
    // built-in label; like any runner it is named by the overview, which says
    // it is for internal use only.
    assert.equal(runnerLabel('lmstudio'), 'lmstudio');
    rememberRunners([{ id: 'lmstudio', displayName: 'LM Studio (internal use only)' }]);
    assert.equal(runnerLabel('lmstudio'), 'LM Studio (internal use only)');
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
        sources: { gguf: { type: 'huggingface', repo: 'unsloth/Qwen3.6-35B-A3B-GGUF', file: 'Qwen3.6-35B-A3B-UD-Q3_K_M.gguf', revision: 'main', quantization: 'UD-Q3_K_M' } },
    });
    assert.deepEqual(modelEntryFromForm({ id: 'granite', sourceKind: 'ollama', tag: 'granite4:tiny-h', repo: 'ignored' }).sources, {
        ollama: { type: 'ollama', tag: 'granite4:tiny-h' },
    });
});

test('the Add model form builds entries for every source kind: GGUF, safetensors, EXL3 and Ollama', () => {
    // The form offers the four kinds, in this order.
    const html = fs.readFileSync(new URL('IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.html', ROOT), 'utf8');
    const select = /<custom-select[^>]*data-name="sourceKind"[^>]*>/.exec(html)?.[0] || '';
    const offered = JSON.parse(decodeURIComponent(/data-options="([^"]*)"/.exec(select)?.[1] || '%5B%5D'));
    assert.deepEqual(offered.map((option) => option.value), ['huggingface', 'hf', 'exl3', 'ollama']);
    assert.ok(offered.every((option) => option.label.length > 0));
    const base = { id: 'X-Model', repo: ' owner/repo ', revision: '', displayName: 'X', license: 'MIT', architecture: 'moe' };
    const entries = {
        gguf: modelEntryFromForm({ ...base, sourceKind: 'huggingface', file: 'm-Q4_K_M.gguf', quantization: 'Q4_K_M' }),
        hf: modelEntryFromForm({ ...base, sourceKind: 'hf', file: 'ignored.gguf', quantization: 'ignored', tag: 'ignored' }),
        exl3: modelEntryFromForm({ ...base, sourceKind: 'exl3', revision: '4.0bpw' }),
        ollama: modelEntryFromForm({ id: 'x-tag', sourceKind: 'ollama', tag: 'qwen2.5:0.5b', repo: 'ignored', file: 'ignored.gguf' }),
    };
    assert.deepEqual(entries.gguf.sources, { gguf: { type: 'huggingface', repo: 'owner/repo', file: 'm-Q4_K_M.gguf', revision: 'main', quantization: 'Q4_K_M' } });
    assert.deepEqual(entries.hf.sources, { hf: { type: 'hf-snapshot', repo: 'owner/repo', revision: 'main' } });
    assert.deepEqual(entries.exl3.sources, { exl3: { type: 'hf-snapshot', repo: 'owner/repo', revision: '4.0bpw' } });
    assert.deepEqual(entries.ollama.sources, { ollama: { type: 'ollama', tag: 'qwen2.5:0.5b' } });
    assert.deepEqual([entries.hf.id, entries.hf.architecture, entries.hf.license, entries.hf.displayName], ['x-model', 'moe', 'MIT', 'X']);
    // Every one is an entry the agent accepts, as the form builds it.
    for (const [kind, entry] of Object.entries(entries)) assert.doesNotThrow(() => validateModel(entry), kind);
    // An unknown kind is a GGUF file, as before.
    assert.ok(modelEntryFromForm({ ...base, sourceKind: 'modelscope', file: 'a.gguf' }).sources.gguf);
    // Sizing passes through for every kind, and the sizing source only where the entry has that source.
    const sized = { contextLength: '32768', layers: '24', kvBytesPerToken: '12288' };
    const fits = (kind, sizingSource) => modelEntryFromForm({ ...base, ...sized, sourceKind: kind, file: 'm.gguf', tag: 't:1', sizingSource }).sizingSource;
    assert.equal(fits('huggingface', 'gguf-header'), 'gguf-header');
    assert.equal(fits('hf', 'gguf-header'), undefined);
    assert.equal(fits('ollama', 'gguf-header'), undefined);
    assert.equal(fits('hf', 'config.json'), 'config.json');
    assert.equal(fits('exl3', 'config.json'), 'config.json');
    assert.equal(fits('huggingface', 'config.json'), undefined);
    for (const kind of ['huggingface', 'hf', 'exl3', 'ollama']) assert.equal(fits(kind, 'manual'), 'manual', kind);
    for (const junk of ['guess', '', null, 7, undefined]) assert.equal(fits('huggingface', junk), undefined, String(junk));
    const hf = modelEntryFromForm({ ...base, ...sized, sourceKind: 'hf', sizingSource: 'config.json' });
    assert.deepEqual([hf.contextLength, hf.memory], [32768, { layers: 24, kvBytesPerToken: 12288 }]);
    assert.doesNotThrow(() => validateModel(hf));
    const gguf = modelEntryFromForm({ ...base, ...sized, sourceKind: 'huggingface', file: 'm.gguf', sizingSource: 'gguf-header' });
    assert.doesNotThrow(() => validateModel(gguf));
    // A lookup format for each kind that has a lookup; an Ollama tag has none.
    assert.deepEqual(['huggingface', 'hf', 'exl3', 'ollama', 'constructor'].map(lookupFormatFor), ['gguf', 'hf', 'exl3', null, null]);
});

test('the sizing source follows what a lookup filled in: untouched values keep it, changed ones are manual, none stays absent', () => {
    const lookup = {
        source: 'gguf-header', repo: 'owner/repo', revision: 'main', file: 'm.gguf', architecture: 'dense',
        values: { contextLength: '32768', layers: '24', kvBytesPerToken: '12288' },
    };
    const form = { repo: 'owner/repo', revision: '', file: 'm.gguf', architecture: 'dense', contextLength: '32768', layers: '24', kvBytesPerToken: '12288' };
    assert.equal(sizingSourceFor(form, lookup), 'gguf-header');
    assert.equal(sizingSourceFor({ ...form, revision: 'main', contextLength: ' 32768 ' }, lookup), 'gguf-header');
    for (const change of [{ layers: '25' }, { kvBytesPerToken: '' }, { contextLength: '16384' }, { architecture: 'moe' }, { file: 'other.gguf' }, { repo: 'owner/other' }, { revision: 'dev' }]) {
        assert.equal(sizingSourceFor({ ...form, ...change }, lookup), 'manual', JSON.stringify(change));
    }
    // Every value cleared: nothing is left to attribute to anyone.
    assert.equal(sizingSourceFor({ ...form, contextLength: '', layers: '', kvBytesPerToken: '' }, lookup), undefined);
    // Without a lookup the entry keeps today's shape, however much was typed; a header that gave no value for a field stays empty.
    assert.equal(sizingSourceFor(form, null), undefined);
    assert.equal(sizingSourceFor(form, { ...lookup, source: 'guess' }), undefined);
    const partial = { ...lookup, values: { contextLength: '32768', layers: '24', kvBytesPerToken: '' } };
    assert.equal(sizingSourceFor({ ...form, kvBytesPerToken: '' }, partial), 'gguf-header');
    assert.equal(sizingSourceFor(form, partial), 'manual');
    // A snapshot's lookup has no file.
    const snapshot = { source: 'config.json', repo: 'owner/repo', revision: 'main', architecture: 'dense', values: lookup.values };
    assert.equal(sizingSourceFor({ ...form, file: 'whatever' }, snapshot), 'config.json');
    // Suggested ids are valid ids.
    const idPattern = /^[a-z0-9][a-z0-9._-]{1,63}$/;
    for (const [repo, quantization, expected] of [
        ['Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'Q4_K_M', 'qwen2.5-0.5b-instruct-gguf-q4_k_m'],
        ['unsloth/gpt-oss-20b-GGUF', 'UD-Q3_K_XL', 'gpt-oss-20b-gguf-ud-q3_k_xl'],
        ['owner/Name With Spaces!', '', 'name-with-spaces'],
        ['owner/' + 'x'.repeat(100), 'Q4_K_M', 'x'.repeat(64)],
        ['owner/---', '', 'model'], ['', '', 'model'],
    ]) {
        const id = suggestModelId(repo, quantization);
        assert.equal(id, expected, repo);
        assert.match(id, idPattern, repo);
    }
});

test('a looked-up commit replaces the revision in every Hugging Face source, and only a real commit does', () => {
    const commit = 'f'.repeat(40);
    const base = { id: 'x-model', repo: 'owner/repo', revision: 'dev', file: 'm.gguf' };
    assert.equal(modelEntryFromForm({ ...base, sourceKind: 'huggingface', commit }).sources.gguf.revision, commit);
    assert.equal(modelEntryFromForm({ ...base, sourceKind: 'hf', commit }).sources.hf.revision, commit);
    assert.equal(modelEntryFromForm({ ...base, sourceKind: 'exl3', commit }).sources.exl3.revision, commit);
    assert.equal(modelEntryFromForm({ ...base, revision: '', sourceKind: 'huggingface', commit }).sources.gguf.revision, commit, 'an empty revision too');
    // Anything that is not 40 lowercase hex characters is ignored, and the revision as typed stands (main when empty).
    for (const junk of [undefined, null, '', 'main', 'F'.repeat(40), 'f'.repeat(39), 'f'.repeat(41), `${commit} `, 'g'.repeat(40), 7, ['f'.repeat(40)]]) {
        assert.equal(modelEntryFromForm({ ...base, sourceKind: 'huggingface', commit: junk }).sources.gguf.revision, 'dev', String(junk));
    }
    assert.equal(modelEntryFromForm({ ...base, revision: '', sourceKind: 'hf', commit: 'nope' }).sources.hf.revision, 'main');
    // An Ollama tag has no revision to replace.
    assert.deepEqual(modelEntryFromForm({ id: 'x-tag', sourceKind: 'ollama', tag: 'q:1', commit }).sources, { ollama: { type: 'ollama', tag: 'q:1' } });
    // The entries validate with a commit as their revision.
    for (const kind of ['huggingface', 'hf', 'exl3']) assert.doesNotThrow(() => validateModel(modelEntryFromForm({ ...base, sourceKind: kind, commit })), kind);
    // The commit comes from the lookup only while the form still names what it read.
    const lookup = { source: 'gguf-header', repo: 'owner/repo', revision: 'main', file: 'm.gguf', architecture: 'dense', commit, values: {} };
    const form = { repo: 'owner/repo', revision: '', file: 'm.gguf' };
    assert.equal(lookupCommitFor(form, lookup), commit);
    assert.equal(lookupCommitFor({ ...form, revision: 'main' }, lookup), commit);
    assert.equal(lookupCommitFor({ ...form, layers: '99', kvBytesPerToken: '1', architecture: 'moe' }, lookup), commit, 'edited sizing does not change the files');
    for (const change of [{ repo: 'owner/other' }, { revision: 'dev' }, { file: 'other.gguf' }]) assert.equal(lookupCommitFor({ ...form, ...change }, lookup), undefined, JSON.stringify(change));
    assert.equal(lookupCommitFor(form, null), undefined);
    assert.equal(lookupCommitFor(form, { ...lookup, commit: undefined }), undefined);
    assert.equal(lookupCommitFor(form, { ...lookup, commit: 'main' }), undefined);
    // A snapshot's lookup names no file, so any file in the form is beside the point.
    assert.equal(lookupCommitFor({ ...form, file: 'whatever.gguf' }, { ...lookup, file: undefined }), commit);
    // A lookup that sized nothing (no source) still carries its commit, and gives no sizing source.
    assert.equal(lookupCommitFor(form, { ...lookup, source: null }), commit);
    assert.equal(sizingSourceFor({ ...form, layers: '24' }, { ...lookup, source: null }), undefined);
    // stripBidi drops the bidirectional controls and nothing else.
    assert.equal(stripBidi('a\u202Eb\u2066c\u2069d'), 'abcd');
    assert.equal(stripBidi('\u200E\u200F'), '\u200E\u200F', 'only the ranges named: embeddings, overrides and isolates');
    assert.equal(stripBidi(null), '');
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
    assert.equal(confirms.length, 5);
    assert.ok(confirms.every((call) => call.endsWith('confirmMessage(')));
});
