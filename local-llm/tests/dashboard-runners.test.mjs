import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { runnersPanelHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-runners-view.js';
import { LocalLlmDashboard } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.js';

const base = new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/', import.meta.url);
const runner = (id, extra = {}) => ({ id, displayName: id, supported: true, installed: true, ...extra });
const order = html => [...html.matchAll(/data-runner-id="([^"]+)"/g)].map(match => match[1]);

test('runner headings have inner spacing and only a softly fading bottom shadow', async () => {
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    const heading = css.match(/\.local-llm-runner-heading \{([^}]+)\}/)?.[1] || '';
    assert.match(heading, /padding: 0px 8px 2px/);
    assert.match(heading, /background: transparent/);
    assert.match(heading, /color: var\(--text\)/);
    assert.match(heading, /box-shadow: none/);
    const shadow = css.match(/\.local-llm-runner-heading::after \{([^}]+)\}/)?.[1] || '';
    assert.match(shadow, /border-radius: 50%/);
    assert.match(shadow, /box-shadow: var\(--shadow-sm\)/);
    assert.match(shadow, /clip-path: inset\(100% -8px -10px -8px\)/);
    assert.match(shadow, /pointer-events: none/);
});

test('runner cards show a compact licence link outside info without inventing missing licence data', () => {
    const licence = { name: 'MIT', url: 'https://example.test/LICENSE', source: 'https://example.test/source' };
    const html = runnersPanelHtml([runner('licensed', { install: { licence, installed: true } }), runner('unreported')]);
    assert.match(html, /data-llm-runner-licence><dt>Licence<\/dt><dd><a href="https:\/\/example.test\/LICENSE" target="_blank" rel="noopener noreferrer">MIT<\/a><\/dd><\/div>/);
    assert.match(html, /data-llm-runner-licence><dt>Licence<\/dt><dd>Not reported<\/dd><\/div>/);
    const cardLines = [...html.matchAll(/data-llm-runner-licence>([^]*?)<\/div>/g)].map(match => match[1]);
    assert.equal(cardLines.length, 2);
    assert.ok(cardLines.every(line => !line.includes('source</a>')), 'source details stay in info');
    assert.match(html, /href="https:\/\/example.test\/source"/, 'full source remains available in info');
    const escaped = runnersPanelHtml([runner('escaped', { install: { licence: { name: '<script>licence</script>' } } })]);
    assert.match(escaped, /data-llm-runner-licence><dt>Licence<\/dt><dd>&lt;script&gt;licence&lt;\/script&gt;/);
    assert.doesNotMatch(escaped, /<script>/);
});

test('Runners is second in both visual and keyboard tab order', async () => {
    const html = await readFile(new URL('local-llm-dashboard.html', base), 'utf8');
    assert.deepEqual([...html.matchAll(/data-llm-tab="([^"]+)"/g)].map(match => match[1]), ['models', 'runners', 'playground', 'logs']);
    const p = new LocalLlmDashboard({}, () => {});
    let focused;
    p.tabs = new Map(['models', 'runners', 'playground', 'logs'].map(tab => [tab, { focus: () => { focused = tab; } }]));
    p.tabList = { contains: () => true };
    p.updateTabUI = () => {};
    for (const [current, key, next] of [['models', 'ArrowRight', 'runners'], ['runners', 'ArrowRight', 'playground'], ['models', 'ArrowLeft', 'logs'], ['runners', 'End', 'logs']]) {
        p.handleTabKeydown({ key, target: { closest: () => ({ dataset: { llmTab: current } }) }, preventDefault() {} });
        assert.equal(p.activeTab, next);
        assert.equal(focused, next);
    }
});

test('compact cards share fact rows and place Install in the status row without a reserved footer', async () => {
    const html = runnersPanelHtml([
        runner('ollama', { displayName: 'Ollama', installed: false, install: { version: '0.34.4', totalBytes: 1.5e9, licence: { name: 'MIT' } } }),
        runner('tabbyapi', { displayName: 'TabbyAPI (ExLlamaV3)', installed: false, supported: false, unsupportedReason: 'Not included in this image.' }),
    ]);
    const cards = [...html.matchAll(/<li class="local-llm-runner-item"[^]*?<\/li>/g)].map(match => match[0]);
    assert.equal(cards.length, 2);
    for (const card of cards) {
        assert.deepEqual([...card.matchAll(/<dt>([^<]+)<\/dt>/g)].map(match => match[1]), ['Version', 'Download', 'Licence']);
        assert.match(card, /<dl class="local-llm-runner-facts settings-card-meta">/);
        assert.match(card, /<div class="local-llm-runner-notes">/);
        assert.match(card, /<span class="local-llm-runner-state-actions">/);
        assert.doesNotMatch(card, /local-llm-runner-footer/);
        assert.ok(card.indexOf('local-llm-runner-state') < card.indexOf('local-llm-runner-facts'));
        assert.ok(card.indexOf('local-llm-runner-facts') < card.indexOf('local-llm-runner-notes'));
    }
    assert.match(cards[0], /<dt>Download<\/dt><dd>1\.5 GB<\/dd>/);
    assert.match(cards[0], /<div class="local-llm-runner-state">[^]*?data-local-action="installRunner ollama"[^]*?<\/button><\/span><\/div>\s*<dl/);
    assert.match(cards[1], /<dt>Version<\/dt><dd>Not reported<\/dd>/);
    assert.match(cards[1], /<dt>Download<\/dt><dd>Not reported<\/dd>/);
    assert.doesNotMatch(cards[1], /data-local-action="installRunner/);
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    assert.match(css, /\.local-llm-runners-panel \.local-llm-runner-item \{[^}]*grid-template-rows: subgrid;[^}]*grid-row: span 6/);
    assert.match(css, /\.local-llm-runners-panel \.local-llm-runner-item \{[^}]*grid-template-columns: minmax\(0, 1fr\);[^}]*justify-content: stretch/);
    assert.match(css, /\.local-llm-runner-facts \{ display: contents; \}/);
    assert.match(css, /\.local-llm-runner-fact \{[^}]*grid-template-columns: 5\.5rem minmax\(0, 1fr\)/);
    assert.match(css, /\.local-llm-runner-state-actions \{[^}]*margin-left: auto/);
    assert.doesNotMatch(css, /\.local-llm-runner-footer/);
});

test('alphabetical order uses display names, ignores case and does not mutate or reorder by status', () => {
    const runners = [runner('z', { displayName: 'vLLM' }), runner('b', { displayName: 'Ollama' }), runner('a', { displayName: 'llama.cpp' }), runner('c', { displayName: 'IK_llama.cpp' })];
    const original = [...runners];
    const expected = ['c', 'a', 'b', 'z'];
    assert.deepEqual(order(runnersPanelHtml(runners)), expected);
    assert.deepEqual(runners, original);
    runners[0].install = { installing: true, state: { phase: 'downloading' } };
    assert.deepEqual(order(runnersPanelHtml(runners)), expected);
});

test('cards expose distinct states, clear icons, native info and bounded responsive columns', async () => {
    const examples = [
        runner('installed'), runner('absent', { installed: false, install: {} }),
        runner('busy', { installed: false, install: { installing: true, state: { phase: 'downloading' } } }),
        runner('error', { installed: false, install: { state: { phase: 'error', error: 'Checksum mismatch' } } }),
        runner('paused', { installed: false, install: { state: { phase: 'paused', pausedReason: 'Disk full' } } }),
        runner('disabled', { enabled: false, disabledReason: 'Operator disabled this runner.' }),
        runner('unavailable', { supported: false, unsupportedReason: 'Requires a compatible GPU.' }),
    ];
    const html = runnersPanelHtml(examples);
    for (const entry of examples) assert.match(html, new RegExp(`data-runner-id="${entry.id}" data-runner-state="${entry.id}"`));
    assert.match(html, /spinner-orbit/);
    assert.match(html, /Installation failed/);
    assert.match(html, /Installation paused/);
    assert.match(html, /aria-label="Details for installed"/);
    assert.equal((html.match(/data-llm-runner-help/g) || []).length, examples.length);
    assert.match(html, /Checksum mismatch<\/p>/, 'actionable errors are outside the info popover');
    assert.match(html, /Requires a compatible GPU\.<\/p>/);
    const css = await readFile(new URL('local-llm-dashboard.css', base), 'utf8');
    assert.match(css, /container: local-llm-runners \/ inline-size/);
    assert.match(css, /@container local-llm-runners \(min-width: 36rem\)[^]*?repeat\(2, minmax\(0, 1fr\)\)/);
    assert.match(css, /@container local-llm-runners \(min-width: 58rem\)[^]*?repeat\(3, minmax\(0, 1fr\)\)/);
    assert.doesNotMatch(css, /@keyframes/, 'uses Explorer animation, not a new spinner');
});

test('installed is not in use; only the current active runner receives In use', () => {
    const runners = [runner('one'), runner('two')];
    assert.doesNotMatch(runnersPanelHtml(runners), /In use/);
    for (const phase of ['starting', 'loading', 'ready', 'stopping']) {
        const html = runnersPanelHtml(runners, { deployment: { runnerId: 'one', phase } });
        assert.equal((html.match(/In use/g) || []).length, 1);
        assert.match(html, /data-runner-id="one"[^]*?In use[^]*?data-runner-id="two"/);
    }
    for (const phase of ['idle', 'downloading', 'failed', 'cancelled']) assert.doesNotMatch(runnersPanelHtml(runners, { deployment: { runnerId: 'one', phase } }), /In use/);
    const p = new LocalLlmDashboard({}, () => {});
    p.overview = { runners, deployment: { runnerId: 'one', phase: 'ready' } };
    p.status = { phase: 'idle', deployment: null };
    p.runnersRegion = { innerHTML: '' };
    p.renderRunners();
    assert.doesNotMatch(p.runnersRegion.innerHTML, /In use/, 'fresh idle status beats a stale overview');
});

test('info identifiers are unique, values are escaped, and forbidden installs stay unavailable', () => {
    const runners = [runner('llama.cpp'), runner('llama-cpp'), runner('bad', { displayName: '<img src=x>', installed: false, supported: false, unsupportedReason: '<blocked>', install: { version: '<v>', totalBytes: 1 } })];
    const html = runnersPanelHtml(runners);
    const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map(match => match[1]);
    assert.equal(new Set(ids).size, ids.length);
    for (const [, target] of html.matchAll(/popovertarget="([^"]+)"/g)) assert.ok(ids.includes(target));
    assert.doesNotMatch(html, /<img|<blocked>|<v>/);
    assert.match(html, /&lt;img src=x&gt;/);
    assert.doesNotMatch(html, /installRunner bad/);
});

test('live runner refresh preserves open details and focused control; identical markup is untouched', t => {
    const previous = globalThis.document;
    t.after(() => { if (previous === undefined) delete globalThis.document; else globalThis.document = previous; });
    const helpId = 'localLlmRunner-61';
    let opens = 0;
    let focuses = 0;
    const button = { id: `${helpId}-button`, focus() { focuses += 1; } };
    const help = { id: helpId, showPopover({ source }) { assert.equal(source, button); opens += 1; } };
    globalThis.document = { activeElement: button };
    const p = new LocalLlmDashboard({}, () => {});
    p.overview = { runners: [runner('a')] };
    p.runnersRegion = {
        innerHTML: '', contains: () => true,
        querySelector: selector => selector === `#${helpId}-button` ? button : help,
    };
    p.renderRunners();
    assert.equal(opens, 1);
    assert.equal(focuses, 1);
    p.renderRunners();
    assert.equal(opens, 1);
    p.overview.runners[0].version = 'new';
    p.renderRunners();
    assert.equal(opens, 2);
    assert.equal(focuses, 2);
});
