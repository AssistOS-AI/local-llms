import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

// The chat endpoint and every MCP tool call start a fresh process, so they
// must not load the engine's controller or runners: only the error class, the
// observation id pattern and the pure chat helpers.
const ENTRIES = ['../src/chatResponder.mjs', '../tools/local_llm_tool.mjs', '../src/testPrompt.mjs'];
const ENGINE_ALLOWED = new Set(['errors.mjs', 'promptObservations.mjs', 'chat.mjs']);

function staticClosure(entryUrl) {
    const seen = new Set();
    const pending = [entryUrl];
    while (pending.length) {
        const url = pending.pop();
        if (seen.has(url)) continue;
        seen.add(url);
        const source = fs.readFileSync(new URL(url), 'utf8');
        for (const match of source.matchAll(/^\s*(?:import|export)\s[^'"]*?from\s+['"]([^'"]+)['"]/gm)) {
            const specifier = match[1];
            if (specifier.startsWith('node:')) continue;
            pending.push(specifier.startsWith('.') ? new URL(specifier, url).href : import.meta.resolve(specifier));
        }
    }
    return [...seen];
}

const engineRoot = fileURLToPath(new URL('./', import.meta.resolve('pworker/local-llm')));

for (const entry of ENTRIES) {
    test(`${entry.slice(3)} loads no engine module beyond the error, observation and chat helpers`, () => {
        const files = staticClosure(new URL(entry, import.meta.url).href).map((url) => fileURLToPath(url));
        const engine = files.filter((file) => file.startsWith(engineRoot)).map((file) => file.slice(engineRoot.length));
        assert.ok(engine.length > 0, 'the walker reached the engine package');
        assert.deepEqual(engine.filter((file) => !ENGINE_ALLOWED.has(file)), []);
    });
}
