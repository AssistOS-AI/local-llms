// V23, the local half: the engine has no default location for a catalog or a runner lock, so nothing here may lean on one. No file
// of the agent asks the engine's loader for its seed catalog without naming the file, none imports an agent-lock helper from the
// engine, and the places that read the agent's own catalog and locks (the CPU golden, the calibration tool, the install check, the
// controller) read the files in this repository.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadRunnerLocks, loadSeedCatalog as engineLoadSeedCatalog } from 'pworker/local-llm';

import { agentHost, agentRunnerLockFile, agentSeedCatalogFile } from '../src/agentHost.mjs';
import { createAgentController, loadSeedCatalog } from '../src/engineOptions.mjs';
import { readSnapshot } from '../src/hardwareLimits.mjs';
import { collectCpuOverview, cpuFs } from './overview-cpu-scenarios.mjs';

const ROOT = new URL('..', import.meta.url).pathname;
const SELF = new URL(import.meta.url).pathname;
const golden = JSON.parse(fs.readFileSync(new URL('./overview-cpu-golden-b2b387f.json', import.meta.url), 'utf8'));

function* sources(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const file = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            if (!['node_modules', 'IDE-plugins', 'docs'].includes(entry.name)) yield* sources(file);
        } else if (/\.mjs$/.test(entry.name) && file !== SELF) {
            yield file;
        }
    }
}

// What a file takes from the engine's package: the names it imports, statically or by `await import`, from a specifier of pworker.
function engineNames(text) {
    const names = new Set();
    for (const match of text.matchAll(/import\s*\{([^}]*)\}\s*from\s*'(pworker[^']*)'/g)) {
        for (const raw of match[1].split(',').map((part) => part.trim()).filter(Boolean)) names.add(raw.split(/\s+as\s+/)[0]);
    }
    for (const match of text.matchAll(/const\s*\{([^}]*)\}\s*=\s*await\s+import\(\s*'(pworker[^']*)'\s*\)/g)) {
        for (const raw of match[1].split(',').map((part) => part.trim()).filter(Boolean)) names.add(raw.split(':')[0].trim());
    }
    return names;
}

/** The problems of one file: an unnamed seed catalog from the engine, or an agent lock helper from it. */
export function hiddenDefaults(text) {
    const names = engineNames(text);
    const problems = [];
    if (names.has('loadSeedCatalog') && /(?<![\w$.])loadSeedCatalog\(\s*\)/.test(text)) problems.push('calls the engine\'s loadSeedCatalog() without a file');
    for (const helper of ['agentRunnerLockFile', 'DEFAULT_RUNNER_LOCK', 'AGENT_LOCK_PLATFORMS']) if (names.has(helper)) problems.push(`imports ${helper} from the engine`);
    return problems;
}

test('no file of the agent asks the engine for its seed catalog unnamed, or for an agent lock helper', () => {
    const files = [...sources(ROOT)];
    for (const expected of ['src/engineOptions.mjs', 'src/agentHost.mjs', 'tools/vllm_mps_calibration.mjs', 'tools/runner_install_check.mjs', 'tests/cpu-host-unchanged.test.mjs', 'tests/vllm.test.mjs']) {
        assert.ok(files.includes(path.join(ROOT, expected)), `${expected} was scanned`);
    }
    assert.ok(files.length > 40, `${files.length} files scanned`);
    for (const file of files) assert.deepEqual(hiddenDefaults(fs.readFileSync(file, 'utf8')), [], path.relative(ROOT, file));
    // The scanner can fail: each of these says what is wrong with it.
    assert.deepEqual(hiddenDefaults("import { loadSeedCatalog } from 'pworker/local-llm';\nconst seed = loadSeedCatalog();\n"), ['calls the engine\'s loadSeedCatalog() without a file']);
    assert.deepEqual(hiddenDefaults("const { loadSeedCatalog } = await import('pworker/local-llm');\nloadSeedCatalog( );\n"), ['calls the engine\'s loadSeedCatalog() without a file']);
    assert.deepEqual(hiddenDefaults("import { agentRunnerLockFile as lock } from 'pworker/local-llm/internal/controller/runnerLock.mjs';\n"), ['imports agentRunnerLockFile from the engine']);
    // The agent's own bound loader, and a call that names its file, are what is allowed.
    assert.deepEqual(hiddenDefaults("import { loadSeedCatalog } from '../src/engineOptions.mjs';\nconst seed = loadSeedCatalog();\n"), []);
    assert.deepEqual(hiddenDefaults("import { loadSeedCatalog } from 'pworker/local-llm';\nloadSeedCatalog(agentSeedCatalogFile());\n"), []);
});

test('the engine itself refuses an unnamed catalog or lock, so a forgotten argument cannot read some other file', () => {
    assert.throws(() => engineLoadSeedCatalog(), (error) => error.code === 'missing_file');
    assert.throws(() => loadRunnerLocks({}), (error) => error.code === 'missing_file');
    assert.throws(() => loadRunnerLocks({ image: null }), (error) => error.code === 'missing_file');
});

test('the agent\'s catalog and locks are the files of this repository, not of the package', () => {
    const catalog = path.join(ROOT, 'catalog');
    assert.equal(agentSeedCatalogFile(), path.join(catalog, 'models.json'));
    assert.equal(agentHost({}).seedCatalogFile, agentSeedCatalogFile());
    assert.equal(agentRunnerLockFile('x64'), path.join(catalog, 'runners.lock.linux-amd64.json'));
    assert.equal(agentRunnerLockFile('arm64'), path.join(catalog, 'runners.lock.linux-arm64.json'));
    for (const file of [agentSeedCatalogFile(), agentRunnerLockFile('x64'), agentRunnerLockFile('arm64')]) {
        assert.ok(fs.existsSync(file), file);
        assert.equal(file.includes('node_modules'), false, file);
    }
    // The bound loader reads the first of them unless a file is named.
    assert.deepEqual(loadSeedCatalog().map((model) => model.id), engineLoadSeedCatalog(agentSeedCatalogFile()).map((model) => model.id));
    // The package ships a seed catalog of its own only as a fixture of its tests; the agent never reads one from there.
    assert.equal(fs.existsSync(path.join(ROOT, 'node_modules', 'pworker', 'lib', 'local-llm', 'catalog', 'models.json')), false);
});

test('the CPU golden follows the agent\'s catalog: one changed pin in a copy of models.json makes the comparison fail', async (t) => {
    const name = 'x64, base contract';
    const own = await collectCpuOverview({ createController: createAgentController, readSnapshot, loadSeedCatalog, arch: 'x64' });
    assert.deepEqual(own, golden.scenarios[name], 'the agent\'s own catalog is the golden');
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-catalog-copy-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const document = JSON.parse(fs.readFileSync(agentSeedCatalogFile(), 'utf8'));
    const model = document.models.find((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m');
    model.sources.gguf.size += 1;
    const copy = path.join(dir, 'models.json');
    fs.writeFileSync(copy, JSON.stringify(document, null, 2));
    const changed = await collectCpuOverview({ createController: createAgentController, readSnapshot, loadSeedCatalog: () => loadSeedCatalog(copy), arch: 'x64' });
    assert.notDeepEqual(changed, golden.scenarios[name], 'a changed pin is seen by the golden');
});

test('the controller reads the seed catalog named by LOCAL_LLM_CATALOG_FILE, and the agent\'s own when none is named', async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-catalog-env-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const document = JSON.parse(fs.readFileSync(agentSeedCatalogFile(), 'utf8'));
    document.models = document.models.filter((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m');
    const only = path.join(dir, 'only.json');
    fs.writeFileSync(only, JSON.stringify(document));
    const ids = async (env) => {
        const controller = createAgentController({ env: { PATH: '/usr/bin', LOCAL_LLM_DATA_DIR: path.join(dir, 'data'), ...env }, fsApi: cpuFs(), execFileImpl: (command, args, options, callback) => callback(Object.assign(new Error('absent'), { code: 'ENOENT' }), '', ''),
            detectRunner: () => ({ installed: false, version: null, reason: 'fixture' }), fileExists: () => false, sharedModelsRoot: null });
        try {
            return (await controller.overview()).models.map((model) => model.id).sort();
        } finally {
            await controller.drain();
        }
    };
    assert.deepEqual(await ids({ LOCAL_LLM_CATALOG_FILE: only }), ['qwen2.5-0.5b-instruct-q4_k_m']);
    // On the CPU profile the overview lists the models the CPU can run: the agent's own catalog has more than the one named above.
    const own = await ids({});
    assert.ok(own.length > 1 && own.includes('qwen2.5-0.5b-instruct-q4_k_m') && own.includes('qwen2.5-1.5b-instruct-q4_k_m'), own.join());
    assert.ok(own.every((id) => loadSeedCatalog().some((model) => model.id === id)));
});
