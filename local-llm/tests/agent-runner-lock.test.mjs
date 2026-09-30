// The agent's own runner lock (DS004): the lock of the platform, merged with the
// image's; .tar.zst archives; the CI check that gates entries; and the runners
// that come from it at run time.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const REPOSITORY = new URL('../../', import.meta.url);
const WORKFLOW = new URL('.github/workflows/runner-lock-check.yml', REPOSITORY);

// The keys of a top-level block of the workflow (two-space indentation), without a YAML parser.
function blockKeys(text, name, indent = 2) {
    const lines = text.split('\n');
    const start = lines.findIndex((line) => line === `${name}:`);
    if (start < 0) return null;
    const keys = [];
    for (const line of lines.slice(start + 1)) {
        if (/^\S/.test(line)) break;
        const match = new RegExp(`^ {${indent}}([A-Za-z_][\\w-]*):`).exec(line);
        if (match) keys.push(match[1]);
    }
    return keys;
}

// The repository's .github directory is not part of a mount of local-llm alone.
const repositoryRoot = fs.existsSync(new URL('AGENTS.md', REPOSITORY));

test('the runner-lock check runs for pull requests and on demand only, on both architectures, inside the published image', {
    skip: !repositoryRoot && 'the repository root is not mounted, so its .github directory is not here',
}, () => {
    const text = fs.readFileSync(WORKFLOW, 'utf8');
    // Never a push: pushing the feature branch must not start CI.
    assert.deepEqual(blockKeys(text, 'on'), ['pull_request', 'workflow_dispatch']);
    assert.doesNotMatch(text, /^\s{2}(push|schedule|workflow_run|pull_request_target):/m);
    // A pull request starts it only when a lock or the installer code changes.
    const paths = text.slice(text.indexOf('paths:'), text.indexOf('workflow_dispatch:'));
    for (const file of [
        'local-llm/catalog/runners.lock.linux-*.json',
        'local-llm/src/controller/runnerInstaller.mjs',
        'local-llm/src/controller/runnerLock.mjs',
        'local-llm/tools/runner_install_check.mjs',
        '.github/workflows/runner-lock-check.yml',
    ]) {
        assert.ok(paths.includes(`'${file}'`), file);
    }
    assert.match(text, /^permissions:\n {2}contents: read$/m);
    // One job per architecture, each on its own runner.
    assert.match(text, /- arch: amd64\n\s+runs-on: ubuntu-24\.04\n/);
    assert.match(text, /- arch: arm64\n\s+runs-on: ubuntu-24\.04-arm\n/);
    // The published image and the repository's own installer code and lock, mounted read-only.
    assert.match(text, /IMAGE: docker\.io\/assistos\/local-llm:latest/);
    assert.match(text, /-v "\$PWD\/local-llm:\/code:ro"/);
    assert.match(text, /node \/code\/tools\/runner_install_check\.mjs "\$runner"/);
    assert.match(text, /--lock "\/code\/catalog\/runners\.lock\.linux-\$\{ARCH\}\.json"/);
    // Every entry of the platform's lock is installed, and a proprietary one is never downloaded.
    assert.match(text, /Object\.keys\(require\(process\.argv\[1\]\)\.runners\)/);
    assert.match(text, /--validate-only/);
    assert.match(text, /--network=none/);
});
