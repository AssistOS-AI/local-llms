// tools/runner_install_check.mjs is the CI and image-publish install check (DS004). The check itself is the engine's; this entry
// gives it the container's host. Its command line, its report and its exit codes are what they were when the check lived here:
// the same cases ran against 34f8f498's tool and this one and printed the same bytes (see the commit message of the move).
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

const TOOL = new URL('../tools/runner_install_check.mjs', import.meta.url).pathname;
const CATALOG = new URL('../catalog/', import.meta.url).pathname;

async function check(args, env = {}) {
    // A dead proxy in the child: any download attempt would fail the check.
    const child = spawn(process.execPath, [TOOL, ...args], {
        stdio: ['ignore', 'pipe', 'pipe'], env: { PATH: process.env.PATH, HTTPS_PROXY: 'http://127.0.0.1:9', NODE_USE_ENV_PROXY: '1', ...env },
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    const code = await new Promise((resolve) => child.on('exit', resolve));
    return { code, stdout, stderr, report: stdout ? JSON.parse(stdout) : null };
}

function lockFixture(t) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-install-entry-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, 'runners.lock.json');
    fs.writeFileSync(file, JSON.stringify({
        schema: 'local-llm.runners-lock/v1',
        runners: {
            lmstudio: {
                version: '0.0.25-1', kind: 'archive',
                licence: { name: 'LM Studio Terms', url: 'https://lmstudio.ai/app-terms', requiresAcceptance: true, proprietary: true },
                files: [{ name: 'x.tar.gz', url: 'https://llmster.lmstudio.ai/download/x.tar.gz', size: 1000, sha256: '4'.repeat(64) }],
            },
            open: {
                version: '1', kind: 'archive', licence: { name: 'MIT', url: 'https://example.org' },
                files: [{ name: 'r.tar.gz', url: 'https://github.com/o/r/releases/download/v1/r.tar.gz', size: 10, sha256: 'a'.repeat(64) }],
            },
        },
    }));
    return { dir, file };
}

test('a proprietary entry is validated and never downloaded; --validate-only does the same for any entry; both print one JSON report and exit 0', async (t) => {
    const { dir, file } = lockFixture(t);
    const cache = path.join(dir, 'cache');
    const proprietary = await check(['lmstudio', '--lock', file, '--cache', cache]);
    assert.equal(proprietary.code, 0);
    assert.deepEqual([proprietary.report.ok, proprietary.report.downloaded], [true, false]);
    assert.match(proprietary.report.skipped, /^proprietary \(LM Studio Terms\): validated from the lock, not downloaded; installing accepts its terms$/);
    assert.deepEqual(proprietary.report.validated.hosts, ['llmster.lmstudio.ai']);
    const open = await check(['open', '--validate-only', '--lock', file, '--cache', cache]);
    assert.equal(open.code, 0);
    assert.match(open.report.skipped, /^validate-only: /);
    assert.equal(fs.existsSync(cache), false, 'nothing was fetched or cached');
    assert.ok(proprietary.stdout.endsWith('\n') && proprietary.stdout.startsWith('{\n  "runnerId": "lmstudio"'), 'two-space JSON, one trailing newline');
});

test('an unknown runner, a missing lock and no arguments fail with a coded report and exit 1, the whole report on a pipe', async (t) => {
    const { file } = lockFixture(t);
    const unknown = await check(['no-such-runner', '--lock', file]);
    assert.equal(unknown.code, 1);
    assert.equal(unknown.report.ok, false);
    assert.equal(unknown.report.error, 'not_installable');
    const absent = await check(['no-such-runner', '--lock', '/nonexistent/lock.json']);
    assert.equal(absent.code, 1);
    assert.equal(absent.report.error, 'not_installable', 'a lock that is not there is an empty lock, as it always was');
    const nothing = await check([]);
    assert.equal(nothing.code, 1);
    assert.equal(nothing.report.ok, false);
});

test('--require-executable refuses an archive entry that names no launch file, and every entry of the shipped agent locks passes it', async (t) => {
    const { file } = lockFixture(t);
    const refused = await check(['open', '--validate-only', '--require-executable', '--lock', file]);
    assert.equal(refused.code, 1);
    assert.equal(refused.report.error, 'invalid_runner_lock');
    for (const arch of ['amd64', 'arm64']) {
        const lock = path.join(CATALOG, `runners.lock.linux-${arch}.json`);
        for (const id of Object.keys(JSON.parse(fs.readFileSync(lock, 'utf8')).runners)) {
            const result = await check([id, '--validate-only', '--require-executable', '--lock', lock]);
            assert.equal(result.code, 0, `${arch} ${id}: ${result.stdout}`);
            assert.equal(result.report.downloaded, false, `${arch} ${id}`);
        }
    }
});

test('with no --lock the container\'s own image lock is read (LOCAL_LLM_RUNNER_LOCK, else /opt/local-llm/runners.lock.json), as the image publish workflow relies on', async (t) => {
    const { file } = lockFixture(t);
    const named = await check(['open', '--validate-only'], { LOCAL_LLM_RUNNER_LOCK: file });
    assert.equal(named.code, 0);
    assert.equal(named.report.runnerId, 'open');
    // Without it the image's lock is read, which is not here: an empty lock, so the runner is not installable.
    const image = await check(['open', '--validate-only']);
    if (!fs.existsSync('/opt/local-llm/runners.lock.json')) {
        assert.equal(image.code, 1);
        assert.equal(image.report.error, 'not_installable');
    }
});
