#!/usr/bin/env node
// Install check for one on-demand runner, run in CI inside the published
// image (runners plan §5.2, "CI install check"). It installs the lock entry
// exactly as the agent would (download, verify, rebuild the runnable copy
// offline), then checks it without a GPU:
//   - every distribution in the lock's check has its pinned version;
//   - every module in check.imports imports; modules in check.gpuImports
//     are tried and listed, since they may need the NVIDIA driver;
//   - every shared object in the runnable copy resolves its libraries. A
//     library ldd cannot find passes only when the Box GPU grant supplies it
//     (libcuda.so.1), when another wheel in the same environment ships it
//     (loaded at run time, as torch loads its CUDA libraries), or when the
//     lock lists it in check.optionalLibraries with the reason the runner
//     never needs it. Anything else fails;
//   - every pinned data file (`into`) is in the runnable copy with its bytes.
// Usage: node runner_install_check.mjs <runnerId> [--cache DIR] [--lock FILE]
// Prints one JSON report; exits 1 if any check fails.

import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { argv, exit, stdout } from 'node:process';

import { createRunnerInstaller } from '../src/controller/runnerInstaller.mjs';
import { loadRunnerLock } from '../src/controller/runnerLock.mjs';

const IMPORT_TIMEOUT_MS = 600_000;
const DRIVER_LIBRARY = 'libcuda.so.1';
const SHARED_OBJECT_RE = /\.so(\.\d+)*$/;

function option(name, fallback) {
    const index = argv.indexOf(name);
    return index > 0 && argv[index + 1] ? argv[index + 1] : fallback;
}

// Every shared object in the tree: regular files to check, and the names of
// all of them (symlinks included) as the libraries the environment provides.
function sharedObjects(dir) {
    const files = [];
    const names = new Set();
    const walk = (current) => {
        for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
            const full = path.join(current, entry.name);
            if (entry.isDirectory()) walk(full);
            else if (SHARED_OBJECT_RE.test(entry.name)) {
                names.add(entry.name);
                if (entry.isFile()) files.push(full);
            }
        }
    };
    walk(dir);
    return { files: files.sort(), names };
}

function missingLibraries(file) {
    const result = spawnSync('ldd', [file], { encoding: 'utf8', env: { PATH: '/usr/bin:/bin', LD_LIBRARY_PATH: '/usr/local/nvidia/lib64' } });
    const output = `${result.stdout || ''}${result.stderr || ''}`;
    if (/not a dynamic executable/.test(output)) return [];
    return output.split('\n').filter((line) => line.includes('not found')).map((line) => line.trim().split(/\s+/)[0]);
}

function globRegExp(pattern) {
    return new RegExp(`^${pattern.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*')}$`);
}

/**
 * Sort ldd's missing libraries: supplied by the driver grant, shipped by
 * another wheel in the environment, optional by the lock (with its reason),
 * or unresolved, which fails the check.
 */
export function classifyMissingLibraries({ missing, provided, optional = {} }) {
    const patterns = Object.entries(optional).map(([pattern, reason]) => [globRegExp(pattern), reason]);
    const unresolved = {};
    const optionalFound = {};
    const providedFound = new Set();
    for (const [file, libraries] of Object.entries(missing)) {
        for (const library of libraries) {
            if (library === DRIVER_LIBRARY) continue;
            if (provided.has(library)) {
                providedFound.add(library);
                continue;
            }
            const match = patterns.find(([regexp]) => regexp.test(library));
            if (match) {
                optionalFound[library] = match[1];
                continue;
            }
            (unresolved[file] ||= []).push(library);
        }
    }
    return { unresolved, optional: optionalFound, providedByEnvironment: [...providedFound].sort() };
}

/** Each pinned data file (`into`) must be in the runnable copy with the lock's bytes. */
export function checkDataFiles(entry, runDir) {
    const placed = {};
    const problems = [];
    for (const file of entry.files.filter((candidate) => candidate.into)) {
        const relative = path.join(file.into, file.name);
        let digest = null;
        try {
            digest = crypto.createHash('sha256').update(fs.readFileSync(path.join(runDir, relative))).digest('hex');
        } catch {}
        placed[relative] = digest === null ? 'missing' : (digest === file.sha256 ? 'ok' : 'differs');
        if (placed[relative] !== 'ok') problems.push(`${relative} ${digest === null ? 'is missing' : 'differs from the lock'}`);
    }
    return { placed, problems };
}

// Written in full before the process exits, even to a pipe.
function finish(report, code) {
    stdout.write(`${JSON.stringify(report, null, 2)}\n`, () => exit(code));
}

async function main() {
    const runnerId = argv[2];
    const lock = loadRunnerLock(option('--lock', undefined));
    const installer = createRunnerInstaller({ lock, cacheRoot: option('--cache', '/tmp/runner-check/cache') });
    const entry = installer.entryFor(runnerId);
    const report = { runnerId, version: entry.version, files: entry.files.length, totalBytes: entry.totalBytes, ok: true, problems: [] };
    const started = Date.now();
    const fetched = await installer.fetchAll(entry);
    report.downloadSeconds = (Date.now() - started) / 1000;
    report.transferredBytes = fetched.transferred;
    const built = await installer.ensureRunnable(runnerId);
    report.rebuild = built;
    const runDir = installer.pathsFor(entry).runDir;
    if (entry.kind === 'python') {
        const python = path.join(runDir, 'venv', 'bin', 'python');
        const names = Object.keys(entry.check.distributions);
        const versions = spawnSync(python, ['-c', 'import importlib.metadata as m, json, sys; print(json.dumps({d: m.version(d) for d in sys.argv[1:]}))', ...names],
            { encoding: 'utf8', timeout: IMPORT_TIMEOUT_MS, env: { PATH: '/usr/bin:/bin', HOME: '/tmp' } });
        report.distributions = versions.status === 0 ? JSON.parse(versions.stdout) : { error: versions.stderr.trim().split('\n').at(-1) };
        for (const [name, expected] of Object.entries(entry.check.distributions)) {
            if (report.distributions[name] !== expected) report.problems.push(`${name} is ${report.distributions[name] ?? 'missing'}, not ${expected}`);
        }
        report.imports = {};
        for (const module of [...entry.check.imports, ...entry.check.gpuImports]) {
            const result = spawnSync(python, ['-c', `import ${module}`], { encoding: 'utf8', timeout: IMPORT_TIMEOUT_MS,
                env: { PATH: '/usr/bin:/bin', HOME: '/tmp', LD_LIBRARY_PATH: '/usr/local/nvidia/lib64' } });
            const ok = result.status === 0;
            const gpuOnly = entry.check.gpuImports.includes(module);
            report.imports[module] = ok ? 'ok' : `${gpuOnly ? 'needs a GPU' : 'failed'}: ${(result.stderr || '').trim().split('\n').at(-1)}`;
            if (!ok && !gpuOnly) report.problems.push(`import ${module} failed`);
        }
    }
    const data = checkDataFiles(entry, runDir);
    report.dataFiles = data.placed;
    report.problems.push(...data.problems);
    const objects = sharedObjects(runDir);
    report.sharedObjects = objects.files.length;
    const missing = {};
    for (const file of objects.files) {
        const libraries = missingLibraries(file);
        if (libraries.length) missing[path.relative(runDir, file)] = libraries;
    }
    const links = classifyMissingLibraries({ missing, provided: objects.names, optional: entry.check.optionalLibraries });
    report.libraries = links;
    if (Object.keys(links.unresolved).length) {
        report.problems.push(`${Object.keys(links.unresolved).length} shared objects need libraries that are neither in the environment nor optional`);
    }
    report.ok = report.problems.length === 0;
    finish(report, report.ok ? 0 : 1);
}

if (import.meta.url === `file://${argv[1]}`) {
    main().catch((error) => finish({ ok: false, error: error.code || 'error', message: error.message }, 1));
}
