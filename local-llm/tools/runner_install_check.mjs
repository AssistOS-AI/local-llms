#!/usr/bin/env node
// Install check for one on-demand runner, run in CI inside the published
// image (runners plan §5.2, "CI install check"). It installs the lock entry
// exactly as the agent would (download, verify, rebuild the runnable copy
// offline), then checks it without a GPU:
//   - every distribution in the lock's check has its pinned version;
//   - the environment holds exactly the lock's wheels: every installed
//     distribution is one of them at its version, and none is missing;
//   - every module in check.imports imports; modules in check.gpuImports
//     are tried and listed, since they may need the NVIDIA driver;
//   - every shared object in the runnable copy resolves its libraries. A
//     library ldd cannot find passes only when the Box GPU grant supplies it
//     (libcuda.so.1), when another wheel in the same environment ships it
//     (loaded at run time, as torch loads its CUDA libraries), or when the
//     lock lists it in check.optionalLibraries with the reason the runner
//     never needs it. Anything else fails;
//   - every pinned data file (`into`) is in the runnable copy with its bytes;
//   - the file the adapter launches (`check.executable`) is in the runnable copy,
//     a regular executable file: `ldd` finds every library it needs (the same three
//     groups as above), and `<executable> --version` runs with the CPU environment
//     (no driver library path, no device visible) and names the pinned version.
//     The installer refuses an unpacked copy without it too, since a wrong strip or a
//     truncated archive otherwise unpacks to a copy that looks installed.
// A proprietary entry (LM Studio) is never downloaded here: that would accept
// its terms on behalf of whoever runs CI. It is validated from the lock alone
// (hosts, sizes, sha256 pins and licence fields), as --validate-only does for
// any entry; its install is proven live on a deployment that accepted them.
// Usage: node runner_install_check.mjs <runnerId> [--validate-only] [--require-executable] [--cache DIR] [--lock FILE]
// --require-executable (the agent's lock, in CI) fails an archive entry that declares no check.executable.
// Prints one JSON report; exits 1 if any check fails.

import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { argv, exit, stdout } from 'node:process';

import { assertLaunchExecutable, createRunnerInstaller } from '../src/controller/runnerInstaller.mjs';
import { loadRunnerLock } from '../src/controller/runnerLock.mjs';

const IMPORT_TIMEOUT_MS = 600_000;
const DRIVER_LIBRARY = 'libcuda.so.1';
const SHARED_OBJECT_RE = /\.so(\.\d+)*$/;
const VERSION_TIMEOUT_MS = 60_000;
const VERSION_OUTPUT_KEPT = 400;

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

/**
 * Prove the file the adapter launches: it is in the runnable copy and executable (assertLaunchExecutable),
 * `ldd` finds its libraries (a missing one fails unless the driver grant supplies it, the copy ships it, or the
 * lock lists it as optional), and `--version` runs under the CPU environment and names the pinned version.
 * `provided` is the set of library names the copy ships. Returns the report and the problems found.
 */
export async function probeExecutable({ entry, runDir, provided = new Set(), ldd = missingLibraries, spawn = spawnSync }) {
    const relative = entry.check.executable;
    const report = { path: relative };
    const problems = [];
    let file;
    try {
        file = await assertLaunchExecutable(entry, runDir);
    } catch (error) {
        return { report: { ...report, found: false }, problems: [error.message] };
    }
    report.found = true;
    const links = classifyMissingLibraries({ missing: { [relative]: ldd(file) }, provided, optional: entry.check.optionalLibraries });
    report.libraries = links;
    if (Object.keys(links.unresolved).length) {
        problems.push(`${relative} needs libraries that are neither in the copy, nor supplied by the driver grant, nor optional: ${links.unresolved[relative].join(', ')}`);
    }
    // The CPU environment: no library path for the driver and no device visible, so a binary that needs the GPU to print its version fails here.
    const result = spawn(file, ['--version'], { encoding: 'utf8', timeout: VERSION_TIMEOUT_MS, env: { PATH: '/usr/bin:/bin', HOME: '/tmp', CUDA_VISIBLE_DEVICES: '' } });
    const output = `${result.stdout || ''}${result.stderr || ''}`.trim();
    const wanted = entry.version.replace(/^b(?=\d)/, '');
    report.version = { status: result.error ? (result.error.code || 'error') : result.status, output: output.slice(0, VERSION_OUTPUT_KEPT), expected: wanted };
    if (result.error || result.status !== 0) {
        problems.push(`${relative} --version failed (${result.error?.code || result.error?.message || `exit ${result.status ?? result.signal}`}): ${output.slice(-200)}`);
    } else if (!output.includes(wanted)) {
        problems.push(`${relative} --version does not name ${wanted}: ${output.slice(0, 200)}`);
    }
    return { report, problems };
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

// PEP 503 names: runs of -, _ and . are one dash, case-insensitive.
function canonicalName(name) {
    return String(name).replace(/[-_.]+/g, '-').toLowerCase();
}

/** The distributions a python entry's wheels install: canonical name to version, from each wheel's file name. */
export function lockedDistributions(entry) {
    const locked = {};
    for (const file of entry.files) {
        if (!file.name.endsWith('.whl')) continue;
        const [name, version] = file.name.slice(0, -'.whl'.length).split('-');
        locked[canonicalName(name)] = version;
    }
    return locked;
}

/**
 * Every installed distribution against the lock's wheels, not only the few
 * named in check.distributions: one the lock does not name, one it names at
 * another version, a locked one that is not installed, or one canonical name
 * installed more than once (a stale copy beside the locked one) fails the
 * check. `installed` lists each distribution's metadata name and version, as
 * [name, version] pairs (or an object of them); duplicates are kept.
 */
export function compareInstalledDistributions(entry, installed) {
    const locked = lockedDistributions(entry);
    const pairs = Array.isArray(installed) ? installed : Object.entries(installed);
    const seen = {};
    const copies = {};
    for (const [name, version] of pairs) {
        const canonical = canonicalName(name);
        (copies[canonical] ||= []).push(`${name} ${version}`);
        seen[canonical] = version;
    }
    const duplicated = Object.keys(copies).filter((name) => copies[name].length > 1).sort()
        .map((name) => `${name} (${copies[name].join(', ')})`);
    const extra = Object.keys(seen).filter((name) => !Object.hasOwn(locked, name)).sort();
    const missing = Object.keys(locked).filter((name) => !Object.hasOwn(seen, name)).sort();
    const mismatched = Object.keys(locked)
        .filter((name) => Object.hasOwn(seen, name) && copies[name].length === 1 && seen[name] !== locked[name]).sort()
        .map((name) => `${name} ${seen[name]} (lock ${locked[name]})`);
    const problems = [
        ...(duplicated.length ? [`installed more than once under one name: ${duplicated.join('; ')}`] : []),
        ...(extra.length ? [`installed but not in the lock: ${extra.join(', ')}`] : []),
        ...(missing.length ? [`in the lock but not installed: ${missing.join(', ')}`] : []),
        ...(mismatched.length ? [`installed at another version than the lock: ${mismatched.join(', ')}`] : []),
    ];
    return { locked: Object.keys(locked).length, installed: pairs.length, duplicated, extra, missing, mismatched, problems };
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
    const requireExecutable = argv.includes('--require-executable');
    const lock = loadRunnerLock(option('--lock', undefined), { requireExecutable });
    const installer = createRunnerInstaller({ lock, cacheRoot: option('--cache', '/tmp/runner-check/cache') });
    const entry = installer.entryFor(runnerId);
    const report = { runnerId, version: entry.version, files: entry.files.length, totalBytes: entry.totalBytes, ok: true, problems: [] };
    if (entry.licence.proprietary || argv.includes('--validate-only')) {
        // loadRunnerLock has already validated every entry: https on an
        // allowed host, a positive size and a sha256 per file, the licence fields.
        report.downloaded = false;
        report.skipped = entry.licence.proprietary
            ? `proprietary (${entry.licence.name}): validated from the lock, not downloaded; installing accepts its terms`
            : 'validate-only: validated from the lock, not downloaded';
        report.validated = {
            files: entry.files.length,
            totalBytes: entry.totalBytes,
            hosts: [...new Set(entry.files.map((file) => new URL(file.url).hostname))].sort(),
            licence: { name: entry.licence.name, url: entry.licence.url, requiresAcceptance: entry.licence.requiresAcceptance, proprietary: entry.licence.proprietary },
        };
        finish(report, 0);
        return;
    }
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
        const all = spawnSync(python, ['-c', 'import importlib.metadata as m, json; print(json.dumps([[d.metadata["Name"], d.version] for d in m.distributions()]))'],
            { encoding: 'utf8', timeout: IMPORT_TIMEOUT_MS, env: { PATH: '/usr/bin:/bin', HOME: '/tmp' } });
        if (all.status === 0) {
            report.installedDistributions = compareInstalledDistributions(entry, JSON.parse(all.stdout));
            report.problems.push(...report.installedDistributions.problems);
        } else {
            report.installedDistributions = { error: all.stderr.trim().split('\n').at(-1) };
            report.problems.push('the installed distributions could not be listed');
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
    if (entry.check.executable) {
        const probed = await probeExecutable({ entry, runDir, provided: objects.names });
        report.executable = probed.report;
        report.problems.push(...probed.problems);
    } else if (requireExecutable && entry.kind === 'archive') {
        report.problems.push('the entry declares no check.executable, so nothing proves the binary the adapter launches');
    }
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
