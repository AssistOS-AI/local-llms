// One supervised runner process (a llama-server or ollama serve): spawned from
// an argv array (never a shell), its output kept in a bounded, sequenced log
// with secrets redacted, the CUDA buffers it reports parsed out, and a stop
// that escalates from SIGTERM to SIGKILL.

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { StringDecoder } from 'node:string_decoder';

const LOG_FILE_MAX_BYTES = 20 * 1024 * 1024;

export function createLogBuffer({ limit = 2000, file = null, fsApi = fs } = {}) {
    let seq = 0;
    const lines = [];
    const secrets = new Set();

    function redact(text) {
        let result = text;
        for (const secret of secrets) result = result.split(secret).join('***');
        return result;
    }

    function append(stream, text) {
        for (const raw of String(text).split(/\r?\n/)) {
            if (!raw) continue;
            seq += 1;
            const line = { seq, at: new Date().toISOString(), stream, line: redact(raw).slice(0, 2000) };
            lines.push(line);
            if (lines.length > limit) lines.shift();
            if (file) {
                try {
                    fsApi.mkdirSync(path.dirname(file), { recursive: true });
                    const size = fsApi.existsSync(file) ? fsApi.statSync(file).size : 0;
                    if (size > LOG_FILE_MAX_BYTES) fsApi.renameSync(file, `${file}.1`);
                    fsApi.appendFileSync(file, `${line.at} [${stream}] ${line.line}\n`);
                } catch {
                    // The in-memory log stays authoritative when the file is unavailable.
                }
            }
        }
    }

    return Object.freeze({
        append,
        addSecret(secret) { if (secret) secrets.add(secret); },
        since(after = 0, max = 200) {
            const selected = lines.filter((line) => line.seq > after);
            return selected.slice(-max);
        },
        get seq() { return seq; },
        all: () => [...lines],
    });
}

const BUFFER_PATTERNS = Object.freeze([
    // llama.cpp says "CUDA0 model buffer size", ik_llama.cpp "CUDA0 buffer size".
    ['modelMiB', /CUDA0 (?:model )?buffer size =\s+([\d.]+) MiB/],
    ['kvMiB', /CUDA0 KV buffer size =\s+([\d.]+) MiB/],
    ['computeMiB', /CUDA0 compute buffer size =\s+([\d.]+) MiB/],
]);

/** What the runner itself reports about its GPU use, from its log lines. */
export function parseRunnerReport(lines) {
    const report = { modelMiB: null, kvMiB: null, computeMiB: null, offloaded: null, device: null };
    const sums = { kvMiB: 0 };
    for (const { line } of lines) {
        for (const [key, pattern] of BUFFER_PATTERNS) {
            const match = pattern.exec(line);
            if (!match) continue;
            if (key === 'kvMiB') {
                sums.kvMiB += Number(match[1]);
                report.kvMiB = sums.kvMiB;
            } else {
                report[key] = Number(match[1]);
            }
        }
        const offloaded = /offloaded (\d+)\/(\d+) layers to GPU/.exec(line);
        if (offloaded) report.offloaded = { layers: Number(offloaded[1]), of: Number(offloaded[2]) };
        const device = /using device (CUDA\d+) \(([^)]+)\)/.exec(line);
        if (device) report.device = `${device[1]} (${device[2]})`;
        // ik_llama.cpp names the GPU in its CUDA init lines instead.
        const listed = /^\s*Device (\d+): ([^,]+), compute capability/.exec(line);
        if (listed) report.device = `CUDA${listed[1]} (${listed[2]})`;
    }
    const known = ['modelMiB', 'kvMiB', 'computeMiB'].map((key) => report[key]).filter((value) => value !== null);
    report.totalMiB = known.length ? Math.round(known.reduce((total, value) => total + value, 0)) : null;
    return report;
}

// Runner output arrives in arbitrary chunks. Only whole lines are logged, so a
// line split across chunks (or a multi-byte character) still parses; a line
// longer than the bound is logged in pieces rather than held forever. Each
// piece after the first is emitted as a continuation, so a line filter never
// mistakes one for a line of its own.
const MAX_PENDING_LINE = 64 * 1024;

function wholeLines(emit) {
    const decoder = new StringDecoder('utf8');
    let pending = '';
    let continuing = false;
    const line = (text) => {
        emit(text, continuing);
        continuing = false;
    };
    return {
        push(chunk) {
            pending += decoder.write(chunk);
            let newline = pending.indexOf('\n');
            while (newline >= 0) {
                line(pending.slice(0, newline));
                pending = pending.slice(newline + 1);
                newline = pending.indexOf('\n');
            }
            if (pending.length > MAX_PENDING_LINE) {
                emit(pending, continuing);
                continuing = true;
                pending = '';
            }
        },
        flush() {
            pending += decoder.end();
            if (pending) line(pending);
            pending = '';
        },
    };
}

// `filter`, when given, makes one line filter per output stream: it returns the
// line to log (possibly shortened) or null to drop it. A runner whose output
// echoes request bodies (LM Studio) uses it to keep them out of the log.
export function startRunnerProcess({ command, args, env, cwd = '/', log, filter = null, spawnImpl = spawn, killImpl = process.kill }) {
    if (!Array.isArray(args) || args.some((value) => typeof value !== 'string')) {
        throw new Error('Runner arguments must be an array of strings');
    }
    // One filter per stream, made before anything is spawned: a filter that
    // cannot be made must not leave a runner nobody controls.
    const keepers = { stdout: filter ? filter() : null, stderr: filter ? filter() : null };
    // The runner leads its own process group, so a stop reaches every process
    // it started (worker processes, a model server's runner child), not just
    // the one the controller spawned.
    const child = spawnImpl(command, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'], detached: true });
    function signalGroup(signal) {
        if (!Number.isInteger(child.pid)) return;
        try {
            killImpl(-child.pid, signal);
            return;
        } catch {
            // No such group (a spawn that could not make the runner a group
            // leader): signal the runner itself.
        }
        try { child.kill(signal); } catch {}
    }
    const exited = new Promise((resolve) => {
        child.once('error', (error) => {
            log.append('controller', `runner failed to start: ${error.message}`);
            resolve({ code: null, signal: null, error });
        });
        child.once('exit', (code, signal) => {
            // Anything the runner left behind in its group goes with it.
            signalGroup('SIGKILL');
            resolve({ code, signal, error: null });
        });
    });
    for (const [name, stream] of [['stdout', child.stdout], ['stderr', child.stderr]]) {
        if (!stream) continue;
        const keep = keepers[name];
        const lines = wholeLines((text, continued) => {
            if (!keep) {
                log.append(name, text);
                return;
            }
            // The rest of an over-long line is never logged through a filter.
            if (continued) return;
            const kept = keep(text.replace(/\r$/, ''));
            if (typeof kept === 'string' && kept) log.append(name, kept);
        });
        stream.on('data', (chunk) => lines.push(chunk));
        stream.on('end', () => lines.flush());
    }
    let exitResult = null;
    exited.then((result) => { exitResult = result; });

    async function stop({ graceMs = 10_000 } = {}) {
        if (exitResult) return exitResult;
        signalGroup('SIGTERM');
        let timeout;
        const timer = new Promise((resolve) => { timeout = setTimeout(resolve, graceMs, 'timeout'); });
        const outcome = await Promise.race([exited, timer]);
        clearTimeout(timeout);
        if (outcome === 'timeout') {
            log.append('controller', `runner did not exit within ${graceMs} ms; sending SIGKILL`);
            signalGroup('SIGKILL');
        }
        return exited;
    }

    return Object.freeze({
        pid: child.pid,
        exited,
        stop,
        get running() { return exitResult === null; },
    });
}
