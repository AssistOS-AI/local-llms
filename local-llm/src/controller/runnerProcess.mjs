// One supervised runner process (llama-server or ollama serve): spawned from
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
    ['modelMiB', /CUDA0 model buffer size =\s+([\d.]+) MiB/],
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
    }
    const known = ['modelMiB', 'kvMiB', 'computeMiB'].map((key) => report[key]).filter((value) => value !== null);
    report.totalMiB = known.length ? Math.round(known.reduce((total, value) => total + value, 0)) : null;
    return report;
}

// Runner output arrives in arbitrary chunks. Only whole lines are logged, so a
// line split across chunks (or a multi-byte character) still parses; a line
// longer than the bound is logged in pieces rather than held forever.
const MAX_PENDING_LINE = 64 * 1024;

function wholeLines(emit) {
    const decoder = new StringDecoder('utf8');
    let pending = '';
    return {
        push(chunk) {
            pending += decoder.write(chunk);
            const newline = pending.lastIndexOf('\n');
            if (newline >= 0) {
                emit(pending.slice(0, newline));
                pending = pending.slice(newline + 1);
            }
            if (pending.length > MAX_PENDING_LINE) {
                emit(pending);
                pending = '';
            }
        },
        flush() {
            pending += decoder.end();
            if (pending) emit(pending);
            pending = '';
        },
    };
}

export function startRunnerProcess({ command, args, env, cwd = '/', log, spawnImpl = spawn }) {
    if (!Array.isArray(args) || args.some((value) => typeof value !== 'string')) {
        throw new Error('Runner arguments must be an array of strings');
    }
    const child = spawnImpl(command, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    const exited = new Promise((resolve) => {
        child.once('error', (error) => {
            log.append('controller', `runner failed to start: ${error.message}`);
            resolve({ code: null, signal: null, error });
        });
        child.once('exit', (code, signal) => resolve({ code, signal, error: null }));
    });
    for (const [name, stream] of [['stdout', child.stdout], ['stderr', child.stderr]]) {
        if (!stream) continue;
        const lines = wholeLines((text) => log.append(name, text));
        stream.on('data', (chunk) => lines.push(chunk));
        stream.on('end', () => lines.flush());
    }
    let exitResult = null;
    exited.then((result) => { exitResult = result; });

    async function stop({ graceMs = 10_000 } = {}) {
        if (exitResult) return exitResult;
        try { child.kill('SIGTERM'); } catch {}
        const timer = new Promise((resolve) => setTimeout(resolve, graceMs, 'timeout'));
        if (await Promise.race([exited, timer]) === 'timeout') {
            log.append('controller', `runner did not exit within ${graceMs} ms; sending SIGKILL`);
            try { child.kill('SIGKILL'); } catch {}
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
