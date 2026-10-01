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

// On the cpu profile llama.cpp logs its buffers under CPU names ("CPU_Mapped model buffer
// size", "CPU KV buffer size", "CPU compute buffer size"). They are read only when the
// run reports no CUDA buffer, so a GPU run's report is what it always was, even though
// its offloaded layers leave CPU lines of their own.
const CPU_BUFFER_PATTERNS = Object.freeze([
    ['modelMiB', /\bCPU(?:_[A-Za-z0-9]+)? model buffer size =\s+([\d.]+) MiB/],
    ['kvMiB', /\bCPU(?:_[A-Za-z0-9]+)? KV buffer size =\s+([\d.]+) MiB/],
    ['computeMiB', /\bCPU(?:_[A-Za-z0-9]+)? compute buffer size =\s+([\d.]+) MiB/],
]);

/**
 * What the runner itself reports about its GPU or CPU use, from its log lines.
 * `profile` is the deployment's: on `cpu` the CPU lines are read whenever no
 * CUDA device or buffer was reported; without it they are read only when the
 * log shows no CUDA backend either, so a GPU run that has not yet logged its
 * device never shows a CPU.
 */
export function parseRunnerReport(lines, { profile = null } = {}) {
    const report = { modelMiB: null, kvMiB: null, computeMiB: null, offloaded: null, device: null };
    const sums = { kvMiB: 0 };
    const cpu = { modelMiB: null, kvMiB: null, computeMiB: null };
    let cpuBackend = null;
    let cpuDevice = false;
    let cudaBackend = false;
    for (const { line } of lines) {
        for (const [key, pattern] of CPU_BUFFER_PATTERNS) {
            const match = pattern.exec(line);
            if (!match) continue;
            // Weights and KV can sit in more than one CPU buffer (mapped, repacked); the compute buffer is one. Sizes are logged
            // with two decimals, so a sum is rounded to two (325.02 + 208.30 is 533.32, not 533.3199999999999).
            cpu[key] = key === 'computeMiB' ? Number(match[1]) : Math.round(((cpu[key] ?? 0) + Number(match[1])) * 100) / 100;
        }
        // "load_backend: loaded CPU backend from /opt/llama.cpp/libggml-cpu-armv8.2_2.so": the variant llama.cpp picked.
        const backend = /load_backend: loaded CPU backend from \S*libggml-cpu-([A-Za-z0-9._+-]+)\.so/.exec(line);
        if (backend) cpuBackend = `CPU (${backend[1]})`;
        else if (/load_backend: loaded CPU backend from \S*libggml-cpu\.so/.test(line)) cpuBackend = 'CPU';
        // b11159 logs no load_backend line for the CPU; its device list names the CPU instead:
        // "common_param:   - CPU     : CPU (5894 MiB, 5894 MiB free)". No instruction-set variant is logged, so none is claimed.
        if (/common_param:\s+-\s+CPU\s+:\s+CPU\b/.test(line)) cpuDevice = true;
        if (/load_backend: loaded CUDA backend/.test(line)) cudaBackend = true;
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
    // A run with no CUDA device or buffer is on the CPU (a CUDA build loads its CPU backend too, so that line alone says nothing).
    if ((profile === 'cpu' || !cudaBackend) && report.device === null && ['modelMiB', 'kvMiB', 'computeMiB'].every((key) => report[key] === null)) {
        Object.assign(report, cpu);
        // The device list also names the CPU beside a GPU, so it says "CPU" only where the deployment is on the cpu profile.
        report.device = cpuBackend ?? (profile === 'cpu' && cpuDevice ? 'CPU' : null);
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
/**
 * Make a runner the kernel OOM killer's first choice. Its memory on an
 * integrated GPU is invisible to RSS, the OOM score and memory cgroups (DS005),
 * so without this an out-of-memory kill would pick another process. Raising
 * the value needs no privilege, and children the runner starts inherit it.
 */
export function setOomScoreAdj(pid, value = 1000, { fsApi = fs } = {}) {
    fsApi.writeFileSync(`/proc/${pid}/oom_score_adj`, String(value));
}

export function startRunnerProcess({
    command, args, env, cwd = '/', log, filter = null, spawnImpl = spawn, killImpl = process.kill,
    oomScoreAdj = 1000, setOomScore = setOomScoreAdj,
}) {
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
    // Right after the spawn, before the runner can load anything (CUDA start-up
    // alone takes longer than this write). A runner that cannot be marked is
    // killed: it must never run as an ordinary OOM candidate.
    let oomError = null;
    if (oomScoreAdj !== null && Number.isInteger(child.pid)) {
        try {
            setOomScore(child.pid, oomScoreAdj);
        } catch (error) {
            // A runner that already exited has nothing left to protect.
            if (!['ENOENT', 'ESRCH'].includes(error?.code)) oomError = error;
        }
    }
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
    if (oomError) {
        log.append('controller', `runner killed: its OOM score could not be set (${oomError.code || oomError.message})`);
        signalGroup('SIGKILL');
    }

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

    // An immediate stop of the whole group, for the unified memory guard: a
    // runner starved of memory has nothing to save, and waiting lets it take more.
    function kill() {
        if (!exitResult) signalGroup('SIGKILL');
        return exited;
    }

    return Object.freeze({
        pid: child.pid,
        exited,
        stop,
        kill,
        get running() { return exitResult === null; },
    });
}
