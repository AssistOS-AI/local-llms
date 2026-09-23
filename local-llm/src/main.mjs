// local-llm controller: the container's main process.
//
// Modelled on AssistOSExplorer/userPersistoAgent/main.mjs. It owns downloads
// and runner processes, serves the container-local control socket, and
// spawns /Agent/server/AgentServer.mjs directly (not AgentServer.sh, whose
// restart loop would hide an MCP crash). On SIGTERM it drains within the 35 s
// targeted-restart budget and exits 0 only after a clean drain.

import { spawn } from 'node:child_process';
import { join } from 'node:path';

import { DEFAULT_SOCKET, startControlServer } from './controlSocket.mjs';
import { createController } from './controller/deployments.mjs';

const DRAIN_DEADLINE_MS = 30_000;
const AGENT_SERVER_KILL_MS = 15_000;
const socketPath = process.env.LOCAL_LLM_SOCKET || DEFAULT_SOCKET;

let controller;
let control;
let agentServer;
let agentServerExit;
let stopping = false;
let shutdownPromise;
let exitCode = 0;

function log(message) {
    process.stderr.write(`[local-llm] ${message}\n`);
}

function shutdown(code = 0) {
    if (code !== 0) exitCode = 1;
    if (shutdownPromise) return shutdownPromise;
    stopping = true;
    shutdownPromise = (async () => {
        const deadline = setTimeout(() => {
            log('drain timed out; exiting 1.');
            process.exit(1);
        }, DRAIN_DEADLINE_MS);
        try {
            await startup.catch(() => {});
            // Stop admitting commands, checkpoint any download, stop the runner.
            if (controller) await controller.drain();
            if (control) await control.close();
            if (agentServer) {
                const killTimer = setTimeout(() => {
                    exitCode = 1;
                    agentServer.kill('SIGKILL');
                }, AGENT_SERVER_KILL_MS);
                try {
                    agentServer.kill('SIGTERM');
                    const result = await agentServerExit;
                    if (result.code !== 0 && result.signal !== 'SIGTERM') exitCode = 1;
                } finally {
                    clearTimeout(killTimer);
                }
            }
        } catch (error) {
            exitCode = 1;
            log(`drain failed: ${error.message}`);
        } finally {
            clearTimeout(deadline);
            process.exit(exitCode);
        }
    })();
    return shutdownPromise;
}

for (const signal of ['SIGTERM', 'SIGINT', 'SIGHUP']) {
    process.on(signal, () => { void shutdown(); });
}

const startup = (async () => {
    controller = createController({ dataDir: process.env.LOCAL_LLM_DATA_DIR || '/data' });
    control = await startControlServer({
        socketPath,
        handlers: {
            overview: (args) => controller.overview(args),
            status: (args) => controller.status(args),
            run: (args) => controller.run(args),
            stop: () => controller.stop(),
            cancelDownload: () => controller.cancelDownload(),
            deleteWeights: (args) => controller.deleteWeights(args),
            addModel: (args) => controller.addModel(args.model),
            updateModel: (args) => controller.updateModel(args.model),
            removeModel: (args) => controller.removeModel(args),
            chatTarget: () => controller.chatTarget(),
            recordCompletion: (args) => controller.recordCompletion(args),
        },
    });
    if (stopping) return;
    // The generic runtime owns MCP, invocation verification and tool dispatch.
    agentServer = spawn(process.execPath, [
        join(process.env.PLOINKY_AGENT_LIB_DIR || '/Agent', 'server', 'AgentServer.mjs'),
    ], {
        env: { ...process.env, LOCAL_LLM_SOCKET: socketPath },
        stdio: 'inherit',
    });
    agentServerExit = new Promise((resolve) => {
        agentServer.once('error', () => resolve({ code: 1, signal: null }));
        agentServer.once('exit', (code, signal) => resolve({ code, signal }));
    });
    void agentServerExit.then(() => {
        if (!stopping) {
            log('AgentServer exited unexpectedly; shutting down so the container restarts.');
            void shutdown(1);
        }
    });
    log(`controller ready on ${socketPath}`);
})();

void startup.catch((error) => {
    log(`startup failed: ${error.message}`);
    void shutdown(1);
});
