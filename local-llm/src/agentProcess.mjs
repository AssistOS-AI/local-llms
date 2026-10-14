// The local-llm agent process (DS001): the controller, its container-local
// control socket, and the AgentServer it spawns. `main.mjs` starts it with the
// production options; a test entry may pass controller options (a loopback
// transport, a fake Hugging Face base URL), which nothing in the environment
// can supply.
//
// The registry credentials key (LOCAL_LLM_CREDENTIALS_KEY, a Ploinky generated
// secret) is read once and deleted from this process's environment before
// anything is spawned, then passed explicitly to the controller and the
// registry service. AgentServer, which gets a copy of the environment, and
// every child spawned without an explicit environment therefore never see it.
//
// Modelled on AssistOSExplorer/userPersistoAgent/main.mjs. It owns downloads
// and runner processes, serves the container-local control socket, and
// spawns /Agent/server/AgentServer.mjs directly (not AgentServer.sh, whose
// restart loop would hide an MCP crash). On SIGTERM it drains within the 35 s
// targeted-restart budget and exits 0 only after a clean drain.

import { spawn } from 'node:child_process';
import { join } from 'node:path';

import { controllerHandlers } from './controlHandlers.mjs';
import { newControlChannel, startControlServer } from './controlSocket.mjs';
import { createController } from './controller/deployments.mjs';
import { createRegistryService } from './controller/registries.mjs';
import { AGENT_SERVER_KILL_MS, DRAIN_DEADLINE_MS } from './drainBudget.mjs';


/**
 * Starts the agent once per process. `controllerOptions` are merged over the production controller options. The
 * other options are test seams for the process's collaborators; production uses the real ones.
 */
export function startAgent({
    controllerOptions = {},
    makeController = createController,
    makeRegistryService = createRegistryService,
    spawnProcess = spawn,
    startServer = startControlServer,
} = {}) {
    // N4: the key is taken out of the environment before anything else runs, and travels only as an option.
    const credentialsKey = process.env.LOCAL_LLM_CREDENTIALS_KEY ?? null;
    delete process.env.LOCAL_LLM_CREDENTIALS_KEY;
    // The control channel is new on every start: an abstract socket name (tests
    // may choose it) and a token that only AgentServer receives, through its
    // environment. A token in this process's own environment is never used and
    // is dropped, so nothing this process spawns can inherit one.
    const channel = newControlChannel();
    const socketPath = process.env.LOCAL_LLM_SOCKET || channel.socketPath;
    delete process.env.LOCAL_LLM_CONTROL_TOKEN;

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
        const options = { dataDir: process.env.LOCAL_LLM_DATA_DIR || '/data', ...controllerOptions };
        const registryService = makeRegistryService({ dataDir: options.dataDir, credentialsKey,
            ...(options.hfBaseUrl ? { hfBaseUrl: options.hfBaseUrl } : {}) });
        controller = makeController({ ...options, credentialsKey, registryService });
        control = await startServer({
            socketPath,
            token: channel.token,
            handlers: controllerHandlers(controller),
        });
        if (stopping) return;
        // The generic runtime owns MCP, invocation verification and tool dispatch.
        agentServer = spawnProcess(process.execPath, [
            join(process.env.PLOINKY_AGENT_LIB_DIR || '/Agent', 'server', 'AgentServer.mjs'),
        ], {
            env: { ...process.env, LOCAL_LLM_SOCKET: socketPath, LOCAL_LLM_CONTROL_TOKEN: channel.token },
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
}
