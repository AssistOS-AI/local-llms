// Time budget of a drain (SIGTERM, SIGINT or SIGHUP). Ploinky's targeted
// restart waits 35 s; the controller gives up at DRAIN_DEADLINE_MS and exits 1.
// The worst case of the stages below and the engine's own stages
// (DRAIN_QUEUE_WAIT_MS and DRAIN_RUNNER_GRACE_MS of pworker's
// lib/local-llm/drainBudget.mjs) leaves at least 5 s under that deadline.
export const DRAIN_DEADLINE_MS = 30_000;
// AgentServer's own shutdown waits up to 20 s for in-flight tool calls
// (TaskQueue.shutdown); it is killed only after that.
export const AGENT_SERVER_SHUTDOWN_MS = 20_000;
export const AGENT_SERVER_KILL_MS = 21_000;
