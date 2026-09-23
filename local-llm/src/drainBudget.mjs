// Time budget of a drain (SIGTERM, SIGINT or SIGHUP). Ploinky's targeted
// restart waits 35 s; the controller gives up at DRAIN_DEADLINE_MS and exits 1.
// The worst case of the stages below leaves at least 5 s under that deadline.
export const DRAIN_DEADLINE_MS = 30_000;
// A command already in progress may finish before the drain goes on.
export const DRAIN_QUEUE_WAIT_MS = 1_000;
// SIGTERM to SIGKILL for the runner during a drain.
export const DRAIN_RUNNER_GRACE_MS = 3_000;
// AgentServer's own shutdown waits up to 20 s for in-flight tool calls
// (TaskQueue.shutdown); it is killed only after that.
export const AGENT_SERVER_SHUTDOWN_MS = 20_000;
export const AGENT_SERVER_KILL_MS = 21_000;
