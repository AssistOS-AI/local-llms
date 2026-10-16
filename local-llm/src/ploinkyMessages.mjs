// The Ploinky wording of the engine's messages (DS002, DS003, DS005): how an
// operator turns a switch on, what to restart and where to look for the GPU
// grant, byte for byte as the agent always worded them. The engine's own
// defaults are neutral; `ploinky-wording.test.mjs` pins every sentence to the
// text it replaced.

import { defineMessages } from 'pworker/local-llm';

export const PLOINKY_MESSAGES = defineMessages({
    notReady: () => 'No local model is ready. Run one from Settings > Agents > Local LLMs.',
    gpuAccessReported: (reason) => `Ploinky reports: ${reason || 'GPU access is unavailable; the cause is unknown. Check host `ploinky gpu status`.'}`,
    gpuAbsent: () => 'No GPU is available to this agent. On the host, `ploinky gpu status` shows why: '
        + 'revoked: run `ploinky gpu grant --agent local-llms/local-llm`; '
        + 'not applied yet: run `ploinky start`; or the host has no usable GPU.',
    hfTokenHint: () => 'set a token with `ploinky var HF_TOKEN <token>` and restart local-llm',
    vulkanRuntimeMissing: () => 'A GPU render node is attached, but this image lacks a supported declared Vulkan runtime; update the local-llm image.',
    apuSwitchOff: (deviceName, switchName, switchValue) => `An AMD integrated GPU (${deviceName}) is attached. `
        + `APU use is off: the operator enables it with ploinky var ${switchName} ${switchValue} and a restart of local-llm.`,
    vllmUnifiedSwitchHint: (switchName, switchValue) => `The operator can turn it on with ploinky var ${switchName} ${switchValue}; `
        + 'it has not been measured on this hardware.',
    lmStudioEnableHint: (switchName, switchValue) => `This deployment's operator enables it with "ploinky var ${switchName} ${switchValue}" and a restart of local-llm.`,
    lmStudioStagedUpdate: () => 'LM Studio has staged an update in its home; restart local-llm to rebuild LM Studio from its verified download.',
    profileRestartGpuRecovered: () => 'GPU access is available now; restart local-llm to select it. Stopping and starting the model does not change the compute profile.',
    profileRestartChanged: () => 'After changing GPU access, machine or provider, restart the agent or Box through Ploinky to select a new compute profile.',
    restartToUseDevice: (profile) => `This agent started with the ${profile} profile; restart local-llm to use the current device or run on the CPU.`,
    restartToRunOnCpu: (profile) => `This agent started with the ${profile} profile; restart local-llm to run on the CPU.`,
    restartForMemoryModel: (current, selected) => `The GPU now reports ${current} memory, but this agent started with the ${selected} profile; restart local-llm to use it.`,
});
