// Test-only process entry (main-drain.test.mjs): the agent process with the loopback test transport and the
// Hugging Face base URL given as the first argument. Production starts through src/main.mjs, which takes neither.
import { startAgent } from '../../src/agentProcess.mjs';
import { createTestTransport } from '../net-fixture.mjs';

const hfBaseUrl = process.argv[2];
startAgent({ controllerOptions: { registryTransport: createTestTransport(), ...(hfBaseUrl ? { hfBaseUrl } : {}) } });
