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
//
// The check itself is the engine's (pworker/local-llm/runner-install-check). This entry gives it the
// container's host, so with no --lock it reads the image's lock and it finds the run root, uv, python and
// the driver libraries where the image keeps them (the image publish workflow passes none of them).

import { argv, exit, stdout } from 'node:process';

import { runInstallCheck } from 'pworker/local-llm/runner-install-check';

import { agentHost } from '../src/agentHost.mjs';

// Written in full before the process exits, even to a pipe.
function finish(report, code) {
    stdout.write(`${JSON.stringify(report, null, 2)}\n`, () => exit(code));
}

if (import.meta.url === `file://${argv[1]}`) {
    try {
        const { report, code } = await runInstallCheck(argv.slice(2), { host: agentHost(process.env) });
        finish(report, code);
    } catch (error) {
        finish({ ok: false, error: error.code || 'error', message: error.message }, 1);
    }
}
