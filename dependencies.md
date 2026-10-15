# Hardware budget dependencies

The hardware budget implementation adds no third-party dependency. It uses Node.js 24 ES modules, native cryptographic hashing, the local budget and qualification modules, and the pinned pworker engine package (below). Its tests use node:test and injected hardware, runner processes and reviewed-data fixtures. Beyond the one-time install of the engine package, no GPU, container, network download or package installation is needed for those tests.

Venus metadata, diagnostics and prompt/startup observations also use only Node.js built-ins and local modules. SSE decoding uses the built-in TextDecoder and ReadableStream APIs. The existing dd per-file cache preparation remains optional and best-effort; observation counts add no cache utility or system operation.

Ploinky continues to supply the shared AchillesAgentLib runtime. Existing GPU queries, runner wheels, binary pins, licenses and installation checks remain governed by local-llm/docs/specs/DS003-gpu-and-resources.md and DS004-on-demand-runners.md. No alternate AgentLib source or new Python code is introduced.

# Engine runtime dependency

The agent depends on one package, `pworker` (the Ploinky-Worker repository), which holds the engine that runs the models (`lib/local-llm`; local-llm/docs/specs/DS000-vision-and-scope.md). It is the only entry in `local-llm/package.json` `dependencies`, a runtime dependency of the agent, declared as `git+https://github.com/AssistOS-AI/Ploinky-Worker.git#4668a3f0e2cae0c85463945def068741dc8c2b68`: the HTTPS form with a full 40-hex commit SHA, never a branch, a tag or an abbreviated SHA. `local-llm/package-lock.json` records the same URL as `resolved`; npm writes `git+ssh://` there when it installs, and the lock is kept at the `git+https` form by hand. Its engine requirement is Node.js 22.13 or later, and the agent image runs Node.js 24.

The package is licensed GPL-3.0 (its `LICENSE` file; its `package.json` has no `license` field), which the owner accepted for the code that moved there from this repository.

The package declares `"dependencies": {}` at the pinned commit, so the install adds this one package and no transitive tree: `package-lock.json` holds exactly the root and `node_modules/pworker`. `local-llm/.npmrc` sets `bin-links=false` and `ignore-scripts=true`, so `npm --prefix local-llm ci` and every other `npm --prefix local-llm` call creates no symbolic links and runs no scripts. Ploinky's own dependency install for a running agent uses its own npm policy (`npm install --no-package-lock`), so there the `package.json` pin decides.

Because `pworker` is a git dependency, installing it needs the network and `git` (GitHub). A cold agent start, with an empty Ploinky dependency cache, therefore needs GitHub; with GitHub blocked and a cold cache the agent fails with a Ploinky dependency error and never starts partially (owner decision U12; local-llm/docs/specs/DS001-agent-contract.md, "Cold start needs GitHub"). The `runner-lock-check` workflow and the tests do the same install first (`npm --prefix local-llm ci`); the tests after it need no network. A change of the pin re-runs the whole agent suite, the overview goldens, the Ploinky wording test and `runner-lock-check`.
