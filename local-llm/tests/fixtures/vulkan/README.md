`venus-summary.txt` and `venus-devices.txt` are unmodified stdout captures from `vulkaninfo --summary` and `llama-server --list-devices` on 2026-10-12. The machine was `ploinky-gpu`, a rootless libkrun Podman machine using krunkit 1.3.2 on Apple M1, with Mesa 25.3.6 and llama.cpp b11159. The captures report `Virtio-GPU Venus (Apple M1)` and 16384 MiB total and free memory. These heap figures do not measure available host memory or change the provisional Venus admission cap.

All other fixtures are synthetic. RADV remains unverified on hardware. The runner log covers nested parentheses in device names.

`radv-integrated-summary.txt` and `radv-integrated-devices.txt` are matching synthetic APU probes. Their 16384 MiB heap figures are a test mapping ceiling, not a measured physical RAM budget or hardware qualification. The APU tests also synthesize buffer, cancellation and pressure events; real UMA buffer names and memory effects remain pending owner-authorized captures.
