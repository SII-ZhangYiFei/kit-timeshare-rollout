# Changelog

## 0.1.0

Initial public rollout adapter, extracted from private diagnostic experiments.

- Installable wheel with no automatic simulator or dataset download.
- Caller-supplied policy actions and native task rewards/termination.
- Independent fixed-topology reset; no global TaskEnv.reset during rollout.
- Independent physics scenes, task/controller routing, clocks, and RNG streams.
- Explicit time limits, no automatic reset, optional inactive-state validation.
- GPU UUID/PCI/device mapping and strict Vulkan device isolation checks.
- Pinned compatibility profiles and local runtime hooks instead of SDK edits.
- Policy example, native acceptance harness, CPU-only regression tests and CI.
- Diagnostic sinusoidal actions, object lifts and test velocities are absent from
  the default rollout interface.

The historical experiment archive is separate. Its videos and capacity results
were produced by the earlier harness and are not retested release guarantees.
