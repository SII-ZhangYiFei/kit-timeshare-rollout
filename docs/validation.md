# Native validation — 2026-11-16

Both backends passed using the same candidate wheel on one isolated RTX 5090.
The final runtime source bytes match the tested installation (see
[source SHA256](validation/tested_source_sha256.json)). Publication adds reports
and documentation; simulator modules remain byte-identical.

| Backend | Environments | Rounds / actions | Independent resets | Median step service | Median reset wall |
|---|---:|---:|---:|---:|---:|
| RoboDojo `stack_bowls` | 2 | 3 / 18 | 6 | 155.2 ms | 91.7 ms |
| BEHAVIOR `picking_up_trash` | 2 | 3 / 18 | 6 | 87.2 ms | 382.9 ms |

All 36 selected actions produced physical state changes. All 36 inactive step
checks and 12 inactive reset checks passed. Native nonconstant RGB observations
were verified. The adapter generated no initial object pose/velocity perturbation.
Both tasks returned native reward 0 for these short diagnostic joint actions;
this is not a trained policy success-rate evaluation.

GPU1 worker-access samples and worker swap were zero. Four pre-existing protected
processes retained their original identities. Only the new test containers were
stopped. Neither installed upstream checkout was edited by the package.

Service time excludes snapshot validation; reset wall time includes diagnostic
checks. Startup was 194.0 s / 179.0 s and included
isolated/offline startup and material dependency waits. These are integration
acceptance timings, not controlled throughput measurements or pure switch costs.

The clean Python environment also passed **16 core regression tests** without
Torch or simulator packages. The GitHub workflow is configured for the same core suite, but hosted jobs
could not start due to an account-level runner restriction: zero test steps
executed. Local clean-install tests and the native checks above passed. Native
GPU acceptance is deliberately not run by public CI.

[Machine-readable summary](validation/summary.json) ·
[Per-call CSV](validation/benchmark.csv) ·
[System profile](validation/system_info.json) ·
[RoboDojo raw acceptance](validation/robodojo_acceptance.json) ·
[BEHAVIOR raw acceptance](validation/behavior_acceptance.json) ·
[Method](validation-method.md)

Three failures were fixed before this result: counting cgroup-denied ghost GPU
nodes as exposed devices, missing upstream reward registration after TaskEnv
reset, and reward predicates sharing mutable argument dictionaries across IDs.
Failure logs are retained in the private diagnostic archive. The public summary
contains the causes and fixes without private machine paths or identifiers.

Earlier diagnostic harnesses reached 42 RoboDojo / 31 BEHAVIOR environments with
CPU physics and shared rendering. Those tests used another API; this release
certifies only the two-environment reference cases above. Fixed-topology reset,
reward/controller/event support and policy scores need validation for each new
task. It does not provide VRAM offload or full-process hibernation.
