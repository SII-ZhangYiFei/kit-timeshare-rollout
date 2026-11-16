# Acceptance method

Each backend uses two native environments on one isolated RTX 5090. The harness
runs three rounds of six alternating, caller-supplied actions (18 calls total),
checks that the selected physical state changes, and compares the other complete
recorded physics/task snapshot before and after each call. Both IDs must exhibit
physical state changes; clock-only changes are insufficient. A three-call wrapper
time limit exercises terminal handling and selected reset. Native success/failure
may end an episode earlier. Reset is also checked against the inactive snapshot.

The harness verifies nonconstant native RGB output, absence of imports from the
old experimental harness, CPU physics mode, and no initial state perturbation.
The external controller samples GPU/process membership, free VRAM, host RAM and
worker swap. It exposes only the selected GPU and corresponding DRM render node,
retains a 1 GiB GPU/12 GiB host-RAM reserve, disables worker swap, and can stop
only its newly named container. Existing protected processes remain running.

The controller uses private installation paths and an already available simulator
image solely as test infrastructure. Neither is a dependency of the public
wheel. The public acceptance script requires only the package and upstream SDK.
Isaac Lab's pre-existing local experience edit is hidden by a read-only overlay
of the clean pinned upstream template, so the package's own private experience
compatibility copy is exercised.

These checks establish sampled physical/task isolation for the reference tasks.
They do not establish policy task success, hidden-cache identity, deterministic
reset replay, arbitrary task compatibility, or high-count capacity. Service time
excludes snapshot verification; reset wall time includes its diagnostic checks.
Startup may include offline material/extension timeouts. This is an integration
acceptance run, not a statistically controlled throughput benchmark.
