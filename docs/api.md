# API and current limits

`make(BehaviorConfig(...))` and `make(RoboDojoConfig(...))` return `RolloutPool`.
Use a new empty `output_dir` for every run. Construction loads the original
upstream task configuration and assets. Configs are frozen dataclasses; inspect
`config.py` for all fields and their defaults.

`step(env_id, action)` returns `StepResult(observation, reward, terminated,
truncated, info)`, unpackable as a normal Gymnasium-style five-tuple. No Gym
registration or action-space conversion layer is promised. BEHAVIOR accepts the
native action vector or dict accepted by that environment. RoboDojo accepts native
`MetaControl` or its dictionary; the default advances ten upstream control
substeps per call. Set `control_substeps` explicitly for another control rate.

`observe(env_id)` renders without intentionally advancing physics. `reset(env_id,
seed=None)` returns `(observation, info)`. `snapshot()` is a diagnostic, not a
portable save/load format. `close()` is idempotent; the context manager calls it.

Task termination is upstream termination. `max_episode_steps` optionally adds a
wrapper time limit in **adapter calls**, not physics substeps. There is no auto
reset. Stepping a finished episode raises `EpisodeDoneError` until reset.

BEHAVIOR reset loads the selected scene's initial state without global simulator
restore, then executes the selected upstream task reset. It requires unchanged
object membership. RoboDojo restores initial rigid-body and articulation state,
selected reward/parser state, controller targets, and lifecycle flags. It does
not run global TaskEnv.reset, reload layouts, create objects, or resample assets.
Articulation/solver warm-start caches are retained; bitwise identical trajectories
after reset are not guaranteed.

RNG isolation covers Python, NumPy, and Torch during selected native operations;
construction and shader compilation are not deterministic replay. Do not run
concurrent stochastic code using those global generators from other threads;
use policy-owned generators. Kit access must stay on the creating thread.

The pinned RoboDojo profile requires implicit joint actuators, fixed topology,
and no interval events. Other reward functions, cameras, custom callbacks,
particle/fluid state, deformables, and topology-changing tasks need separate
validation. Native PhysX contacts and renderer caches are retained, not exported.
A snapshot comparison covers the recorded state, not every hidden native cache.

Version checks fail closed by default. `strict_versions=False` explicitly opts
into an unverified SDK version; source-shape invariants still apply. RoboDojo's
URDF importer dependency is adjusted to 2.4.30 in a private experience copy for
the pinned Isaac Sim profile. This does not edit the installed Isaac Lab file.

Shared-renderer CPU physics can change performance and numerical behavior from
upstream GPU physics. Validate policy scores and required metrics separately
before comparing them with a benchmark leaderboard.
