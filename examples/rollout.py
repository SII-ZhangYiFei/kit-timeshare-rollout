"""Connect your policy without modifying either benchmark's installed source.

python examples/rollout.py behavior --upstream-root /path/to/BEHAVIOR-1K \
  --policy my_policy:act --output-dir runs/new_run --num-envs 2

my_policy.act(observation, env_id) returns a native benchmark action.
"""

import argparse, importlib, json
from pathlib import Path
from kit_timeshare import BehaviorConfig, RoboDojoConfig, make


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=["behavior", "robodojo"])
    p.add_argument("--upstream-root", required=True)
    p.add_argument(
        "--policy",
        required=True,
        help="importable module:callable, called as policy(obs, env_id)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--task", default="stack_bowls")
    p.add_argument("--config-path")
    p.add_argument("--verify-frozen", action="store_true")
    a = p.parse_args()
    # Import the policy AFTER make() selects the GPU and initializes Kit; policy
    # modules may import Torch, but must not create another SimulationApp.
    args = dict(
        upstream_root=a.upstream_root,
        output_dir=a.output_dir,
        num_envs=a.num_envs,
        verify_frozen=a.verify_frozen,
    )
    cfg = (
        BehaviorConfig(**args, config_path=a.config_path)
        if a.backend == "behavior"
        else RoboDojoConfig(**args, task=a.task)
    )
    with make(cfg) as pool:
        module, name = a.policy.split(":", 1)
        policy = getattr(importlib.import_module(module), name)
        obs = [pool.observe(e) for e in range(pool.num_envs)]
        with (Path(a.output_dir) / "transitions.jsonl").open("w") as log:
            for turn in range(a.steps):
                e = turn % pool.num_envs
                r = pool.step(e, policy(obs[e], e))
                obs[e] = r.observation
                log.write(
                    json.dumps(
                        {
                            "turn": turn,
                            "env_id": e,
                            "reward": r.reward,
                            "terminated": r.terminated,
                            "truncated": r.truncated,
                            "timeshare": r.info["timeshare"],
                        }
                    )
                    + "\n"
                )
                log.flush()
                if r.terminated or r.truncated:
                    obs[e], _ = pool.reset(e)


if __name__ == "__main__":
    main()
