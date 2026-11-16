"""Two-env integration check. Run using a pinned upstream SDK and isolated GPU.

This demo generates simple joint targets solely for testing. Production callers
supply their own policy actions to pool.step(). No initial poses are perturbed.
"""

import argparse, hashlib, json, sys, time, traceback, faulthandler, signal

faulthandler.enable()
faulthandler.register(signal.SIGUSR2)
from pathlib import Path
from kit_timeshare import BehaviorConfig, RoboDojoConfig, make


def demo_action(pool, e, turn):
    import numpy as np

    b = pool.backend
    if hasattr(b, "envs"):
        robot = b.envs[e].robots[0]
        action = np.zeros(robot.action_dim, dtype=np.float32)
        action[0] = 0.03 if turn % 2 == 0 else -0.03
        return action
    rm = b.env.robot_manager
    control = {}
    for i, r in enumerate(rm.robot_list):
        q = (
            b.arms[i]
            .data.joint_pos[e, r.arm_joint_indices]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        q[0] += 0.015 if turn % 2 == 0 else -0.015
        control[rm.process_name(r.arm_name)] = {
            "position": q.tolist(),
            "velocity": [0.0] * len(q),
        }
        control[rm.process_name(r.gripper_name)] = {
            "position": [float(r.gripper_scale[1])],
            "velocity": [0.0],
        }
    return control


def images(value):
    """Collect native nonconstant RGB arrays without imposing an observation schema."""
    if isinstance(value, dict):
        for v in value.values():
            yield from images(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from images(v)
    elif (
        hasattr(value, "shape") and len(value.shape) == 3 and value.shape[-1] in (3, 4)
    ):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        import numpy as np

        x = np.asarray(value)
        if x.size:
            yield {
                "shape": list(x.shape),
                "std": float(x.std()),
                "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["behavior", "robodojo"])
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config-path")
    a = p.parse_args()
    run = Path(a.output_dir)
    report = {"backend": a.backend, "passed": False, "rounds": [], "phase": "startup"}

    def save():
        run.mkdir(parents=True, exist_ok=True)
        (run / "acceptance.json").write_text(json.dumps(report, indent=2))

    cfg = dict(
        upstream_root=a.upstream_root,
        output_dir=run,
        num_envs=2,
        verify_frozen=True,
        max_episode_steps=3,
    )
    if a.backend == "behavior":
        config = BehaviorConfig(**cfg, config_path=a.config_path)
    else:
        config = RoboDojoConfig(**cfg)
    start = time.perf_counter()
    pool = None
    try:
        pool = make(config)
        report["startup_s"] = time.perf_counter() - start
        report["phase"] = "observe"
        save()
        initial = pool.snapshot()
        obs = pool.observe(0)
        rgb = list(images(obs))
        if not any(i["std"] > 1 for i in rgb):
            raise AssertionError("No nonblank native RGB image")
        report["rgb"] = rgb
        for rep in range(3):
            report["phase"] = f"round_{rep}"
            save()
            rows = []
            for turn in range(6):
                e = turn % 2
                before = pool.snapshot()
                action = demo_action(pool, e, turn)
                result = pool.step(e, action)
                after = pool.snapshot()
                if before[1 - e] != after[1 - e]:
                    raise AssertionError("Inactive environment changed")
                if before[e] == after[e]:
                    raise AssertionError("Active environment did not advance")
                physical_keys = (
                    ["state"] if "state" in before[e] else ["bodies", "joints"]
                )
                moved = any(before[e][k] != after[e][k] for k in physical_keys)
                rows.append(
                    {
                        "env_id": e,
                        "reward": result.reward,
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                        "service_s": result.info["timeshare"]["service_seconds"],
                        "physical_state_changed": moved,
                    }
                )
                if result.terminated or result.truncated:
                    before = pool.snapshot()
                    t = time.perf_counter()
                    pool.reset(e, seed=rep * 10 + e)
                    after = pool.snapshot()
                    if before[1 - e] != after[1 - e]:
                        raise AssertionError("Reset changed another environment")
                    rows[-1]["reset_s"] = time.perf_counter() - t
            report["rounds"].append(rows)
            save()
        if not all(
            any(
                row["env_id"] == e and row["physical_state_changed"]
                for round in report["rounds"]
                for row in round
            )
            for e in range(2)
        ):
            raise AssertionError(
                "Policy actions did not produce physical state changes in both environments"
            )
        report.update(
            passed=True,
            phase="complete",
            inactive_step_checks=18,
            adapter_module=str(type(pool.backend).__module__),
            initial_state_perturbed=False,
        )
        report["loaded_legacy_modules"] = [
            m
            for m in sys.modules
            if m in ["harness", "batch_adapters", "independent_control", "batch_worker"]
        ]
        if report["loaded_legacy_modules"]:
            raise AssertionError("Legacy harness dependency")
    except BaseException:
        report.update(passed=False, traceback=traceback.format_exc())
        raise
    finally:
        save()
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    main()
