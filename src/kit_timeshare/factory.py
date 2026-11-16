from pathlib import Path
import json
from .config import BehaviorConfig, RoboDojoConfig
from .compat import check_versions
from .gpu import select_isolated_gpu
from .pool import RolloutPool

_used = False


def make(config):
    global _used
    if not isinstance(config, (BehaviorConfig, RoboDojoConfig)):
        raise TypeError("Use BehaviorConfig or RoboDojoConfig")
    if _used:
        raise RuntimeError(
            "Kit supports one native pool lifetime per process; use a fresh process for another pool"
        )
    kind = "behavior" if isinstance(config, BehaviorConfig) else "robodojo"
    profile = check_versions(kind, config.upstream_root, config.strict_versions)
    gpu = select_isolated_gpu(config.gpu_uuid)
    run = Path(config.output_dir).expanduser().resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Use a new, empty output_dir")
    run.mkdir(parents=True, exist_ok=True)
    _used = True
    if kind == "behavior":
        from .backends.behavior import BehaviorBackend

        backend = BehaviorBackend(config, run)
    else:
        from .backends.robodojo import RoboDojoBackend

        backend = RoboDojoBackend(config, run)
    (run / "runtime_profile.json").write_text(
        json.dumps(
            {
                "backend": kind,
                "compatibility": profile,
                "physical_gpu": gpu.__dict__,
                "num_envs": config.num_envs,
            },
            indent=2,
        )
    )
    return RolloutPool(
        backend,
        verify_frozen=config.verify_frozen,
        max_episode_steps=config.max_episode_steps,
    )
