"""Policy-facing time-shared rollout; native SDKs are imported only by make()."""

from .config import BehaviorConfig, RoboDojoConfig
from .pool import RolloutPool, StepResult

__version__ = "0.1.0"


def make(config):
    """Create a native pool inside an already installed upstream SDK environment."""
    from .factory import make as create

    return create(config)


__all__ = ["BehaviorConfig", "RoboDojoConfig", "RolloutPool", "StepResult", "make"]
