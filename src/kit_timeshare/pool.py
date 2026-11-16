from __future__ import annotations
from threading import RLock, get_ident
from typing import Any, NamedTuple
import copy
import time


def first_difference(a, b, path="state"):
    if type(a) is dict and type(b) is dict:
        for key in sorted(a.keys() | b.keys(), key=str):
            if key not in a or key not in b:
                return path + "." + str(key) + ": key added/removed"
            if a[key] != b[key]:
                return first_difference(a[key], b[key], path + "." + str(key))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return path + ": length changed"
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return first_difference(x, y, path + "[" + str(i) + "]")
    return path + ": " + repr(a)[:100] + " -> " + repr(b)[:100]


class StepResult(NamedTuple):
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class FrozenStateError(RuntimeError):
    pass


class EpisodeDoneError(RuntimeError):
    pass


class RolloutPool:
    """Serialize native calls while exposing independent logical environments.

    Backends implement step/reset/observe/snapshot/close. No action generation,
    initial-state perturbation, implicit reset, or benchmark scoring is added here.
    """

    def __init__(self, backend, *, verify_frozen=False, max_episode_steps=None):
        self.backend = backend
        self.num_envs = backend.num_envs
        self.verify_frozen = verify_frozen
        self.max_episode_steps = max_episode_steps
        self._owner = get_ident()
        self._lock = RLock()
        self._closed = False
        self._failed = False
        self._steps = [0] * self.num_envs
        self._done = [False] * self.num_envs

    def _check(self, e):
        if get_ident() != self._owner:
            raise RuntimeError("Call Kit on the pool creation thread")
        if self._closed:
            raise RuntimeError("Pool is closed")
        if self._failed:
            raise RuntimeError("Pool failed validation; close it before continuing")
        if isinstance(e, bool) or not isinstance(e, int) or not 0 <= e < self.num_envs:
            raise IndexError(f"env_id must be an integer in [0, {self.num_envs})")

    def _before(self, e):
        return copy.deepcopy(self.backend.snapshot()) if self.verify_frozen else None

    def _validate(self, e, before):
        if before is None:
            return
        after = self.backend.snapshot()
        changed = [i for i in range(self.num_envs) if i != e and before[i] != after[i]]
        if changed:
            self._failed = True
            detail = first_difference(
                before[changed[0]], after[changed[0]], f"env[{changed[0]}]"
            )
            raise FrozenStateError(
                f"Inactive environments changed: {changed}; {detail}"
            )

    def step(self, env_id, action):
        with self._lock:
            self._check(env_id)
            if self._done[env_id]:
                raise EpisodeDoneError("Reset this environment before its next step")
            if action is None:
                raise ValueError("Provide a real policy action; None is not an action")
            before = self._before(env_id)
            start = time.perf_counter()
            try:
                result = StepResult(*self.backend.step(env_id, action))
                if not isinstance(result.info, dict):
                    raise TypeError("Backend info must be a dict")
                self._steps[env_id] += 1
                truncated = bool(result.truncated) or (
                    self.max_episode_steps is not None
                    and self._steps[env_id] >= self.max_episode_steps
                )
                self._done[env_id] = bool(result.terminated or truncated)
                service_seconds = time.perf_counter() - start
                self._validate(env_id, before)
            except Exception:
                self._failed = True
                raise
            info = dict(result.info)
            info["timeshare"] = {
                "env_id": env_id,
                "episode_steps": self._steps[env_id],
                "service_seconds": service_seconds,
                "auto_reset": False,
            }
            return StepResult(
                result.observation,
                float(result.reward),
                bool(result.terminated),
                truncated,
                info,
            )

    def reset(self, env_id, *, seed=None, options=None):
        with self._lock:
            self._check(env_id)
            before = self._before(env_id)
            try:
                obs, info = self.backend.reset(env_id, seed=seed, options=options)
                self._validate(env_id, before)
            except Exception:
                self._failed = True
                raise
            self._steps[env_id] = 0
            self._done[env_id] = False
            return obs, {
                **info,
                "timeshare": {
                    "env_id": env_id,
                    "episode_steps": 0,
                    "auto_reset": False,
                },
            }

    def observe(self, env_id):
        with self._lock:
            self._check(env_id)
            return self.backend.observe(env_id)

    def snapshot(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Pool is closed")
            return copy.deepcopy(self.backend.snapshot())

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
