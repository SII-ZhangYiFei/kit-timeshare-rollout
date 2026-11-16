from __future__ import annotations
from contextlib import contextmanager
import copy, random
import numpy as np
import torch


class NativeRNG:
    """Per-env RNG streams; preserve the caller's RNG state around native calls."""

    def _capture_rng(self):
        return (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state(0) if torch.cuda.is_initialized() else None,
        )

    def _load_rng(self, s):
        random.setstate(s[0])
        np.random.set_state(s[1])
        torch.set_rng_state(s[2])
        if s[3] is not None:
            torch.cuda.set_rng_state(s[3], 0)

    def _init_rng(self, seed):
        outside = self._capture_rng()
        self._rng = []
        try:
            for e in range(self.num_envs):
                random.seed(seed + e)
                np.random.seed((seed + e) % 2**32)
                torch.manual_seed(seed + e)
                self._rng.append(self._capture_rng())
        finally:
            self._load_rng(outside)

    @contextmanager
    def rng(self, e, seed=None):
        outside = self._capture_rng()
        try:
            self._load_rng(self._rng[e])
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.manual_seed(seed)
            yield
        finally:
            self._rng[e] = self._capture_rng()
            self._load_rng(outside)
