import copy, importlib, subprocess, sys, threading, time, unittest
from kit_timeshare import RolloutPool, BehaviorConfig, RoboDojoConfig
from kit_timeshare.pool import EpisodeDoneError, FrozenStateError


class Backend:
    num_envs = 2

    def __init__(self):
        self.state = [{"x": 0, "reward_cache": 0}, {"x": 0, "reward_cache": 0}]
        self.actions = []
        self.closed = 0
        self.corrupt = False

    def snapshot(self):
        return copy.deepcopy(self.state)

    def step(self, e, a):
        self.actions.append((e, a))
        self.state[e]["x"] += a
        if self.corrupt:
            self.state[1 - e]["reward_cache"] += 1
        return {"x": self.state[e]["x"]}, a / 10, False, False, {"native": True}

    def reset(self, e, *, seed=None, options=None):
        self.state[e] = {"x": 0, "reward_cache": 0}
        return {"x": 0}, {"seed": seed}

    def observe(self, e):
        return {"x": self.state[e]["x"]}

    def close(self):
        self.closed += 1


class PoolTests(unittest.TestCase):
    def test_action_forwarding_and_independent_reset(self):
        b = Backend()
        with RolloutPool(b, verify_frozen=True) as pool:
            r = pool.step(1, 7)
            self.assertEqual(tuple(r)[:4], ({"x": 7}, 0.7, False, False))
            self.assertEqual(b.actions, [(1, 7)])
            self.assertEqual(pool.snapshot()[0]["x"], 0)
            pool.step(0, 4)
            before = pool.snapshot()[0]
            obs, info = pool.reset(1, seed=123)
            self.assertEqual(pool.snapshot()[0], before)
            self.assertEqual(obs, {"x": 0})
            self.assertEqual(info["seed"], 123)
            self.assertEqual(pool.step(1, 2).info["timeshare"]["episode_steps"], 1)
        pool.close()
        self.assertEqual(b.closed, 1)

    def test_terminal_requires_selected_reset(self):
        b = Backend()
        p = RolloutPool(b, max_episode_steps=1)
        self.assertTrue(p.step(0, 1).truncated)
        with self.assertRaises(EpisodeDoneError):
            p.step(0, 2)
        self.assertTrue(p.step(1, 3).truncated)
        p.reset(0)
        self.assertTrue(p.step(0, 4).truncated)

    def test_hidden_reward_mutation_stops_pool(self):
        b = Backend()
        b.corrupt = True
        p = RolloutPool(b, verify_frozen=True)
        with self.assertRaises(FrozenStateError):
            p.step(0, 3)
        with self.assertRaisesRegex(RuntimeError, "failed validation"):
            p.step(1, 2)

    def test_invalid_input_is_not_forwarded(self):
        b = Backend()
        p = RolloutPool(b)
        for e in [True, -1, 2, 1.5]:
            with self.assertRaises(IndexError):
                p.step(e, 1)
        with self.assertRaises(ValueError):
            p.step(0, None)
        self.assertFalse(b.actions)

    def test_snapshot_not_live_reference(self):
        p = RolloutPool(Backend())
        s = p.snapshot()
        s[0]["x"] = 99
        self.assertEqual(p.snapshot()[0]["x"], 0)

    def test_import_without_sdk(self):
        script = "import sys,kit_timeshare; assert not any(x in sys.modules for x in ['torch','numpy','isaacsim','omnigibson','isaaclab'])"
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_worker_thread_is_rejected_before_native_call(self):
        b = Backend()
        p = RolloutPool(b)
        errors = []

        def call():
            try:
                p.step(0, 1)
            except RuntimeError as exc:
                errors.append(str(exc))

        t = threading.Thread(target=call)
        t.start()
        t.join()
        self.assertIn("creation thread", errors[0])
        self.assertFalse(b.actions)

    def test_config_limits(self):
        for kwargs in [
            {"num_envs": 0},
            {"num_envs": 1.5},
            {"num_envs": True},
            {"cpu_threads": 0},
            {"max_episode_steps": 0},
        ]:
            with self.assertRaises(ValueError):
                BehaviorConfig(**kwargs)
        with self.assertRaises(ValueError):
            RoboDojoConfig(task="../bad")


if __name__ == "__main__":
    unittest.main()
