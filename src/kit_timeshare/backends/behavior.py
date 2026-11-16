from __future__ import annotations
from pathlib import Path
import copy, json, os, time
import numpy as np
import torch
from .base import NativeRNG


class BehaviorBackend(NativeRNG):
    def __init__(self, cfg, run):
        self.num_envs = cfg.num_envs
        self.run = run
        self.steps = [0] * self.num_envs
        if cfg.upstream_root:
            import sys

            root = Path(cfg.upstream_root).expanduser().resolve()
            sys.path.insert(
                0, str(root / "OmniGibson" if (root / "OmniGibson").is_dir() else root)
            )
        import isaacsim

        original = isaacsim.SimulationApp

        def launch(config, *args, **kwargs):
            config = dict(
                config,
                headless=cfg.headless,
                multi_gpu=False,
                active_gpu=0,
                physics_gpu=0,
                limit_cpu_threads=cfg.cpu_threads,
            )
            extra = [
                f"--/rtx/descriptorSets={cfg.descriptor_sets}",
                f"--/rtx/reservedDescriptors={cfg.reserved_descriptors}",
                f"--/plugins/carb.tasking.plugin/threadCount={cfg.cpu_threads}",
                f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={cfg.cpu_threads}",
            ]
            config["extra_args"] = list(config.get("extra_args", [])) + extra
            return original(config, *args, **kwargs)

        isaacsim.SimulationApp = launch
        import omnigibson as og
        from omnigibson.macros import gm

        self.og = og
        if og.sim is not None:
            raise RuntimeError(
                "Create the pool in a fresh process before another Kit simulation"
            )
        with gm.unlocked():
            gm.HEADLESS = cfg.headless
            gm.USE_GPU_DYNAMICS = False
            gm.ENABLE_FLATCACHE = True
        try:
            og.launch()
        finally:
            isaacsim.SimulationApp = original
        from .._runtime.behavior_patches import (
            prepare_behavior,
            configure_behavior_initial,
        )

        old = os.environ.get("BATCH_READ_ONLY_MATERIAL_QUERY")
        os.environ["BATCH_READ_ONLY_MATERIAL_QUERY"] = (
            "1" if cfg.read_only_material_queries else "0"
        )
        try:
            prepare_behavior(og, run)
        finally:
            if old is None:
                os.environ.pop("BATCH_READ_ONLY_MATERIAL_QUERY", None)
            else:
                os.environ["BATCH_READ_ONLY_MATERIAL_QUERY"] = old
        # Scene-local presampled poses must remain local when multiple scenes coexist.
        import inspect, textwrap
        from omnigibson.tasks.behavior_task import BehaviorTask

        fn = BehaviorTask.reset
        source = textwrap.dedent(inspect.getsource(fn))
        needle = 'robot.set_position_orientation(robot_pose["position"], robot_pose["orientation"])'
        if source.count(needle) != 1:
            raise RuntimeError("Unsupported BehaviorTask.reset source")
        patched = source.replace(needle, needle[:-1] + ', frame="scene")').replace(
            "super().reset(env)", "super(BehaviorTask, self).reset(env)"
        )
        (run / "reset_scene_frame.py").write_text(patched)
        scope = dict(fn.__globals__, BehaviorTask=BehaviorTask)
        exec(compile(patched, str(run / "reset_scene_frame.py"), "exec"), scope)
        BehaviorTask.reset = scope["reset"]
        import yaml

        if cfg.config is not None:
            config = copy.deepcopy(cfg.config)
        else:
            path = (
                Path(cfg.config_path)
                if cfg.config_path
                else Path(og.example_config_path) / "r1pro_behavior.yaml"
            )
            config = yaml.safe_load(path.read_text())
        (run / "environment_config.json").write_text(
            json.dumps(config, indent=2, default=str)
        )
        self.router = configure_behavior_initial(
            og, run, self.num_envs, cfg.scene_spacing
        )
        self.vec = og.VectorEnvironment(num_envs=self.num_envs, config=config)
        self.envs = self.vec.envs
        self.sim = og.sim
        self.router.initializing = False
        with self.sim.editing_usd():
            changed = self.router.refresh_ownership()
            if changed:
                raise RuntimeError(
                    "Late actor ownership changes are unsafe; unsupported upstream initialization"
                )
            self.router.install()
        self.sim._pre_physics_step_callback = None
        self.sim._post_physics_step_callback = None
        self.router.before_physics = self.sim._on_pre_physics_step
        self.router.after_physics = self.sim._on_post_physics_step
        self.sim._render_on_step = False
        self._members = [{o.name for o in e.scene.objects} for e in self.envs]
        for e in self.envs:
            e._automatic_reset = False
        if self.sim._physics_context.is_gpu_dynamics_enabled():
            raise RuntimeError("CPU physics is required")
        self._init_rng(cfg.seed)

    def step(self, e, action):
        self.router.select(e)
        env = self.envs[e]
        with self.rng(e):
            native = env._convert_action_to_tensor(action)
            env._pre_step(native)
            self.sim.step()
            self.sim.render()
            self.sim.render()
            result = env._post_step(native)
        torch.cuda.synchronize()
        self.steps[e] += 1
        return result

    def observe(self, e):
        self.router.select(e)
        self.sim.render()
        self.sim.render()
        return self.envs[e].get_obs()[0]

    def reset(self, e, *, seed=None, options=None):
        if options:
            raise ValueError(
                "Reset options are not yet supported; create another pool for a different task/config"
            )
        env = self.envs[e]
        if {o.name for o in env.scene.objects} != self._members[e]:
            raise RuntimeError(
                "Independent reset cannot restore spawned/removed objects. Use a new pool for changed topology."
            )
        self.router.select(e)
        original = env.scene.reset

        # Scene.restore(hard=True) restores the global simulator, touching other scenes.
        # Topology is unchanged here, so load only this scene's initial state.
        def selected_reset(hard=True):
            return original(hard=False)

        env.scene.reset = selected_reset
        try:
            with self.rng(e, seed):
                obs, info = env.reset()
        finally:
            env.scene.reset = original
        self.steps[e] = 0
        self.sim.render()
        self.sim.render()
        return env.get_obs()[0], dict(info)

    def snapshot(self):
        from omnigibson.utils.usd_utils import RigidContactAPI

        selected = self.router.active
        rows = []
        for e, env in enumerate(self.envs):
            self.router.select(e)
            rows.append(
                {
                    "state": env.scene.dump_state(serialized=True)
                    .detach()
                    .cpu()
                    .tolist(),
                    "contacts": sorted(
                        [
                            list(p)
                            for p in RigidContactAPI.get_contact_pairs(
                                e, {env.robots[0]}, None, True
                            )
                        ]
                    ),
                    "steps": self.steps[e],
                    "time": self.router.times[e],
                    "ticks": self.router.ticks[e],
                    "episode_steps": env._current_step,
                    "episode": env._current_episode,
                    "potential": float(env.task.get_potential(env)),
                }
            )
        self.router.select(selected)
        return rows

    def close(self):
        self.og.shutdown()
