from __future__ import annotations
from pathlib import Path
import ast, copy, hashlib, inspect, json, os, shlex, textwrap, types, time
import numpy as np
import torch
from .base import NativeRNG


def prepare_experience(repo, run, options):
    source = (
        Path(options.experience)
        if options.experience
        else repo / "third_party/IsaacLab/apps/isaaclab.python.kit"
    )
    text = source.read_text()
    if options.urdf_importer_version:
        import re

        pattern = (
            r'("isaacsim\.asset\.importer\.urdf"\s*=\s*\{version\s*=\s*")[^"]+(".*)'
        )
        text, count = re.subn(
            pattern, lambda m: m[1] + options.urdf_importer_version + m[2], text
        )
        if count != 1:
            raise RuntimeError("Unsupported IsaacLab experience template")
    text = text.replace("${app}", str(source.parent.resolve()))
    target = run / "isaaclab.timeshare.kit"
    target.write_text(text)
    return target


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if callable(value):
        return {"callable": getattr(value, "__qualname__", type(value).__name__)}
    raise TypeError(f"Cannot serialize rollout state {type(value).__name__}")


def copy_state(value):
    # Registered reward predicates can be bound methods of Func_Parser. Copying
    # their owners would recurse into the entire live simulator.
    if callable(value):
        return value
    if isinstance(value, dict):
        return {k: copy_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [copy_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(copy_state(v) for v in value)
    return copy.deepcopy(value)


def isolate_reward_rows(manager):
    """Detach predicate argument containers shared by upstream registrations."""
    for obj in (manager, manager.func_parser):
        for name, value in list(vars(obj).items()):
            if isinstance(value, list) and len(value) == manager.num_envs:
                setattr(obj, name, [copy_state(row) for row in value])


def patch_reward_manager(manager, router, run):
    """Retain upstream formulas; scope side-effecting reward loops to the selected ID."""

    class Scope(ast.NodeTransformer):
        def visit_For(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "env_idx"
                and ast.unparse(node.iter) == "range(self.num_envs)"
            ):
                appends = any(
                    isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Attribute)
                    and isinstance(x.func.value, ast.Name)
                    and x.func.value.id == "reward_lst"
                    and x.func.attr == "append"
                    for st in node.body
                    for x in ast.walk(st)
                )
                code = "if env_idx != _timeshare_active():\n"
                if appends:
                    code += "    reward_lst.append(self._timeshare_rewards[env_idx])\n"
                code += "    continue"
                node.body = ast.parse(code).body + node.body
            return node

    manager._timeshare_rewards = [0.0] * manager.num_envs
    manifest = {}
    for name in ["get_reward", "_final_check", "_final_score"]:
        original = getattr(manager, name).__func__
        source = textwrap.dedent(inspect.getsource(original))
        if "for env_idx in range(self.num_envs)" not in source:
            raise RuntimeError("Unsupported reward source: " + name)
        tree = ast.fix_missing_locations(Scope().visit(ast.parse(source)))
        scope = dict(original.__globals__)
        scope["_timeshare_active"] = lambda: router.active
        exec(compile(tree, str(run / (name + "_selected.py")), "exec"), scope)
        setattr(manager, name, types.MethodType(scope[name], manager))
        manifest[name] = hashlib.sha256(source.encode()).hexdigest()
    (run / "reward_patch_hashes.json").write_text(json.dumps(manifest, indent=2))


class RoboDojoBackend(NativeRNG):
    def __init__(self, options, run):
        self.num_envs = options.num_envs
        self.run = run
        self.steps = [0] * self.num_envs
        previous = Path.cwd()
        try:
            self._initialize(options, run)
        finally:
            os.chdir(previous)
        self.router = self.env.sim._independent_router
        self.router.install()
        self.arms = self.robots
        self.rl = self.env.sim
        if not all(
            "Implicit" in type(act).__name__
            for arm in self.arms
            for act in arm.actuators.values()
        ):
            raise RuntimeError("This profile requires implicit joint actuators")
        if self.rl.cfg.events and "interval" in self.rl.event_manager.available_modes:
            raise RuntimeError(
                "Interval events require a task-specific isolation hook and are not supported in this profile"
            )
        # EvalEnv registers these task conditions after TaskEnv.reset; an empty
        # RewardManager would otherwise report vacuous success on its first call.
        self.env.run_reward()
        if hasattr(self.env, "get_score"):
            self.env.get_score()
        rm = self.env.reward_manager
        if not any(
            rm.check_list[e]
            or rm.final_check_list[e]
            or rm.query_list[e]
            or rm.trigger_check_list[e]
            or rm.trigger_query_list[e]
            for e in range(self.num_envs)
        ):
            raise RuntimeError(
                "Task registered no success conditions; refusing vacuous reward"
            )
        isolate_reward_rows(rm)
        patch_reward_manager(rm, self.router, run)
        if hasattr(self.env, "get_score"):
            rm._gated_score_lst = rm.get_score(reward_lst=[0.0] * self.num_envs)
        self._init_rng(options.seed)
        self._initial = [self._capture(e) for e in range(self.num_envs)]

    def _initialize(self, options, run):
        n = options.num_envs
        self.n = n
        import argparse, sys

        repo = Path(options.upstream_root).expanduser().resolve()
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        from isaaclab.app import AppLauncher

        p = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(p)
        experience = prepare_experience(repo, run, options)
        args = p.parse_args(
            (["--headless"] if options.headless else [])
            + [
                "--enable_cameras",
                "--device",
                "cpu",
                "--experience",
                str(experience),
                "--kit_args",
                f'--portable-root {shlex.quote(str(run/"kit"))} --/renderer/activeGpu=0 --/physics/cudaDevice=0 --/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount={options.cpu_threads} --/plugins/omni.tbb.globalcontrol/maxThreadCount={options.cpu_threads}',
            ]
        )
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

        add_zero_delay_kit_args(args)
        t = time.perf_counter()
        self.app = AppLauncher(args).app
        self.launch_s = time.perf_counter() - t
        import carb
        from omni.physx.bindings import _physx as pxb

        settings = carb.settings.get_settings()
        settings.set(pxb.SETTING_PHYSX_DISPATCHER, True)
        settings.set(pxb.SETTING_NUM_THREADS, 0)
        (run / "physics_dispatcher.json").write_text(
            json.dumps(
                {
                    "physx_dispatcher": settings.get(pxb.SETTING_PHYSX_DISPATCHER),
                    "num_threads": settings.get(pxb.SETTING_NUM_THREADS),
                    "reason": "Serialize small CPU scenes within this process; avoid task-queue waits during multi-scene initialization.",
                },
                indent=2,
            )
        )
        from omegaconf import OmegaConf
        from utils.pipeline_utils import process_config, process_randomization
        import yaml

        load = lambda x: yaml.safe_load(open(repo / x))
        ev = load(options.eval_config)
        ev.update(task_name=options.task, num_envs=n, seed=options.seed)
        cfg = OmegaConf.create(
            {
                "sim": load(options.sim_config),
                "scene": load(options.scene_config),
                "robot": load(options.robot_config),
                "camera": load(options.camera_config),
                "task_env": load(f"task/{options.benchmark}/config/{options.task}.yml"),
                "eval_cfg": ev,
            }
        )
        cfg.sim.scene.num_envs = n
        cfg.sim.seed = [options.seed + e for e in range(n)]
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, options.task)
        cfg.camera.default_frequency = 25
        (run / "config.json").write_text(OmegaConf.to_yaml(cfg))
        # RoboDojo otherwise overrides the CPU frontend with force-GPU physics.
        # Change only the benchmark process and record the exact SDK method patch.
        if True:
            import inspect, textwrap, hashlib
            from env.environment import base_env
            from omni.physx import acquire_physx_interface

            original = base_env.BaseEnv.setup_physics
            source = textwrap.dedent(inspect.getsource(original))
            if source.count("overwrite_gpu_setting(1)") != 1:
                raise RuntimeError("Unsupported RoboDojo physics setup source")
            corrected = source.replace(
                "overwrite_gpu_setting(1)", "overwrite_gpu_setting(0)"
            )
            target = run / "robodojo_cpu_physics_setup.py"
            target.write_text(corrected)
            namespace = dict(original.__globals__)
            exec(compile(corrected, str(target), "exec"), namespace)
            base_env.BaseEnv.setup_physics = namespace["setup_physics"]
            acquire_physx_interface().overwrite_gpu_setting(0)
            cfg.sim.device = "cpu"
            (run / "cpu_physics_patch.json").write_text(
                json.dumps(
                    {
                        "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "patched_sha256": hashlib.sha256(
                            corrected.encode()
                        ).hexdigest(),
                        "scope": "Process-local BaseEnv.setup_physics: 0 forces CPU; this is a mode, not a GPU index.",
                    },
                    indent=2,
                )
            )
        # Build independent CPU scenes before the initial PhysX parse. Changing
        # ownership after an IsaacLab physics-replicated scene was initialized
        # left secondary actors unadvanced in the first diagnostic.
        from env.environment.isaac.isaac_rl_env import IsaacRLEnv
        from .._runtime.scenes import SceneRouter

        original_setup = IsaacRLEnv._setup_scene

        def independent_setup(rl):
            rl.cfg.scene.replicate_physics = False
            rl.scene.cfg.replicate_physics = False
            original_setup(rl)
            roots = [str(x) for x in rl.scene.env_prim_paths]
            rl._independent_router = SceneRouter(
                rl.sim.stage, rl.sim.get_physics_context(), roots, run
            )

        IsaacRLEnv._setup_scene = independent_setup
        # Reset can create fresh bowls after initial scene setup. Assign their
        # owners before their native rigid-body views are initialized/stepped.
        from env.scene_manager.scene_manager import SceneManager
        import inspect, textwrap

        original_reload = SceneManager.reload_envs
        source = textwrap.dedent(inspect.getsource(original_reload))
        needle = "    for obj in self.pending_initialization:"
        if source.count(needle) != 1:
            raise RuntimeError("Unsupported scene reload source")
        patched = source.replace(
            needle, "    _assign_independent_owners(self)\n" + needle
        )
        target = run / "scene_reload_independent_owners.py"
        target.write_text(patched)

        def assign_independent_owners(manager):
            router = getattr(manager.sim, "_independent_router", None)
            if router is not None:
                router.refresh_ownership()

        namespace = dict(original_reload.__globals__)
        namespace["_assign_independent_owners"] = assign_independent_owners
        exec(compile(patched, str(target), "exec"), namespace)
        SceneManager.reload_envs = namespace["reload_envs"]
        import importlib

        task_class = getattr(
            importlib.import_module(f"task.{options.benchmark}.tasks.{options.task}"),
            options.task,
        )

        def phase(name):
            (run / "startup_phase.json").write_text(
                json.dumps({"phase": name, "time": time.time()})
            )

        phase("construct_task")
        t = time.perf_counter()
        self.env = task_class(cfg, self.app)
        # Native EvalEnv initializes these lifecycle flags before TaskEnv.reset.
        # 'success' here means an active/non-failed episode, not task completion.
        self.env.success = [True] * n
        self.env.end_flag = [False] * n
        self.env.take_action_cnt = [0] * n
        layout = (
            Path(options.layout_path).expanduser().resolve()
            if options.layout_path
            else repo
            / f"Assets/Eval_Layout/{options.benchmark}/arx_x5/0/{options.task}_0.json"
        )
        [
            (
                self.env.scene_manager.layout_manager.set_saved_layout(
                    e, json.loads(layout.read_text())
                )
            )
            for e in range(n)
        ]
        self.scene_load_s = time.perf_counter() - t
        phase("reset_task")
        t = time.perf_counter()
        self.env.reset(seed=[options.seed + e for e in range(n)])
        self.env.scene_manager.apply_saved_poses(env_idx_list=list(range(n)))
        self.env.robot_manager.set_origin_endpose()
        self.env.robot_manager.set_robot_init_state()
        self.env.reward_manager.init_state()
        for _ in range(200):
            self.env.sim_step()
        phase("warmup_complete")
        self.reset_s = time.perf_counter() - t
        self.i = 0
        self.robots = self.env.robot_manager.robot_key
        self.initial = [r.data.joint_pos.clone() for r in self.robots]
        self.objects = self.env.scene_manager._rigid_and_dynamic_objects[0]
        self.info = {
            "task": options.task,
            "layout": str(layout),
            "object_count": len(self.objects),
            "arm_count": len(self.robots),
            "tensor_device": str(self.env.sim.sim.device),
            "planner_enabled": True,
            "camera_names": self.env.camera_manager.camera_names,
        }
        self.info["physics_context"] = {
            name: getattr(self.env.physics_context, name)()
            for name in ["is_gpu_dynamics_enabled", "get_broadphase_type"]
            if callable(getattr(self.env.physics_context, name, None))
        }
        self.info["physics_runtime_gpu_override"] = (
            self.env.physics_interface.get_overwrite_gpu_setting()
        )
        self.info["cpu_physics_requested"] = True
        if self.info["cpu_physics_requested"]:
            if self.info["physics_runtime_gpu_override"] != 0:
                raise RuntimeError("CPU PhysX override was not applied")
            if self.info["physics_context"]["is_gpu_dynamics_enabled"] is not False:
                raise RuntimeError("GPU dynamics must be disabled")
            if self.info["physics_context"]["get_broadphase_type"] == "GPU":
                raise RuntimeError("GPU broadphase must be disabled")
        self.info["physics_backend_note"] = (
            "RoboDojo setup_physics normally forces GPU with overwrite_gpu_setting(1); USD/SimulationManager CPU tensor defaults may still report false/MBP. Runtime override 1 means force GPU."
        )
        if self.info["cpu_physics_requested"]:
            self.info["physics_backend_note"] = (
                "CPU solver forced by runtime override0, CPU frontend, GPU dynamics disabled and CPU broadphase; RTX renderer and cuRobo planners remain on GPU0."
            )
        self.info["lifecycle"] = (
            "Native TaskEnv; success flag means episode validity, not task success"
        )
        self.options = options

    def _rows(self, obj, e):
        return {
            k: copy_state(v[e])
            for k, v in vars(obj).items()
            if isinstance(v, list) and len(v) == self.num_envs
        }

    def _capture(self, e):
        bodies = {}
        for name, o in self.env.scene_manager._rigid_and_dynamic_objects[e].items():
            p, q = o.get_world_pose()
            bodies[name] = [
                copy.deepcopy(x)
                for x in [p, q, o.get_linear_velocity(), o.get_angular_velocity()]
            ]
        joints = []
        for arm in self.arms:
            joints.append(
                [
                    v[e].clone()
                    for v in [
                        arm.data.joint_pos,
                        arm.data.joint_vel,
                        arm.data.joint_pos_target,
                        arm.data.joint_vel_target,
                        arm.data.joint_effort_target,
                    ]
                ]
            )
        rm = self.env.reward_manager
        return {
            "bodies": bodies,
            "joints": joints,
            "reward": self._rows(rm, e),
            "parser": self._rows(rm.func_parser, e),
            "prev_control": copy.deepcopy(
                self.env.robot_manager.control_manager.prev_control[e]
            ),
            "flags": {
                k: copy.deepcopy(getattr(self.env, k)[e])
                for k in ["success", "end_flag", "take_action_cnt"]
            },
        }

    def step(self, e, action):
        from env.robot_manager.control_manager import MetaControl

        if isinstance(action, dict):
            action = MetaControl(copy.deepcopy(action))
        if not isinstance(action, MetaControl):
            raise TypeError(
                "RoboDojo actions must be a native MetaControl or its control_info_dict"
            )
        self.router.select(e)
        actions = [None] * self.num_envs
        actions[e] = action
        with self.rng(e):
            self.env.step(actions)
            for _ in range(self.options.control_substeps):
                for _ in range(self.rl.cfg.decimation):
                    self.rl._sim_step_counter += 1
                    for arm in self.arms:
                        ids = torch.tensor([e], dtype=torch.int32, device=arm.device)
                        arm.root_physx_view.set_dof_position_targets(
                            arm.data.joint_pos_target, ids
                        )
                        arm.root_physx_view.set_dof_velocity_targets(
                            arm.data.joint_vel_target, ids
                        )
                        arm.root_physx_view.set_dof_actuation_forces(
                            arm.data.joint_effort_target, ids
                        )
                    self.rl.sim.step(render=False)
                    if (
                        self.rl.cfg.sim.render_interval > 0
                        and self.rl._sim_step_counter % self.rl.cfg.sim.render_interval
                        == 0
                    ):
                        self.rl.sim.render()
                    self.rl.scene.update(dt=self.rl.physics_dt)
                self.rl.episode_length_buf[e] += 1
                self.rl.common_step_counter += 1
            self.env.take_action_cnt[e] += 1
            rm = self.env.reward_manager
            rm.step(env_idx_list=[e])
            rewards = rm.get_reward(
                final_check=bool(
                    not self.env.success[e]
                    or self.env.take_action_cnt[e] >= self.env.step_lim
                )
            )
            rm._timeshare_rewards[e] = rewards[e]
            obs = self.observe(e)
        self.steps[e] += 1
        torch.cuda.synchronize()
        reward = float(rewards[e])
        success = reward >= 1.0
        info = {
            "is_success": success,
            "native_episode_valid": bool(self.env.success[e]),
            "native_end_flag": bool(self.env.end_flag[e]),
            "reward_source": "upstream RewardManager.get_reward",
            "simulated_dt_s": self.options.control_substeps * self.rl.step_dt,
        }
        if rm._gated_score_lst is not None:
            info["native_score"] = rm._gated_score_lst[e]
        return (
            obs,
            reward,
            bool(success or not self.env.success[e] or self.env.end_flag[e]),
            bool(self.env.take_action_cnt[e] >= self.env.step_lim),
            info,
        )

    def observe(self, e):
        from env.camera_manager.capture.render_sync import wait_for_latest_cameras

        wait_for_latest_cameras(
            self.env,
            capture_manager=self.env.capture_manager,
            min_passes=2,
            max_passes=8,
        )
        sensors = self.env.capture_manager.step(env_ids=[e])
        return {
            "sensors": sensors,
            "joint_positions": [r.data.joint_pos[e].clone() for r in self.arms],
            "joint_velocities": [r.data.joint_vel[e].clone() for r in self.arms],
        }

    def reset(self, e, *, seed=None, options=None):
        if options:
            raise ValueError(
                "Only reset to the loaded layout is supported; create a new pool for a different layout"
            )
        s = self._initial[e]
        objects = self.env.scene_manager._rigid_and_dynamic_objects[e]
        if set(objects) != set(s["bodies"]):
            raise RuntimeError(
                "Object membership changed; independent reset requires fixed topology"
            )
        self.router.select(e)
        with self.rng(e, seed):
            for name, (p, q, v, w) in s["bodies"].items():
                obj = objects[name]
                obj.set_world_pose(
                    position=copy.deepcopy(p), orientation=copy.deepcopy(q)
                )
                obj.set_linear_velocity(copy.deepcopy(v))
                obj.set_angular_velocity(copy.deepcopy(w))
            for arm, (q, v, p_target, v_target, effort) in zip(self.arms, s["joints"]):
                ids = torch.tensor([e], device=arm.device, dtype=torch.long)
                arm.write_joint_state_to_sim(
                    q[None].clone(), v[None].clone(), env_ids=ids
                )
                arm.set_joint_position_target(p_target[None].clone(), env_ids=ids)
                arm.set_joint_velocity_target(v_target[None].clone(), env_ids=ids)
                arm.set_joint_effort_target(effort[None].clone(), env_ids=ids)
            rm = self.env.reward_manager
            for obj, data in [(rm, s["reward"]), (rm.func_parser, s["parser"])]:
                for k, v in data.items():
                    getattr(obj, k)[e] = copy_state(v)
            self.env.robot_manager.control_manager.prev_control[e] = copy.deepcopy(
                s["prev_control"]
            )
            for k, v in s["flags"].items():
                getattr(self.env, k)[e] = copy.deepcopy(v)
            self.rl.episode_length_buf[e] = 0
            self.steps[e] = 0
            obs = self.observe(e)
        return obs, {
            "reset_mode": "loaded_layout_snapshot",
            "seed": seed,
            "layout_resampled": False,
        }

    def snapshot(self):
        result = []
        for e in range(self.num_envs):
            data = self._capture(e)
            data.pop("prev_control", None)
            data.update(
                steps=self.steps[e],
                time=self.router.times[e],
                ticks=self.router.ticks[e],
                episode_steps=int(self.rl.episode_length_buf[e]),
            )
            result.append(plain(data))
        return result

    def close(self):
        try:
            self.env.close()
        finally:
            self.app.close()
