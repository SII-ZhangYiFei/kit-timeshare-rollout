"""Process-local native routing derived from the validated rigid-body adapter."""

from pathlib import Path
import json, time, types, os
import numpy as np
import torch
from .scenes import SceneRouter


def prepare_behavior(og, run):
    """Keep controllers and time-dependent object-state updates environment-local."""
    import inspect, textwrap, hashlib, os
    from omnigibson.controllers.controller_view import ControllerView
    from omnigibson.object_states.tensorized_value_state import TensorizedValueState
    from omnigibson.utils.usd_utils import RigidContactAPI

    sim = og.sim
    manifest = {}
    if os.environ.get("BATCH_READ_ONLY_MATERIAL_QUERY", "0") == "1":
        from omnigibson.prims.xform_prim import XFormPrim
        from pxr import Usd, UsdShade, Sdf

        original_has_material = XFormPrim.has_material
        query_counts = {"skipped_empty_api_writes": 0, "delegated_queries": 0}

        def has_material_readonly(prim_self):
            prim = prim_self.prim
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                rel = prim.GetRelationship("material:binding")
                if not rel or not rel.HasAuthoredTargets():
                    query_counts["skipped_empty_api_writes"] += 1
                    return False
            query_counts["delegated_queries"] += 1
            return original_has_material(prim_self)

        XFormPrim.has_material = has_material_readonly
        sim._independent_material_query_counts = query_counts
        native = Usd.Stage.CreateInMemory()
        a = native.DefinePrim("/empty", "Xform")
        b = native.DefinePrim("/legacy", "Xform")
        m = UsdShade.Material.Define(native, "/mat")
        if not (
            not a.HasAPI(UsdShade.MaterialBindingAPI)
            and (not a.GetRelationship("material:binding"))
        ):
            raise RuntimeError("Unsupported native SDK or scene ownership invariant")
        if (
            not UsdShade.MaterialBindingAPI(a).GetDirectBinding().GetMaterialPath()
            == Sdf.Path.emptyPath
        ):
            raise RuntimeError("Unsupported native SDK or scene ownership invariant")
        b.CreateRelationship("material:binding").SetTargets([m.GetPath()])
        if not not b.HasAPI(UsdShade.MaterialBindingAPI):
            raise RuntimeError("Unsupported native SDK or scene ownership invariant")
        if (
            not UsdShade.MaterialBindingAPI(b).GetDirectBinding().GetMaterialPath()
            == m.GetPath()
        ):
            raise RuntimeError("Unsupported native SDK or scene ownership invariant")
        applied = UsdShade.MaterialBindingAPI.Apply(b)
        if not applied.GetDirectBinding().GetMaterialPath() == m.GetPath():
            raise RuntimeError("Unsupported native SDK or scene ownership invariant")
        source = inspect.getsource(original_has_material)
        corrected = inspect.getsource(has_material_readonly)
        (run / "material_query_original.py").write_text(source)
        (run / "material_query_readonly.py").write_text(corrected)
        manifest["read_only_empty_material_query"] = {
            "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "patched_sha256": hashlib.sha256(corrected.encode()).hexdigest(),
            "native_usd_empty_and_legacy_binding_checks_pass": True,
            "scope": "Process-only read of empty unbound nodes; SDK implementation retained for all applied schemas or authored material bindings.",
        }
    original_key = ControllerView._make_key.__func__

    def unique_key(cls, articulation_root_path, body_part, controller_cfg):
        return (
            original_key(cls, articulation_root_path, body_part, controller_cfg)
            + "__independent__"
            + articulation_root_path
        )

    ControllerView._make_key = classmethod(unique_key)

    def step_selected(cls):
        router = getattr(sim, "_independent_router", None)
        for controller in cls._controller_groups.values():
            if router is None or any(
                (
                    p.startswith(router.roots[router.active] + "/")
                    for p in controller._articulation_root_paths
                )
            ):
                controller.step()

    ControllerView.step_all = classmethod(step_selected)
    original_global = TensorizedValueState.global_update.__func__

    def global_selected(cls):
        router = getattr(sim, "_independent_router", None)
        if router is None:
            return original_global(cls)
        if len(cls.VALUES) == 0:
            return
        original = cls.VALUES
        candidate = cls._update_values(values=original)
        mask = torch.tensor(
            [o.scene.idx == router.active for o in cls.IDX_OBJS],
            dtype=torch.bool,
            device=original.device,
        )
        changed = mask & torch.any(
            (candidate != original).reshape(len(original), -1), dim=-1
        )
        for idx in torch.where(changed)[0].tolist():
            cls.IDX_OBJS[idx].state_updated()
        result = original.clone()
        result[mask] = candidate[mask]
        cls.VALUES = result

    TensorizedValueState.global_update = classmethod(global_selected)

    def scenes_selected(s):
        router = getattr(s, "_independent_router", None)
        return (
            s.scenes
            if router is None
            else [s.scenes[router.active]] if router.active < len(s.scenes) else []
        )

    for name in ["_on_pre_physics_step", "_non_physics_step", "step"]:
        fn = inspect.unwrap(getattr(sim, name).__func__)
        source = textwrap.dedent(inspect.getsource(fn))
        needle = "for scene in self.scenes:"
        if not needle in source:
            raise RuntimeError((name, source[:150]))
        corrected = source.replace(needle, "for scene in _independent_scenes(self):")
        path = run / (name.strip("_") + "_independent.py")
        path.write_text(corrected)
        namespace = dict(fn.__globals__)
        namespace["_independent_scenes"] = scenes_selected
        exec(compile(corrected, str(path), "exec"), namespace)
        setattr(sim, name, types.MethodType(namespace[name], sim))
        manifest[name] = {
            "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "patched_sha256": hashlib.sha256(corrected.encode()).hexdigest(),
        }

    def selected_contacts(api):
        router = getattr(sim, "_independent_router", None)
        return (
            list(api._CONTACT_VIEW)
            if router is None
            else [router.active] if router.active in api._CONTACT_VIEW else []
        )

    for name in ["add_contacts_from_physics_step", "update_contact_cache"]:
        fn = inspect.unwrap(getattr(RigidContactAPI, name).__func__)
        source = textwrap.dedent(inspect.getsource(fn))
        needle = "for scene_idx in list(self._CONTACT_VIEW.keys()):"
        if not needle in source:
            raise RuntimeError(name)
        corrected = source.replace(
            needle, "for scene_idx in _independent_contact_keys(self):"
        )
        path = run / (name + "_independent.py")
        path.write_text(corrected)
        namespace = dict(fn.__globals__)
        namespace["_independent_contact_keys"] = selected_contacts
        exec(compile(corrected, str(path), "exec"), namespace)
        setattr(
            RigidContactAPI, name, types.MethodType(namespace[name], RigidContactAPI)
        )
        manifest[name] = {
            "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "patched_sha256": hashlib.sha256(corrected.encode()).hexdigest(),
        }

    def independent_time(s):
        router = getattr(s, "_independent_router", None)
        return (
            s._sim_context.current_time
            if router is None
            else router.times[router.active]
        )

    type(sim).current_time = property(independent_time)
    (run / "independent_process_patches.json").write_text(
        json.dumps(manifest, indent=2)
    )


def configure_behavior_initial(og, run, n, scene_spacing=40.0):
    """Assign owners before play and initialize/reset only the selected world."""
    import copy, math
    from contextlib import nullcontext
    import omnigibson.lazy as lazy
    from omnigibson.scenes.scene_base import Scene

    sim = og.sim

    def usd_scope():
        return nullcontext() if sim._editing_usd else sim.editing_usd()

    with usd_scope():
        router = SceneRouter(
            sim.stage,
            sim._physics_context,
            [f"/World/scene_{i}" for i in range(n)],
            run,
            allow_empty=True,
        )
    router.initializing = True
    sim._independent_router = router
    import inspect, textwrap, hashlib, sys
    from pxr import Usd, UsdPhysics, Sdf
    from omnigibson.utils import usd_utils

    original_add = usd_utils.add_asset_to_stage

    def assign_new_asset(prim):
        path = str(prim.GetPath())
        e = next(
            (
                i
                for (i, r) in enumerate(router.roots)
                if path == r or path.startswith(r + "/")
            ),
            None,
        )
        if e is None:
            return
        count = 0
        target = [Sdf.Path(router.paths[e])]
        for item in Usd.PrimRange(prim):
            if item.HasAPI(UsdPhysics.RigidBodyAPI):
                api = UsdPhysics.RigidBodyAPI(item)
            elif item.HasAPI(UsdPhysics.CollisionAPI):
                api = UsdPhysics.CollisionAPI(item)
            else:
                continue
            api.CreateSimulationOwnerRel().SetTargets(target)
            count += 1
        with (run / "new_asset_owners.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "path": path,
                        "env": e,
                        "owner": router.paths[e],
                        "bodies_and_colliders": count,
                        "phase": "Inside add_asset_to_stage editing_usd, before first native sync.",
                    }
                )
                + "\n"
            )

    source = textwrap.dedent(inspect.getsource(original_add))
    needle = "        return prim"
    if not source.count(needle) == 1:
        raise RuntimeError("Unsupported native SDK or scene ownership invariant")
    patched = source.replace(
        needle, "        _independent_assign_asset(prim)\n" + needle
    )
    target = run / "add_asset_to_stage_independent.py"
    target.write_text(patched)
    namespace = dict(original_add.__globals__)
    namespace["_independent_assign_asset"] = assign_new_asset
    exec(compile(patched, str(target), "exec"), namespace)
    replacement = namespace["add_asset_to_stage"]
    aliases = []
    for name, module in list(sys.modules.items()):
        if (
            module is not None
            and name.startswith("omnigibson")
            and (vars(module).get("add_asset_to_stage") is original_add)
        ):
            setattr(module, "add_asset_to_stage", replacement)
            aliases.append(name)
    (run / "new_asset_owner_patch.json").write_text(
        json.dumps(
            {
                "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
                "aliases": aliases,
                "scope": "Process-local asset reference transaction only; installed SDK/data remain read-only.",
            },
            indent=2,
        )
    )

    def select_initial(e):
        with usd_scope():
            router.select_initial(e)

    select_initial(0)
    old_play = sim.play

    def routed_play():
        with usd_scope():
            router.refresh_ownership(
                only_env=router.active if router.initializing else None
            )
        return old_play()

    sim.play = routed_play
    old_update_handles = sim.update_handles

    def owned_update_handles():
        if router.initializing:
            with usd_scope():
                router.refresh_ownership(only_env=router.active)
        return old_update_handles()

    sim.update_handles = owned_update_handles
    old_post = sim._on_post_physics_step

    def initial_post():
        old_post()
        if router.initializing:
            dt = float(sim.get_physics_dt())
            router.times[router.active] += dt
            router.ticks[router.active] += 1

    sim._on_post_physics_step = initial_post
    old_load = Scene._load_scene_prim_with_objects
    cols = math.ceil(math.sqrt(n))
    layout = []

    def grid_load(scene, *args, **kwargs):
        (edge, position) = old_load(scene, *args, **kwargs)
        (lower, upper) = lazy.omni.usd.get_context().compute_path_world_bounding_box(
            scene.prim_path
        )
        size = np.asarray(upper) - np.asarray(lower)
        if not max(size[:2]) < scene_spacing - 10:
            raise RuntimeError(("Scene exceeds configured spacing", scene.idx, size))
        desired = torch.tensor(
            [scene.idx % cols * scene_spacing, scene.idx // cols * scene_spacing, 0.0]
        )
        scene._scene_prim.set_position_orientation(
            position=desired, orientation=torch.tensor([0.0, 0.0, 0.0, 1.0])
        )
        layout.append(
            {
                "env": scene.idx,
                "origin": desired.tolist(),
                "pre_grid_extent": size.tolist(),
                "pitch_m": scene_spacing,
            }
        )
        (run / "scene_grid_layout.json").write_text(json.dumps(layout, indent=2))
        return (edge, desired)

    Scene._load_scene_prim_with_objects = grid_load

    def vector_init(vec, num_envs, config):
        vec.num_envs = num_envs
        if og.sim is not None:
            og.sim.stop()
        vec.envs = []
        for e in range(num_envs):
            (run / "initialization_progress.json").write_text(
                json.dumps(
                    {"phase": "construct", "env": e, "n": num_envs, "time": time.time()}
                )
            )
            select_initial(e)
            vec.envs.append(
                og.Environment(configs=copy.deepcopy(config), in_vec_env=True)
            )
        og.sim.play()
        for e, env in enumerate(vec.envs):
            (run / "initialization_progress.json").write_text(
                json.dumps(
                    {
                        "phase": "post_play_reset",
                        "env": e,
                        "n": num_envs,
                        "time": time.time(),
                    }
                )
            )
            select_initial(e)
            env.post_play_load()

    def vector_reset(vec, get_obs=True, **kwargs):
        obs = []
        infos = []
        for e, env in enumerate(vec.envs):
            (run / "initialization_progress.json").write_text(
                json.dumps(
                    {
                        "phase": "explicit_reset",
                        "env": e,
                        "n": len(vec.envs),
                        "time": time.time(),
                    }
                )
            )
            select_initial(e)
            result = env.reset(get_obs=get_obs, **kwargs)
            if get_obs:
                obs.append(result[0])
                infos.append(result[1])
        return (obs, infos) if get_obs else None

    og.VectorEnvironment.__init__ = vector_init
    og.VectorEnvironment.reset = vector_reset
    return router
