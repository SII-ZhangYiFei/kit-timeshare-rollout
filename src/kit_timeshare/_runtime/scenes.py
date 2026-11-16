"""Process-local native routing derived from the validated rigid-body adapter."""

from pathlib import Path
import json, time, types


class SceneRouter:

    def __init__(self, stage, physics_context, roots, run, allow_empty=False):
        from pxr import Usd, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Sdf

        self.pc = physics_context
        self.stage = stage
        self.roots = roots
        self.run = run
        self.n = len(roots)
        self.active = 0
        self.times = [0.0] * self.n
        self.ticks = [0] * self.n
        self.before_physics = None
        self.after_physics = None
        self.UsdPhysics = UsdPhysics
        self.PhysxSchema = PhysxSchema
        source = stage.GetPrimAtPath(physics_context.prim_path)
        if not source.IsValid():
            raise RuntimeError(physics_context.prim_path)
        self.paths = [str(source.GetPath())] + [
            f"/World/independentPhysics/scene_{i}" for i in range(1, self.n)
        ]
        for i, path in enumerate(self.paths):
            sc = UsdPhysics.Scene.Define(stage, path)
            if i:
                for attr in source.GetAttributes():
                    if attr.HasValue():
                        sc.GetPrim().CreateAttribute(
                            attr.GetName(), attr.GetTypeName(), attr.IsCustom()
                        ).Set(attr.Get())
            api = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim())
            api.CreateEnableGPUDynamicsAttr().Set(False)
            api.CreateBroadphaseTypeAttr().Set("MBP")
        self.sdf_paths = [Sdf.Path(p) for p in self.paths]
        self.ids = [PhysicsSchemaTools.sdfPathToInt(p) for p in self.sdf_paths]
        counts = [{"rigid_bodies": 0, "colliders": 0} for _ in roots]
        shared = []
        global_fixed = []
        for prim in stage.Traverse():
            is_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            is_col = prim.HasAPI(UsdPhysics.CollisionAPI)
            if not is_body and (not is_col):
                continue
            path = str(prim.GetPath())
            e = next(
                (
                    j
                    for (j, r) in enumerate(roots)
                    if path == r or path.startswith(r + "/")
                ),
                None,
            )
            if e is None:
                if is_body:
                    rb = UsdPhysics.RigidBodyAPI(prim)
                    disabled = rb.GetRigidBodyEnabledAttr().Get() is False
                    kinematic = rb.GetKinematicEnabledAttr().Get() is True
                    if not (disabled or kinematic):
                        raise RuntimeError(
                            ("Dynamic body outside environment roots", path)
                        )
                    if not not any(
                        (a.GetNumTimeSamples() for a in prim.GetAttributes())
                    ):
                        raise RuntimeError(("Animated global body", path))
                    global_fixed.append(
                        {
                            "path": path,
                            "rigid_body_disabled": disabled,
                            "kinematic": kinematic,
                        }
                    )
                    continue
                targets = [Sdf.Path(x) for x in self.paths]
                shared.append(path)
            else:
                targets = [Sdf.Path(self.paths[e])]
                counts[e]["rigid_bodies"] += int(is_body)
                counts[e]["colliders"] += int(is_col)
            api = (
                UsdPhysics.RigidBodyAPI(prim)
                if is_body
                else UsdPhysics.CollisionAPI(prim)
            )
            api.CreateSimulationOwnerRel().SetTargets(targets)
        if not (allow_empty or all((c["rigid_bodies"] > 0 for c in counts))):
            raise RuntimeError(counts)
        self.info = {
            "physics_scenes": self.paths,
            "physics_scene_ids": self.ids,
            "sdf_paths_retained": True,
            "environment_roots": roots,
            "actor_counts": counts,
            "shared_static_colliders": shared,
            "global_fixed_bodies": global_fixed,
            "method": "Native simulate_scene/fetch_results_scene; disabled automatic scene updates; retained scene/actor/contact objects.",
        }
        (run / "physics_scene_ownership.json").write_text(
            json.dumps(self.info, indent=2)
        )

    def refresh_ownership(self, only_env=None):
        from pxr import Sdf

        started = time.perf_counter()
        changed = []
        counts = [dict(c) for c in self.info["actor_counts"]]
        if only_env is None:
            counts = [{"rigid_bodies": 0, "colliders": 0} for _ in self.roots]
            prims = self.stage.Traverse()
        else:
            from pxr import Usd

            root = self.stage.GetPrimAtPath(self.roots[only_env])
            if not root.IsValid():
                return self.refresh_ownership()
            counts[only_env] = {"rigid_bodies": 0, "colliders": 0}
            prims = Usd.PrimRange(root)
        visited = 0
        for prim in prims:
            visited += 1
            body = prim.HasAPI(self.UsdPhysics.RigidBodyAPI)
            collider = prim.HasAPI(self.UsdPhysics.CollisionAPI)
            if not body and (not collider):
                continue
            path = str(prim.GetPath())
            e = (
                only_env
                if only_env is not None
                else next(
                    (
                        j
                        for (j, r) in enumerate(self.roots)
                        if path == r or path.startswith(r + "/")
                    ),
                    None,
                )
            )
            if e is None:
                if body:
                    rb = self.UsdPhysics.RigidBodyAPI(prim)
                    if not (
                        rb.GetRigidBodyEnabledAttr().Get() is False
                        or rb.GetKinematicEnabledAttr().Get() is True
                    ):
                        raise RuntimeError(("Unscoped dynamic body", path))
                    continue
                targets = [Sdf.Path(p) for p in self.paths]
            else:
                targets = [Sdf.Path(self.paths[e])]
                counts[e]["rigid_bodies"] += int(body)
                counts[e]["colliders"] += int(collider)
            api = (
                self.UsdPhysics.RigidBodyAPI(prim)
                if body
                else self.UsdPhysics.CollisionAPI(prim)
            )
            if api.GetSimulationOwnerRel().GetTargets() != targets:
                api.CreateSimulationOwnerRel().SetTargets(targets)
                changed.append(path)
        self.info["actor_counts"] = counts
        (self.run / "physics_scene_ownership.json").write_text(
            json.dumps(self.info, indent=2)
        )
        with (self.run / "ownership_refresh.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "scope_env": only_env,
                        "visited_prims": visited,
                        "changed_owners": len(changed),
                        "seconds": time.perf_counter() - started,
                        "time": time.time(),
                    }
                )
                + "\n"
            )
        return changed

    def install(self):
        for path in self.paths:
            self.PhysxSchema.PhysxSceneAPI(
                self.stage.GetPrimAtPath(path)
            ).CreateUpdateTypeAttr().Set(self.PhysxSchema.Tokens.Disabled)
        self.original_step = self.pc._step
        router = self

        def selected_step(pc, current_time, update_fabric=False):
            e = router.active
            dt = float(pc.get_physics_dt())
            if router.before_physics is not None:
                router.before_physics()
            pc._physx_sim_interface.simulate_scene(router.ids[e], dt, router.times[e])
            pc._physx_sim_interface.fetch_results_scene(router.ids[e])
            router.times[e] += dt
            router.ticks[e] += 1
            if router.after_physics is not None:
                router.after_physics()
            if update_fabric and pc._physx_fabric_interface is not None:
                pc._physx_fabric_interface.update(router.times[e], dt)

        self.pc._step = types.MethodType(selected_step, self.pc)

    def select(self, e):
        if not 0 <= e < self.n:
            raise RuntimeError("Unsupported native SDK or scene ownership invariant")
        self.active = e

    def select_initial(self, e):
        self.select(e)
        for i, path in enumerate(self.paths):
            attr = self.PhysxSchema.PhysxSceneAPI(
                self.stage.GetPrimAtPath(path)
            ).CreateUpdateTypeAttr()
            target = "Synchronous" if i == e else "Disabled"
            if attr.Get() != target:
                attr.Set(target)
