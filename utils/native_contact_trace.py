"""Read native tube SDF/force evidence without recomputing force or stepping physics."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from utils.isaaclab_compat import as_torch, simulation_snapshot

PAD_NAMES = tuple(f"{side}.{finger}" for side in ("left", "right") for finger in ("finger_0", "finger_1"))
TUBE_LABELS = ("tube0", "tube1", "tube2")


def array_copy(value):
    value = as_torch(value)
    if hasattr(value, "detach"):
        return value.detach().clone().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy().copy()
    return np.asarray(value).copy()


def query_native_sdf(sensor):
    """Use the same official coordinate boundary with existing actual taxel positions."""
    import warp as wp

    body = wp.to_torch(sensor._contact_object_body_view.get_transforms())
    points = sensor._data.tactile_points_pos_w
    local, _ = sensor._transform_points_to_contact_object_local(points, body[:, :3], body[:, 3:])
    values = wp.to_torch(sensor._contact_object_sdf_view.get_sdf_and_gradients(wp.from_torch(local)))
    return values[..., -1][0]


def native_task_inventory(env):
    """Inspect live USD/native paths; sensor mounting joints remain legitimate."""
    from pxr import UsdPhysics

    manager = env.scene_manager
    stage = env.tactile_manager.stage
    roots, objects = {}, {}
    for label in (*TUBE_LABELS, "slot"):
        name = manager.layout_manager.get_instance_name(0, label=label)
        candidates = dict(manager._rigid_and_dynamic_objects[0])
        candidates.update(manager._geometry_objects[0])
        if name not in candidates:
            raise RuntimeError(f"Native task inventory missing {label}: {name}")
        obj = candidates[name]
        root = str(obj.prim_path)
        prim = stage.GetPrimAtPath(root)
        if not prim.IsValid():
            raise RuntimeError(f"Native task inventory prim missing: {root}")
        roots[label] = root
        objects[label] = {"prim_path": root, "wrapper_type": type(obj).__name__}
        if label in TUBE_LABELS:
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Native tube has no RigidBodyAPI: {root}")
            body = UsdPhysics.RigidBodyAPI(prim)
            enabled = body.GetRigidBodyEnabledAttr().Get() is not False
            kinematic = body.GetKinematicEnabledAttr().Get() is True
            if not enabled or kinematic:
                raise RuntimeError(f"Native tube must be a dynamic enabled body: {root}")
            objects[label].update(
                rigid_body_enabled=enabled, kinematic_enabled=kinematic, flag_source="actual_USD_RigidBodyAPI"
            )
        else:
            objects[label].update(
                physics_role="static_geometry_rack", rigid_body_api=prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
    for label, root in roots.items():
        objects[label]["collision_approximations"] = [
            {
                "prim_path": str(prim.GetPath()),
                "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
            }
            for prim in stage.Traverse()
            if (str(prim.GetPath()) == root or str(prim.GetPath()).startswith(root + "/"))
            and prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        ]
    active_paths, forbidden, attached = [], [], []
    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        path = str(prim.GetPath())
        active_paths.append(path)
        if any(
            word in path.lower()
            for word in ("nut", "probe", "contact_calibration", "tactilecontactdemo", "tactile_contact_demo")
        ):
            forbidden.append(path)
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = [str(p) for relation in (joint.GetBody0Rel(), joint.GetBody1Rel()) for p in relation.GetTargets()]
            if any(
                target == roots[label] or target.startswith(roots[label] + "/")
                for label in TUBE_LABELS
                for target in targets
            ):
                attached.append({"joint": path, "targets": targets})
    calibration = getattr(manager, "_tactile_calibration_objects", None)
    if forbidden or attached or calibration and any(calibration):
        raise RuntimeError(f"Standard tube task contains forbidden fixtures or attachments: {forbidden}, {attached}")
    if len(manager._rigid_and_dynamic_objects[0]) != 3:
        raise RuntimeError("Standard insert_tubes must contain exactly three task tube bodies")
    slot_name = manager.layout_manager.get_instance_name(0, label="slot")
    ambient_geometry = []
    for name, obj in manager._geometry_objects[0].items():
        if name == slot_name:
            continue
        if not str(name).startswith("camera_stand_"):
            raise RuntimeError(f"Unexpected additional task geometry in standard insert_tubes: {name}")
        ambient_geometry.append(
            {"instance": name, "prim_path": str(obj.prim_path), "role": "original_canonical_camera_stand"}
        )
    time_s, frame = simulation_snapshot(env.sim.sim)
    return {
        "status": "passed",
        "physics_frame": frame,
        "timestamp_ns": round(time_s * 1e9),
        "objects": objects,
        "ambient_geometry": ambient_geometry,
        "active_prim_paths": active_paths,
        "forbidden_fixture_paths": forbidden,
        "tube_fixed_joints": attached,
        "tube_joint_constraints": attached,
    }


class NativeContactTrace:
    """Stream per-collider sensor evidence to disk and publish an independent NPZ."""

    def __init__(self, env, output, capacity, *, query=None, inventory=None):
        self.env, self.directory, self.capacity = env, Path(output), int(capacity)
        self.query = query or query_native_sdf
        self.inventory = inventory or native_task_inventory
        self.count = 0
        self._arrays = {}
        self._closed = False
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scratch = self.directory / ".native-contact-trace"
        self.scratch.mkdir()
        self.path = self.directory / "native-contact-trace.npz"
        manager = env.tactile_manager
        self.paths = tuple(manager.contact_paths_by_env[0])
        if len(self.paths) != 3:
            raise RuntimeError("Standard tube trace requires exactly three native tube mesh colliders")
        roots = {}
        for label in TUBE_LABELS:
            name = env.scene_manager.layout_manager.get_instance_name(0, label=label)
            roots[label] = str(env.scene_manager._rigid_and_dynamic_objects[0][name].prim_path)
        self.labels = []
        for path in self.paths:
            matches = [label for label, root in roots.items() if path == root or path.startswith(root + "/")]
            if len(matches) != 1:
                raise RuntimeError(f"Trace collider is not an unambiguous native tube: {path}")
            self.labels.append(matches[0])
        if set(self.labels) != set(TUBE_LABELS):
            raise RuntimeError("Trace colliders must cover each of the three native tubes")
        self.sensors = []
        for side in ("left", "right"):
            for finger in ("finger_0", "finger_1"):
                sensors = manager.sensors[side][finger][0]
                if len(sensors) != 3:
                    raise RuntimeError("Each native pad must have one sensor per actual tube")
                for sensor, path in zip(sensors, self.paths, strict=True):
                    if str(sensor.cfg.contact_object_prim_path_expr) != path:
                        raise RuntimeError("Native sensor collider order disagrees with trace metadata")
                self.sensors.append(sensors)
        for sensors in self.sensors:
            for sensor in sensors:
                if getattr(sensor._contact_object_body_view, "count", None) != 1:
                    raise RuntimeError("Native tube collider must resolve exactly one live rigid body")
        before = simulation_snapshot(env.sim.sim)
        self.query_backends = [
            [getattr(sensor, "contact_query_backend", None) for sensor in sensors] for sensors in self.sensors
        ]
        self.query_geometry = [
            [getattr(sensor, "contact_query_metadata", {}) for sensor in sensors] for sensors in self.sensors
        ]
        self.initial_inventory = self.inventory(env)
        if simulation_snapshot(env.sim.sim) != before:
            raise RuntimeError("Initial native inventory advanced physics")
        (self.directory / "native-task-inventory.json").write_text(
            json.dumps({"initial": self.initial_inventory}, indent=2)
        )

    def append(self, sync):
        context = self.env.sim.sim
        start = simulation_snapshot(context)
        if (start[1], round(start[0] * 1e9)) != (sync["physics_frame"], sync["timestamp_ns"]):
            raise RuntimeError("Native contact trace clock disagrees with sensor capture")
        if self.count >= self.capacity or self._closed:
            raise RuntimeError("Native contact trace is full or closed")
        values = {
            "sdf_m": [],
            "normal_force_N": [],
            "shear_force_N": [],
            "penetration_depth_m": [],
            "contact_body_pose_w_xyzw": [],
            "gel_pose_w_xyzw": [],
            "taxel_points_pos_w_m": [],
        }
        for sensors in self.sensors:
            per_pad = {key: [] for key in values}
            for sensor in sensors:
                # Access stored buffers directly; .data could lazily refresh sensing.
                normal = array_copy(sensor._data.tactile_normal_force[0])
                shear = array_copy(sensor._data.tactile_shear_force[0])
                penetration = array_copy(sensor._data.penetration_depth[0])
                body = array_copy(as_torch(sensor._contact_object_body_view.get_transforms()))[0]
                gel = array_copy(as_torch(sensor._elastomer_body_view.get_transforms()))[0]
                points = array_copy(sensor._data.tactile_points_pos_w[0])
                if body.shape != (7,) or gel.shape != (7,) or points.shape != (*normal.shape, 3):
                    raise RuntimeError("Native contact trace pose/point dimensions are invalid")
                sdf = array_copy(self.query(sensor))
                if normal.ndim != 1 or shear.shape != (*normal.shape, 2) or sdf.shape != normal.shape:
                    raise RuntimeError("Native SDF/force trace has an invalid taxel shape")
                if penetration.shape != normal.shape or not all(
                    np.isfinite(a).all() for a in (normal, shear, sdf, penetration, body, gel, points)
                ):
                    raise RuntimeError("Native SDF/force trace contains invalid values")
                if not np.array_equal(normal, array_copy(sensor._data.tactile_normal_force[0])) or not np.array_equal(
                    shear, array_copy(sensor._data.tactile_shear_force[0])
                ):
                    raise RuntimeError("Diagnostic SDF query changed native force buffers")
                for key, value in zip(per_pad, (sdf, normal, shear, penetration, body, gel, points), strict=True):
                    per_pad[key].append(value)
            for key in values:
                values[key].append(np.stack(per_pad[key]))
        if simulation_snapshot(context) != start:
            raise RuntimeError("Physics advanced during native contact diagnostics")
        sample = {key: np.stack(value) for key, value in values.items()}
        sample.update(physics_frame=np.int64(sync["physics_frame"]), timestamp_ns=np.int64(sync["timestamp_ns"]))
        for key, value in sample.items():
            if key not in self._arrays:
                self._arrays[key] = np.lib.format.open_memmap(
                    self.scratch / (key + ".npy"), mode="w+", dtype=value.dtype, shape=(self.capacity, *value.shape)
                )
            self._arrays[key][self.count] = value
        self.count += 1

    def complete(self):
        if self._closed:
            return self.path
        if not self.count:
            raise RuntimeError("Cannot publish an empty native contact trace")
        before = simulation_snapshot(self.env.sim.sim)
        final = self.inventory(self.env)
        if simulation_snapshot(self.env.sim.sim) != before:
            raise RuntimeError("Native inventory advanced physics")
        temporary = self.path.with_suffix(".npz.partial")
        for value in self._arrays.values():
            value.flush()
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **{key: value[: self.count] for key, value in self._arrays.items()},
                collider_labels=np.asarray(self.labels),
                collider_paths=np.asarray(self.paths),
                pad_names=np.asarray(PAD_NAMES),
            )
        temporary.replace(self.path)
        (self.directory / "native-task-inventory.json").write_text(
            json.dumps({"initial": self.initial_inventory, "final": final}, indent=2) + "\n"
        )
        (self.directory / "native-contact-trace.json").write_text(
            json.dumps(
                {
                    "status": "captured",
                    "task_success_claimed": False,
                    "file": str(self.path.resolve()),
                    "frames": self.count,
                    "normal_force_frame": "official_taxel_z",
                    "shear_force_frame": "official_taxel_xy",
                    "sdf_units": "m",
                    "source": "actual_per_collider_configured_SDF_query_and_unmodified_existing_force_buffers",
                    "contact_query_backends": self.query_backends,
                    "contact_query_geometry": self.query_geometry,
                    "physics_held_fixed": True,
                },
                indent=2,
            )
            + "\n"
        )
        # Close mappings before unlinking on /mnt NFS. A loop-local reference
        # otherwise keeps the last .npy open and creates an undeletable .nfs file.
        for value in self._arrays.values():
            value._mmap.close()
        self._arrays.clear()
        shutil.rmtree(self.scratch)
        self._closed = True
        return self.path
