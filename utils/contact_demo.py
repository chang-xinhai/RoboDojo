"""Real kinematic contact calibration for the four mounted X5 TacSL pads.

Fixtures are authored before PhysX initialization. Runtime commands use native
kinematic targets, never USD edits or synthetic sensor images/forces. This is a
calibration rig with prescribed probe motion, not a learned grasping policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.isaaclab_compat import simulation_snapshot

ROOT = Path(__file__).resolve().parents[1]
PADS = tuple((side, finger) for side in ("left", "right") for finger in ("finger_0", "finger_1"))
PHASES = (
    "baseline",
    "press",
    "hold",
    "slide_positive",
    "slide_negative",
    "slide_center",
    "twist_positive",
    "twist_negative",
    "unload",
    "released",
)


def configure_contact_physics(sim_context, *, install_bridge=True):
    """Enable real CPU-native kinematic targets before the first PhysX warmup.

    Isaac Sim 6.1's DirectGPU rigid view does not implement kinematic targets.
    GPU dynamics still run on CUDA, while actual native CPU pose/velocity
    buffers cross the established host-readback bridge for CUDA processing.
    TacSL reports its real Warp triangle-mesh signed-distance query backend.
    """
    if getattr(sim_context.physics_manager, "_view", None) is not None:
        raise RuntimeError("Contact host readback must be configured before native physics initialization")
    if getattr(sim_context, "_robodojo_contact_readback_configured", False):
        return getattr(sim_context, "_robodojo_contact_readback_handle", None)
    sim_context._robodojo_contact_readback_previous_marker = bool(
        sim_context.get_setting("/isaaclab/robodojo/contact_calibration_readback")
    )
    sim_context.set_setting("/physics/suppressReadback", False)
    sim_context.set_setting("/physics/updateToUsd", True)
    sim_context.set_setting("/isaaclab/robodojo/particle_readback", True)
    sim_context.set_setting("/isaaclab/robodojo/contact_calibration_readback", True)
    sim_context._robodojo_contact_readback_configured = True
    if install_bridge:
        from utils.particle_readback_bridge import install_particle_readback_bridge

        handle = install_particle_readback_bridge(sim_context.physics_manager)
        sim_context._robodojo_contact_readback_handle = handle
        return handle
    return None


def close_contact_physics(sim_context):
    """Drop this profile's callback and restore its provenance marker on teardown.

    Native readback settings remain valid until the owning context closes;
    changing its physics pipeline with live native handles is not supported.
    A new SimulationContext selects its own backend/readback configuration.
    """
    handle = getattr(sim_context, "_robodojo_contact_readback_handle", None)
    if handle is not None:
        handle.deregister()
        sim_context._robodojo_contact_readback_handle = None
    if getattr(sim_context, "_robodojo_contact_readback_configured", False):
        sim_context.set_setting(
            "/isaaclab/robodojo/contact_calibration_readback",
            sim_context._robodojo_contact_readback_previous_marker,
        )
        sim_context._robodojo_contact_readback_configured = False


@dataclass(frozen=True)
class ContactDemoConfig:
    shape: str = "official_nut"
    asset_path: str | None = None
    asset_scale: float = 0.6
    indentation_m: float = 0.0006
    clearance_m: float = 0.003
    slide_m: float = 0.002
    twist_deg: float = 20.0
    phase_duration_s: float = 0.4
    annulus_inner_radius_m: float = 0.004
    annulus_outer_radius_m: float = 0.009
    annulus_height_m: float = 0.004
    annulus_sections: int = 128

    def __post_init__(self):
        if self.shape not in ("official_nut", "annulus"):
            raise ValueError("Contact demo shape must be official_nut or annulus")
        positive = (
            self.asset_scale,
            self.indentation_m,
            self.clearance_m,
            self.slide_m,
            self.twist_deg,
            self.phase_duration_s,
            self.annulus_inner_radius_m,
            self.annulus_outer_radius_m,
            self.annulus_height_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Contact demo dimensions, durations and amplitudes must be positive and finite")
        if self.indentation_m > 0.0008 or self.slide_m > 0.003 or self.twist_deg > 30:
            raise ValueError("Contact demo exceeds the calibrated 0.8mm indentation/3mm slide/30deg bounds")
        if not 0 < self.asset_scale <= 1:
            raise ValueError("Official nut uniform scale must be in (0,1]")
        if self.annulus_inner_radius_m >= self.annulus_outer_radius_m:
            raise ValueError("Annulus inner radius must be smaller than outer radius")
        if not isinstance(self.annulus_sections, int) or self.annulus_sections < 16:
            raise ValueError("Annulus must contain at least 16 sections")

    @property
    def duration_s(self):
        return len(PHASES) * self.phase_duration_s


def phase_command(config, elapsed_time_s):
    """C2-continuous prescribed pose offsets; no sensor signal is generated."""
    if not math.isfinite(elapsed_time_s) or elapsed_time_s < 0:
        raise ValueError("Contact elapsed time must be finite and nonnegative")
    index = min(int(elapsed_time_s / config.phase_duration_s), len(PHASES) - 1)
    u = np.clip(elapsed_time_s / config.phase_duration_s - index, 0.0, 1.0)
    blend = float(u**3 * (10 + u * (-15 + 6 * u)))
    gap, depth, slide, twist = config.clearance_m, config.indentation_m, config.slide_m, math.radians(config.twist_deg)
    # Positive separation denotes an unloaded gap; negative separation is the
    # commanded geometric penetration, distinct from measured SDF indentation.
    endpoints = (
        ((gap, 0, 0), (gap, 0, 0)),
        ((gap, 0, 0), (-depth, 0, 0)),
        ((-depth, 0, 0), (-depth, 0, 0)),
        ((-depth, 0, 0), (-depth, slide, 0)),
        ((-depth, slide, 0), (-depth, -slide, 0)),
        ((-depth, -slide, 0), (-depth, 0, 0)),
        ((-depth, 0, 0), (-depth, 0, twist)),
        ((-depth, 0, twist), (-depth, 0, -twist)),
        ((-depth, 0, -twist), (gap, 0, 0)),
        ((gap, 0, 0), (gap, 0, 0)),
    )
    start, end = (np.asarray(value, dtype=np.float64) for value in endpoints[index])
    values = start + blend * (end - start)
    return {
        "phase": PHASES[index],
        "elapsed_time_s": float(elapsed_time_s),
        "normal_separation_m": float(values[0]),
        "tangent_offset_m": float(values[1]),
        "twist_rad": float(values[2]),
    }


def annulus_mesh(config):
    """Closed triangulated ring, with its bearing-face normal along local Z."""
    count = config.annulus_sections
    angle = np.arange(count) * (2 * math.pi / count)
    vertices = []
    for z in (-config.annulus_height_m / 2, config.annulus_height_m / 2):
        for radius in (config.annulus_inner_radius_m, config.annulus_outer_radius_m):
            vertices.extend(np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.full(count, z))))
    faces = []
    for i in range(count):
        j = (i + 1) % count
        # Eight triangles per section: bottom, top, outer and inner walls.
        faces.extend(
            (
                (i, j + count, i + count),
                (i, j, j + count),
                (i + 2 * count, i + 3 * count, j + 3 * count),
                (i + 2 * count, j + 3 * count, j + 2 * count),
                (i + count, j + count, j + 3 * count),
                (i + count, j + 3 * count, i + 3 * count),
                (i, i + 2 * count, j + 2 * count),
                (i, j + 2 * count, j),
            )
        )
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def load_contact_geometry(config):
    """Read actual official collider topology or the explicitly selected ring."""
    if config.shape == "annulus":
        vertices, faces = annulus_mesh(config)
        source = {"kind": "procedural_annulus", "geometry": asdict(config)}
    else:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        path = (
            Path(config.asset_path) if config.asset_path else ROOT / "Assets/TacSL/contact_objects/factory_nut_m16.usd"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Official TacSL nut asset required: {path}")
        stage = Usd.Stage.Open(str(path))
        if stage is None or not stage.GetDefaultPrim():
            raise ValueError(f"Invalid official nut USD stage: {path}")
        colliders = [
            prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(colliders) != 1:
            raise ValueError("Official nut fixture requires one actual mesh collider")
        mesh = UsdGeom.Mesh(colliders[0])
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not len(counts) or np.any(counts != 3):
            raise ValueError("Official nut collider must contain native triangle topology")
        cache = UsdGeom.XformCache()
        matrix = cache.GetLocalToWorldTransform(colliders[0])
        vertices = np.asarray(
            [matrix.Transform(Gf.Vec3d(*point)) for point in mesh.GetPointsAttr().Get()], dtype=np.float64
        )
        metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(metres_per_unit) or metres_per_unit <= 0:
            raise ValueError("Official nut source stage must declare a positive physical length unit")
        vertices *= metres_per_unit
        source_center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        vertices = ((vertices - source_center) * config.asset_scale).astype(np.float32)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        source = {
            "kind": "official_tacsl_factory_nut_m16",
            "asset_path": str(path.resolve()),
            "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_mesh_path": str(colliders[0].GetPath()),
            "source_bbox_center_m": source_center.tolist(),
            "source_meters_per_unit": metres_per_unit,
            "mesh_orientation": str(mesh.GetOrientationAttr().Get()),
            "uniform_scale": config.asset_scale,
            "geometry_transform": "actual USD collider-to-world transform, bbox recenter, uniform scale",
            "render_geometry": "same official collider triangles",
        }
    if not len(vertices) or not np.isfinite(vertices).all() or not len(faces):
        raise ValueError("Contact mesh geometry must be finite and nonempty")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Contact mesh triangle indices refer to missing vertices")
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.any(area > 0):
        raise ValueError("Contact fixture has no valid triangle surface")
    degenerate = np.flatnonzero(area == 0)
    if config.shape == "annulus" and len(degenerate):
        raise ValueError("Procedural annulus contains degenerate triangles")
    # The official rigid nut has 34 exactly zero-area source triangles. Its
    # own NVIDIA TacSL demo passes the raw collider to PhysX rigid cooking,
    # which cleans degenerate triangles. Preserve the source here and record
    # the diagnosis; only real native contact tests can certify cooking.
    source.update(
        zero_area_source_triangle_count=len(degenerate),
        zero_area_source_triangle_ids=degenerate.tolist(),
        cooking_policy="raw source topology; native PhysX rigid mesh cooking",
    )
    source.update(
        vertex_count=len(vertices),
        triangle_count=len(faces),
        bounds_m=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        geometry_sha256=hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
    )
    return vertices, faces, source


def quat_xyzw_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("Native quaternion must be a finite nonzero XYZW vector")
    x, y, z, w = q / np.linalg.norm(q)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def pose_at_pad(center, gel_quaternion_xyzw, support_z_m, command):
    """Map actual gel pose to a probe pose whose local -Z face approaches gel."""
    from scipy.spatial.transform import Rotation

    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Measured gel center must be a finite world XYZ vector")
    gel = quat_xyzw_matrix(gel_quaternion_xyzw)
    tangent = gel[:, 0]
    outward = -gel[:, 1]
    basis = np.column_stack((tangent, np.cross(outward, tangent), outward))
    twist = command["twist_rad"]
    local_twist = np.asarray(((math.cos(twist), -math.sin(twist), 0), (math.sin(twist), math.cos(twist), 0), (0, 0, 1)))
    position = center + outward * (support_z_m + command["normal_separation_m"]) + tangent * command["tangent_offset_m"]
    return np.concatenate((position, Rotation.from_matrix(basis @ local_twist).as_quat())).astype(np.float32)


class ContactFixtures:
    """Four independent real kinematic rigid meshes outside the task inventory."""

    def __init__(self, config=None, num_envs=1):
        self.config = config or ContactDemoConfig()
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("Contact fixture num_envs must be a positive integer")
        self.num_envs = num_envs
        self.objects = [{} for _ in range(num_envs)]
        self.paths = [[] for _ in range(num_envs)]
        self.spawned = False

    def spawn(self, stage):
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        if self.spawned:
            raise RuntimeError("Contact fixtures must be authored only once before PhysX initialization")
        vertices, faces, self.geometry = load_contact_geometry(self.config)
        self.support_z_m = float(-vertices[:, 2].min())
        for env_id in range(self.num_envs):
            for index, (side, finger) in enumerate(PADS):
                label = side + "_" + finger
                path = f"/World/envs/env_{env_id}/contact_calibration/{label}"
                if stage.GetPrimAtPath(path).IsValid():
                    raise ValueError(f"Contact fixture prim already exists: {path}")
                root = UsdGeom.Xform.Define(stage, path)
                root.AddTranslateOp().Set(Gf.Vec3d(10 + env_id, index, 1))
                body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
                body.CreateKinematicEnabledAttr(True)
                UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.02)
                PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateDisableGravityAttr(True)
                mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
                mesh.CreatePointsAttr(vertices.tolist())
                mesh.CreateFaceVertexCountsAttr([3] * len(faces))
                mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateOrientationAttr(self.geometry.get("mesh_orientation", "rightHanded"))
                mesh.CreateDisplayColorAttr([Gf.Vec3f(0.24, 0.3, 0.36)])
                collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                collision.CreateCollisionEnabledAttr(True)
                surface = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
                surface.CreateContactOffsetAttr(0.0002)
                surface.CreateRestOffsetAttr(0.0)
                self.objects[env_id][label] = SimpleNamespace(prim_path=path)
                self.paths[env_id].append(path)
        self.spawned = True
        return self.objects

    def configuration(self):
        if not self.spawned:
            raise RuntimeError("Contact fixtures have not been authored")
        return {
            "enabled": True,
            "purpose": "prescribed kinematic contact calibration, not policy grasping",
            "config": asdict(self.config),
            "phases": list(PHASES),
            "duration_s": self.config.duration_s,
            "geometry": self.geometry,
            "prim_paths_by_env": self.paths,
            "physics_actuation": "native PhysX set_kinematic_targets",
            "force_semantics": "official TacSL SDF penalty estimate, not measured load-cell force",
            "required_physics_pipeline": "CUDA GPU dynamics with real CPU-native host readback",
            "host_readback_reason": "Isaac Sim 6.1 GPU rigid views do not implement kinematic targets",
            "required_contact_query_backend": "warp_triangle_mesh_signed_distance",
        }


def contact_demo_task_class(task_class, config=None):
    """Extend the original task only with pre-cook calibration fixture spawning."""
    fixture_config = config or ContactDemoConfig()

    class ContactCalibrationTask(task_class):
        def __init__(self, config, *args, **kwargs):
            if config.task_env.get("Garment"):
                raise NotImplementedError(
                    "Kinematic contact calibration host readback cannot share the surface FEM pipeline"
                )
            config.sim["particle_readback"] = True
            config.sim["host_readback_reason"] = "kinematic_contact_calibration"
            super().__init__(config, *args, **kwargs)

        def setup_physics(self, sim):
            super().setup_physics(sim)
            # BaseEnv has already installed the shared native CPU-to-CUDA
            # bridge. Set calibration provenance without installing it twice.
            configure_contact_physics(sim.sim, install_bridge=False)

        def _setup_contact_fixtures(self, sim):
            super()._setup_contact_fixtures(sim)
            self.contact_fixtures = ContactFixtures(fixture_config, self.num_envs)
            self.scene_manager._tactile_calibration_objects = self.contact_fixtures.spawn(sim.sim.stage)

        def close(self):
            context = self.sim.sim if self.sim is not None else None
            try:
                super().close()
            finally:
                if context is not None:
                    close_contact_physics(context)

    return ContactCalibrationTask


class ContactMotionDemo:
    """Command real native targets before stepping; record measured native state."""

    def __init__(self, fixtures, tactile_manager, sim_context, env_id=0):
        import torch
        import warp as wp

        from utils.particle_readback_bridge import ReadbackTensorView

        if not fixtures.spawned or not tactile_manager.enabled:
            raise RuntimeError("Contact demo requires preauthored fixtures and enabled mounted TacSL sensors")
        if not 0 <= env_id < fixtures.num_envs:
            raise ValueError("Contact demo environment is out of range")
        self.fixtures, self.manager, self.sim, self.env_id = fixtures, tactile_manager, sim_context, env_id
        self.config = fixtures.config
        self.wp, self.torch = wp, torch
        self.views = []
        physics = sim_context.physics_manager.get_physics_sim_view()
        if not isinstance(physics, ReadbackTensorView):
            raise RuntimeError(
                "Isaac Sim 6.1 kinematic contact requires native host readback; "
                "configure_contact_physics must run before first physics initialization"
            )
        for path in fixtures.paths[env_id]:
            view = physics.create_rigid_body_view([path])
            if view.count != 1:
                raise RuntimeError(f"Native kinematic fixture view unavailable: {path}")
            self.views.append(view)
        device = wp.to_torch(self.views[0].get_transforms()).device
        self.indices = wp.array([0], dtype=wp.int32, device=str(device))
        self.device = device
        self.last_command = None
        self.history = []
        self.reset()

    def configuration(self):
        return {
            **self.fixtures.configuration(),
            "readback_suppressed": bool(self.sim.get_setting("/physics/suppressReadback")),
            "contact_calibration_readback": bool(
                self.sim.get_setting("/isaaclab/robodojo/contact_calibration_readback")
            ),
            "measured_buffer_device": str(self.device),
            "native_buffer_devices": [str(view._native.get_transforms().device) for view in self.views],
            "contact_query_backends": {
                side + "." + finger: [
                    sensor.contact_query_backend for sensor in self.manager.sensors[side][finger][self.env_id]
                ]
                for side, finger in PADS
            },
        }

    def _poses(self, command):
        poses = []
        for side, finger in PADS:
            sensor = self.manager.sensors[side][finger][self.env_id][0]
            sensor._update_force_field(slice(None))
            points = sensor.data.tactile_points_pos_w[0].detach().cpu().numpy()
            transform = self.wp.to_torch(sensor._elastomer_body_view.get_transforms())[0].detach().cpu().numpy()
            poses.append(pose_at_pad(points.mean(axis=0), transform[3:], self.fixtures.support_z_m, command))
        return poses

    def _native_array(self, pose):
        return self.wp.from_torch(
            self.torch.as_tensor(pose[None], device=self.device, dtype=self.torch.float32).contiguous()
        )

    def reset(self):
        """Place unloaded probes after task reset, before the baseline capture."""
        command = phase_command(self.config, 0.0)
        for view, pose in zip(self.views, self._poses(command), strict=True):
            view.set_transforms(self._native_array(pose), self.indices)
            view.set_kinematic_targets(self._native_array(pose), self.indices)
        self.last_command = None
        self.history.clear()

    def park(self):
        """Explicitly unload/park before another TaskEnv reset and calibration.

        This is a reset operation, distinct from the continuous runtime path.
        Keep the views alive until reset finishes, then call reset() to derive
        unloaded poses from the new actual mounted-pad poses.
        """
        for index, view in enumerate(self.views):
            pose = np.asarray([10 + self.env_id, index, 1, 0, 0, 0, 1], dtype=np.float32)
            view.set_transforms(self._native_array(pose), self.indices)
            view.set_kinematic_targets(self._native_array(pose), self.indices)
        self.last_command = None
        self.history.clear()

    def apply(self, elapsed_time_s, physics_frame):
        if not isinstance(physics_frame, int) or physics_frame < 0:
            raise ValueError("Contact command physics_frame must be a nonnegative integer")
        command = phase_command(self.config, elapsed_time_s)
        poses = self._poses(command)
        for view, pose in zip(self.views, poses, strict=True):
            view.set_kinematic_targets(self._native_array(pose), self.indices)
        self.last_command = {
            **command,
            "issued_before_physics_frame": physics_frame,
            "commanded_probe_poses_w_xyzw": np.stack(poses),
        }
        return self.last_command

    def sample(self):
        if self.last_command is None:
            raise RuntimeError("Contact sample requires a command followed by a real physics step")
        snapshot = simulation_snapshot(self.sim)
        if snapshot[1] <= self.last_command["issued_before_physics_frame"]:
            raise RuntimeError("Contact command has not advanced through a real physics step")
        poses = np.concatenate(
            [self.wp.to_torch(view.get_transforms()).detach().cpu().numpy().copy() for view in self.views]
        )
        velocities = np.concatenate(
            [self.wp.to_torch(view.get_velocities()).detach().cpu().numpy().copy() for view in self.views]
        )
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Native contact fixture pose/velocity is nonfinite")
        if simulation_snapshot(self.sim) != snapshot:
            raise RuntimeError("Physics advanced while reading contact fixture state")
        record = {
            "last_command": {
                **self.last_command,
                "commanded_probe_poses_w_xyzw": self.last_command["commanded_probe_poses_w_xyzw"].copy(),
            },
            "measured_probe_poses_w_xyzw": poses,
            "measured_probe_velocities_w": velocities,
            "probe_paths": self.fixtures.paths[self.env_id],
            "kinematic_calibration": True,
            "physics_frame": snapshot[1],
            "timestamp_ns": round(snapshot[0] * 1e9),
        }
        self.history.append(record)
        return record

    def validate(self):
        if len(self.history) < 2:
            raise RuntimeError("Contact sequence requires multiple measured physics samples")
        phases = {record["last_command"]["phase"] for record in self.history}
        missing = set(PHASES) - phases
        if missing:
            raise RuntimeError(f"Contact sequence did not record all phases: {sorted(missing)}")
        actual = np.stack([record["measured_probe_poses_w_xyzw"] for record in self.history])
        target = np.stack([record["last_command"]["commanded_probe_poses_w_xyzw"] for record in self.history])
        error = np.linalg.norm(actual[..., :3] - target[..., :3], axis=-1)
        if float(error.max()) > 0.0002:
            raise RuntimeError("Native kinematic probe did not follow the prescribed contact path within 0.2mm")
        actual_q, target_q = actual[..., 3:].astype(np.float64), target[..., 3:].astype(np.float64)
        actual_norm = np.linalg.norm(actual_q, axis=-1, keepdims=True)
        target_norm = np.linalg.norm(target_q, axis=-1, keepdims=True)
        if np.any(actual_norm <= 1e-9) or np.any(target_norm <= 1e-9):
            raise RuntimeError("Native contact probe quaternion has zero norm")
        norm_error = float(np.max(np.abs(actual_norm - 1)))
        if norm_error > 0.0001:
            raise RuntimeError("Native contact probe quaternion is not unit length")
        actual_q, target_q = actual_q / actual_norm, target_q / target_norm
        dots = np.abs(np.sum(actual_q * target_q, axis=-1))
        angle_error = 2 * np.arccos(np.clip(dots, 0, 1))
        if float(angle_error.max()) > 0.003:
            raise RuntimeError("Native kinematic probe did not follow its prescribed twist within 0.003rad")
        rotation_from_start = 2 * np.arccos(np.clip(np.abs(np.sum(actual_q * actual_q[0], axis=-1)), 0, 1))
        if np.any(np.max(np.ptp(actual[..., :3], axis=0), axis=1) < self.config.slide_m):
            raise RuntimeError("One mounted pad's native contact probe did not measurably move")
        velocities = np.stack([record["measured_probe_velocities_w"] for record in self.history])
        if np.any(np.max(np.linalg.norm(velocities[..., :3], axis=-1), axis=0) <= 1e-5):
            raise RuntimeError("Native kinematic contact linear velocities remained zero")
        if np.any(np.max(np.linalg.norm(velocities[..., 3:], axis=-1), axis=0) <= 1e-3):
            raise RuntimeError("Native contact probes did not physically twist")
        return {
            "configuration": self.configuration(),
            "sample_count": len(self.history),
            "phases_observed": sorted(phases),
            "native_pose_max_error_m": float(error.max()),
            "native_orientation_max_error_rad": float(angle_error.max()),
            "native_quaternion_norm_max_error": norm_error,
            "native_rotation_from_start_max_rad": np.max(rotation_from_start, axis=0).tolist(),
            "native_linear_speed_max_m_s": np.max(np.linalg.norm(velocities[..., :3], axis=-1), axis=0).tolist(),
            "native_angular_speed_max_rad_s": np.max(np.linalg.norm(velocities[..., 3:], axis=-1), axis=0).tolist(),
            "tactile_response_validation": "separate measured sensor contact metrics required",
        }

    def close(self):
        """Release Python native views before the owning TaskEnv tears down PhysX."""
        self.views.clear()
        self.history.clear()
