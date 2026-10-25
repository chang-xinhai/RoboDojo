"""Mount official TacSL elastomers and collect synchronous visuotactile data.

The force field is the official TacSL penalty/SDF model, not a load-cell reading.
Depth is camera distance in metres; penetration_depth is SDF indentation in metres.
All configured rigid task objects participate through independent official SDF views.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

from .contact_paths import contact_path_for_env


class TactileManager:
    def __init__(self, num_envs, config, robot_manager, device):
        self.num_envs = num_envs
        self.config = config or {}
        labels = self.config.get("contact_object_labels")
        if labels is not None and (
            not isinstance(labels, Sequence)
            or isinstance(labels, str)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("TacSL contact_object_labels must be a nonempty list of unique task labels")
        if labels is not None and self.config.get("contact_object_paths"):
            raise ValueError("Use task contact_object_labels or explicit contact_object_paths, not both")
        if self.config.get("contact_conditioned_observations") and labels is None:
            raise ValueError("Contact-conditioned tactile observations require explicit native task contact labels")
        # Align active gel grid with X5 jaw contact-face bbox center: X=63.5mm,
        # Z=-1.196808mm. Gel grid center is Z=67.76mm in its actor frame.
        mount = self.config.get("mount_position_link", (0.063500002, 0.024, -0.068956808))
        if not isinstance(mount, Sequence) or isinstance(mount, str) or len(mount) != 3:
            raise ValueError(
                "TacSL mount_position_link must contain [forward_x, absolute_inward_y, origin_z] in metres"
            )
        self.mount_position_link = tuple(float(value) for value in mount)
        if not all(math.isfinite(value) for value in self.mount_position_link) or self.mount_position_link[1] < 0:
            raise ValueError("TacSL mount_position_link must be finite with a nonnegative absolute inward Y offset")
        # Lazy import avoids bringing Warp/Kit into normal disabled-manager construction.
        self.contact_query_backend = self.config.get("contact_query_backend", "auto")
        if self.contact_query_backend not in ("auto", "physx_native_sdf", "warp_triangle_mesh"):
            raise ValueError(f"Unsupported TacSL contact_query_backend: {self.contact_query_backend!r}")
        self.enabled = bool(self.config.get("enabled", False))
        self.enable_camera_tactile = bool(self.config.get("enable_camera_tactile", True))
        self.robot_manager = robot_manager
        self.device = str(device)
        self.sim = None
        self.scene_manager = None
        self.sensors = {}
        self.contact_paths = []
        self.contact_paths_by_env = [[] for _ in range(num_envs)]
        self._calibrated = False
        self.force_field_available = False
        self.asset_root = Path(__file__).resolve().parents[2] / "Assets" / "TacSL"

    def initialize(self, sim, scene_manager):
        """Call after scene objects spawn and before the first PhysX reset."""
        if not self.enabled:
            return
        if not self.device.startswith("cuda"):
            raise ValueError("TacSL SDF force fields require a CUDA PhysX device; set sim.device=cuda:0.")
        import omni.usd

        self.sim = sim
        self.scene_manager = scene_manager
        # Publish optical requirements before PhysX reset, including tasks
        # that do not configure a head/wrist camera manager.
        if self.enable_camera_tactile:
            sim.sim.set_setting("/isaaclab/render/rtx_sensors", True)
        self.stage = omni.usd.get_context().get_stage()
        asset = self.asset_root / "gelsight_r15_finger" / "gelsight_r15_finger.usd"
        if not asset.exists():
            raise FileNotFoundError(
                f"Missing official TacSL assets: {asset}. Run reproduction/isaac61/fetch_tacsl_assets.py."
            )
        paths = self.config.get("contact_object_paths")
        if isinstance(paths, str):
            raise TypeError("TacSL contact_object_paths must be a list of paths, not one string")
        for env_index in range(self.num_envs):
            if paths:
                self.contact_paths_by_env[env_index] = self._contact_meshes(
                    [contact_path_for_env(p, env_index) for p in paths]
                )
                if not self.contact_paths_by_env[env_index]:
                    raise ValueError(f"Configured TacSL contacts in env_{env_index} have no rigid mesh colliders")
            else:
                self.contact_paths_by_env[env_index] = self._discover_contact_bodies(scene_manager, env_index)
        self.contact_paths = [path for paths in self.contact_paths_by_env for path in paths]
        self.force_field_available = any(self.contact_paths_by_env)
        self.synchronize_contact_objects()
        self._prepare_contact_collisions()
        targets = [r for r in self.robot_manager.robot_list if getattr(r, "type", "target") == "target"]
        if len(targets) != 2:
            raise ValueError(f"Bimanual tactile collection requires two target arms, found {len(targets)}.")
        for side, robot in zip(("left", "right"), targets, strict=True):
            robot_index = self.robot_manager.robot_list.index(robot)
            if robot.robot_name != "x5":
                raise ValueError(
                    f"No validated TacSL mount for {robot.robot_name}; configure a mount before enabling tactile."
                )
            self.sensors[side] = {}
            for finger, link, sign in (("finger_0", "link7", -1), ("finger_1", "link8", 1)):
                self.sensors[side][finger] = {}
                for env_index in range(self.num_envs):
                    robot_path = f"/World/envs/env_{env_index}/robot{robot_index}"
                    if self.stage.GetPrimAtPath(robot_path + "/" + link).IsValid():
                        self._mount(robot_path, finger, link, sign, asset)
                    else:
                        raise RuntimeError(f"Missing cloned TacSL robot gripper: {robot_path}/{link}")
                    self.sensors[side][finger][env_index] = self._create_sensors(
                        robot_path + f"/tacsl_{finger}", self.contact_paths_by_env[env_index]
                    )

    def _discover_contact_bodies(self, scene_manager, env_index=0):
        """Each actual mesh collider gets an independent SDF/body view."""

        objects = dict(scene_manager._rigid_and_dynamic_objects[env_index])
        objects.update(scene_manager._articulation_objects[env_index])
        calibration_objects = getattr(scene_manager, "_tactile_calibration_objects", None)
        labels = self.config.get("contact_object_labels")
        if labels is not None:
            if calibration_objects is not None and calibration_objects[env_index]:
                raise ValueError("Native task contact labels forbid tactile calibration/probe objects")
            paths = []
            for label in labels:
                instance_name = scene_manager.layout_manager.get_instance_name(env_index, label=label)
                if instance_name is None or instance_name not in objects:
                    raise ValueError(
                        f"Missing native TacSL contact label {label} in env_{env_index}; "
                        f"resolved instance={instance_name!r}"
                    )
                matches = self._contact_meshes([objects[instance_name].prim_path])
                if not matches:
                    raise ValueError(
                        f"Native TacSL contact label {label} in env_{env_index} has no actual rigid mesh colliders"
                    )
                paths.extend(matches)
            return paths
        roots = [obj.prim_path for obj in objects.values()]
        if calibration_objects is not None:
            roots.extend(obj.prim_path for obj in calibration_objects[env_index].values())
        return self._contact_meshes(roots)

    def _contact_meshes(self, roots):
        from pxr import Usd, UsdGeom, UsdPhysics

        paths = []
        for path in roots:
            root = self.stage.GetPrimAtPath(path)
            if not root.IsValid():
                raise ValueError(f"No actual TacSL contact prim at {path}")
            for mesh in Usd.PrimRange(root):
                if not mesh.IsA(UsdGeom.Mesh) or not mesh.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get() is False:
                    continue
                parent = mesh
                while parent.IsValid() and str(parent.GetPath()) != "/":
                    if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                        paths.append(str(mesh.GetPath()))
                        break
                    parent = parent.GetParent()
        return list(dict.fromkeys(paths))

    def synchronize_contact_objects(self):
        """Verify reset/capture topology still agrees with prebuilt PhysX SDF views.

        CUDA SceneManager retains rigid/dynamic/articulated assets across reset;
        only poses change. A new/removed rigid body requires a fresh PhysX scene
        build because adding an SDF schema after cooking is not a valid force
        sensor initialization. Fail explicitly rather than silently omit touch.
        """
        if not self.enabled or self.scene_manager is None:
            return
        if not self.config.get("contact_object_paths"):
            for env_index in range(self.num_envs):
                current = self._discover_contact_bodies(self.scene_manager, env_index)
                expected = self.contact_paths_by_env[env_index]
                if set(current) != set(expected):
                    raise RuntimeError(
                        "TacSL rigid contact topology changed after PhysX initialization; recreate the TaskEnv "
                        f"before capture. env_{env_index}: added={sorted(set(current) - set(expected))}, "
                        f"removed={sorted(set(expected) - set(current))}."
                    )

    def _prepare_sdf(self, expression):
        """Validate an already-authored SDF collider; never recook task physics."""
        from pxr import Usd, UsdGeom, UsdPhysics

        root = self.stage.GetPrimAtPath(expression)
        if not root.IsValid():
            raise ValueError(f"No TacSL contact object at {expression}")
        meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
        if not meshes or any(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() != "sdf" for p in meshes):
            raise ValueError("Native TacSL SDF needs authored SDF cooking; use auto to preserve other colliders")

    def _prepare_contact_collisions(self):
        # Auto queries existing SDF meshes natively and other original surfaces
        # through the triangle backend. Neither mode authors collision schemas.
        if self.contact_query_backend == "physx_native_sdf":
            for path in self.contact_paths:
                self._prepare_sdf(path)

    def _mount(self, robot_path, finger, link_name, sign, asset):
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        link_path = robot_path + "/" + link_name
        link = self.stage.GetPrimAtPath(link_path)
        if not link.IsValid() or not link.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Expected rigid gripper link {link_path}")
        sensor_path = robot_path + "/tacsl_" + finger
        root = UsdGeom.Xform.Define(self.stage, sensor_path)
        root.GetPrim().GetReferences().AddReference(str(asset))
        for relative in ("elastomer/visuals", "elastomer/collisions", "elastomer_tip/cam"):
            if not self.stage.GetPrimAtPath(sensor_path + "/" + relative).IsValid():
                raise RuntimeError(f"Incomplete official TacSL sensor asset: {sensor_path}/{relative}")
        # R1.5 gel faces -Y in its native frame. ARX link7 faces -Y,
        # link8 faces +Y. The gel is centred over the opposing jaw face.
        position = Gf.Vec3d(self._mount_translation(sign))
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 180 if sign > 0 else 0)
        mount_matrix = Gf.Matrix4d(1).SetRotate(rotation)
        mount_matrix.SetTranslateOnly(position)
        cache = UsdGeom.XformCache()
        robot_world = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(robot_path))
        link_in_robot = cache.GetLocalToWorldTransform(link) * robot_world.GetInverse()
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(mount_matrix * link_in_robot)
        # The sensor joins the existing robot articulation. Its original fixed
        # world joint must be disabled before PhysX builds the articulation.
        self.stage.GetPrimAtPath(sensor_path + "/root_joint").SetActive(False)
        for suffix in ("/gelsight_finger/visuals", "/gelsight_finger/collisions"):
            self.stage.GetPrimAtPath(sensor_path + suffix).SetActive(False)
        for prim in Usd.PrimRange(root.GetPrim()):
            if prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI(prim).CreateMassAttr(0.005)
            # Distinct names prevent collisions between the four added bodies.
            if prim.HasAttribute("isaac:nameOverride"):
                prim.GetAttribute("isaac:nameOverride").Set("tacsl_" + finger + "_" + prim.GetName())
        joint = UsdPhysics.FixedJoint.Define(self.stage, sensor_path + "/mount_joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(link_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(sensor_path + "/gelsight_finger")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(position))
        joint.CreateLocalRot0Attr(Gf.Quatf(rotation.GetQuat()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateCollisionEnabledAttr(False)
        # The gel overlaps its mounting jaw's backing surface. They are two
        # links apart in the fixed-joint tree, so disabling joint collisions
        # alone does not filter this pair when robot self-collision is enabled.
        gel = self.stage.GetPrimAtPath(sensor_path + "/elastomer")
        UsdPhysics.FilteredPairsAPI.Apply(gel).CreateFilteredPairsRel().AddTarget(Sdf.Path(link_path))
        # Allow a real compliant contact penetration, as required by TacSL.
        material = UsdShade.Material.Define(self.stage, sensor_path + "/compliant_gel")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(0.8)
        compliant = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        compliant.CreateCompliantContactStiffnessAttr(float(self.config.get("compliance_stiffness", 1500.0)))
        compliant.CreateCompliantContactDampingAttr(float(self.config.get("compliance_damping", 5.0)))
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(sensor_path + "/elastomer/collisions")):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(sensor_path + "/elastomer_tip/cam"))
        # The X5 mounting jaw lies inside the native internal optical volume.
        # Clip its backing (16.53 mm from camera) while retaining the front gel
        # and real contact objects. Intrinsics and calibrated far gel plane stay
        # unchanged; this near plane is specific to the validated X5 mount.
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.0173, 0.0186))

    def _mount_translation(self, sign):
        """Actual fixed-joint/asset translation, applying the jaw's inward sign."""
        if sign not in (-1, 1):
            raise ValueError("TacSL mount requires a finger sign of -1 or +1")
        x, inward_y, z = self.mount_position_link
        return x, sign * inward_y, z

    def _create_sensors(self, base, contact_paths):
        from isaaclab.sensors import CameraCfg
        import isaaclab.sim as sim_utils
        from isaaclab_assets.sensors import GELSIGHT_R15_CFG
        from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        from .tacsl_sensor import RoboDojoTacSLSensor

        render_cfg = GELSIGHT_R15_CFG.copy()
        render_cfg.base_data_path = str(self.asset_root)
        result = []
        for index, path in enumerate(contact_paths or [None]):
            camera = (
                CameraCfg(
                    prim_path=base + "/elastomer_tip/cam",
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=2.0342858,
                        horizontal_aperture=2.3977,
                        vertical_aperture=3.19693,
                        clipping_range=(0.0173, 0.0186),
                    ),
                    width=render_cfg.image_width,
                    height=render_cfg.image_height,
                    data_types=["distance_to_image_plane"],
                    update_period=0,
                    renderer_cfg=IsaacRtxRendererCfg(depth_clipping_behavior="max", enable_scene_partitioning=False),
                )
                if index == 0 and self.enable_camera_tactile
                else None
            )
            cfg = VisuoTactileSensorCfg(
                prim_path=base + f"/elastomer/sensor_{index}",
                update_period=0,
                enable_camera_tactile=index == 0 and self.enable_camera_tactile,
                enable_force_field=path is not None,
                camera_cfg=camera,
                render_cfg=render_cfg,
                tactile_array_size=tuple(self.config.get("array_size", (12, 16))),
                tactile_margin=float(self.config.get("margin", 0.003)),
                contact_object_prim_path_expr=path,
                normal_contact_stiffness=float(self.config.get("normal_stiffness", 1500.0)),
                tangential_stiffness=float(self.config.get("tangential_stiffness", 0.1)),
                friction_coefficient=float(self.config.get("friction", 0.8)),
            )
            result.append(RoboDojoTacSLSensor(cfg, contact_query_backend=self.contact_query_backend))
        return result

    def _sensor_groups(self):
        for fingers in self.sensors.values():
            for environments in fingers.values():
                yield from environments.items()

    def reset(self, env_ids=None):
        """Clear histories; preserve the initial no-contact depth calibration."""
        selected = set(range(self.num_envs)) if env_ids is None else {int(i) for i in env_ids}
        for env_index, sensors in self._sensor_groups():
            if env_index in selected:
                for sensor in sensors:
                    sensor.reset()

    def calibrate(self):
        """Acquire the initial no-contact RTX depth with open empty jaws."""
        if not self.enabled or self._calibrated or not self.enable_camera_tactile:
            return
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        import torch

        from utils.isaaclab_compat import as_torch

        self._wait_for_optical_products()
        for _, sensors in self._sensor_groups():
            for _attempt in range(8):
                # A just-created RTX product can return an empty all-zero frame.
                # Warm actual rendering while physics stays frozen; never accept
                # a zero-filled buffer as the sensor's optical calibration.
                ensure_isaac_rtx_render_update(force=True)
                initial = sensors[0].get_initial_render()
                depth = as_torch(initial["distance_to_image_plane"])
                if torch.isfinite(depth).all() and bool((depth > 0).all()):
                    break
                # RenderContext's generation invalidates the per-frame RTX
                # pump cache; force=True alone only bypasses headless gating.
                self.sim.sim.render()
            else:
                raise RuntimeError("Invalid/empty TacSL baseline; verify RTX clipping and gel visibility.")
            for sensor in sensors:
                if sensor.cfg.enable_force_field:
                    sensor._update_force_field(slice(None))
                    if bool((sensor.data.penetration_depth > 1e-5).any()):
                        raise RuntimeError("TacSL baseline calibration requires open jaws without object contact.")
        self._calibrated = True

    def update(self, dt=0.0, force_recompute=True):
        if not self.enabled:
            return
        if self.enable_camera_tactile and self._calibrated:
            self._wait_for_optical_products()
        self.calibrate()
        for _, sensors in self._sensor_groups():
            for sensor in sensors:
                sensor.update(dt, force_recompute=force_recompute)

    def _wait_for_optical_products(self):
        """Fence each actual gel depth product before reading or calibration."""
        from env.camera_manager.capture.render_sync import RenderProductDeliveryFence, wait_for_camera_data
        from utils.isaaclab_compat import simulation_snapshot

        primaries = [sensors[0] for _, sensors in self._sensor_groups()]
        if getattr(self, "_optical_delivery_fence", None) is None:
            self._optical_delivery_fence = RenderProductDeliveryFence(
                render_products=[sensor._camera_sensor._render_data.render_product for sensor in primaries]
            )
        def current_depth():
            # get_initial_render() replaces the no-contact calibration. Read
            # current delivered products here without recalibrating on contact.
            samples = []
            for sensor in primaries:
                camera = sensor._camera_sensor
                camera.update(dt=0.0, force_recompute=True)
                samples.append({"distance_to_image_plane": [{"data": camera.data.output["distance_to_image_plane"]}]})
            return samples

        wait_for_camera_data(
            self.sim.sim,
            current_depth,
            physics_state=lambda: simulation_snapshot(self.sim.sim),
            max_passes=32,
            delivery_clock=self._optical_delivery_fence.snapshot,
        )

    def get_observations(self):
        """Collect each actual environment independently and stack image batches."""
        if not self.enabled:
            return {}
        import torch
        import warp as wp

        from utils.isaaclab_compat import as_torch, simulation_snapshot

        from .contact_observation import condition_contact_observation
        from .force_projection import project_force_to_pad

        self.synchronize_contact_objects()
        context = self.sim.sim
        time_s, frame = simulation_snapshot(context)
        self.update()
        timestamp_ns = round(time_s * 1e9)
        output = {}
        for side, fingers in self.sensors.items():
            output[side] = {}
            for finger, environments in fingers.items():
                samples = []
                for env_index in range(self.num_envs):
                    sensors = environments[env_index]
                    primary = sensors[0].data
                    baseline_depth = as_torch(sensors[0]._nominal_tactile["distance_to_image_plane"])[0]
                    available = bool(self.contact_paths_by_env[env_index])
                    normal = shear = penetration = points = None
                    body_force = pad_normal = pad_shear = taxel_quat = gel_pose = None
                    if available:
                        normal = primary.tactile_normal_force[0].clone()
                        shear = primary.tactile_shear_force[0].clone()
                        penetration = primary.penetration_depth[0].clone()
                        points = primary.tactile_points_pos_w[0].clone()
                        for sensor in sensors[1:]:
                            normal += sensor.data.tactile_normal_force[0]
                            shear += sensor.data.tactile_shear_force[0]
                            penetration = penetration.maximum(sensor.data.penetration_depth[0])
                        taxel_quat = primary.tactile_points_quat_w[0].clone()
                        gel_pose = wp.to_torch(sensors[0]._elastomer_body_view.get_transforms())[0].clone()
                        body_force, pad_normal, pad_shear = project_force_to_pad(
                            normal, shear, taxel_quat, gel_pose[3:]
                        )
                    optical = {
                        "rgb": primary.tactile_rgb_image[0].to(torch.uint8).clone(),
                        "depth": primary.tactile_depth_image[0].clone(),
                        "optical_depth_delta": (baseline_depth - primary.tactile_depth_image[0]).clone(),
                    }
                    conditioned = bool(self.config.get("contact_conditioned_observations", False))
                    if conditioned:
                        optical = condition_contact_observation(
                            optical["rgb"],
                            optical["depth"],
                            optical["optical_depth_delta"],
                            penetration,
                            normal,
                            shear,
                        )
                    samples.append(
                        {
                            **optical,
                            "normal_force": normal,
                            "shear_force": shear,
                            "penetration_depth": penetration,
                            "points_pos_w": points,
                            "force_body_xyz": body_force,
                            "pad_normal_force": pad_normal,
                            "pad_shear_force_xz": pad_shear,
                            "taxel_orientation_w_xyzw": taxel_quat,
                            "gel_pose_w_xyzw": gel_pose,
                            "provenance": {
                                "backend": "isaaclab_tacsl",
                                "optical_freshness": "native render-product completion fence; frozen physics",
                                "native_render_product_completions": dict(zip(
                                    self._optical_delivery_fence.paths,
                                    self._optical_delivery_fence.snapshot(), strict=True
                                )) if self.enable_camera_tactile else None,
                                "model": "official_TacSL_penalty_and_Taxim",
                                "sensor": "GelSight_R1.5",
                                "contact_query_backend": sensors[0].contact_query_backend if available else None,
                                "contact_query_metadata": [sensor.contact_query_metadata for sensor in sensors]
                                if available
                                else [],
                                "force_contact_scope": (
                                    "selected_native_task_labels"
                                    if self.config.get("contact_object_labels")
                                    else "registered_rigid_mesh_colliders"
                                )
                                if available
                                else None,
                                "contact_conditioned_observations": conditioned,
                                "contact_active_source": "any actual registered native task taxel SDF penetration > 0"
                                if conditioned
                                else None,
                                "force_field_available": available,
                                "force_unavailable_reason": None
                                if available
                                else "No supported rigid mesh SDF contact bodies",
                            },
                            "metadata": {
                                "depth_units": "m",
                                "depth_semantics": "distance_to_image_plane",
                                "penetration_units": "m",
                                "force_units": "N",
                                "force_frame": "official_tacsl_taxel_local",
                                "normal_force_axis": "official_taxel_z",
                                "shear_components": ["x", "y"],
                                "points_pos_w_units": "m",
                                "points_pos_w_frame": "world",
                                "pad_normal_force_frame": "actual_R1.5_elastomer_body",
                                "force_body_xyz_units": "N",
                                "pad_normal_force_units": "N",
                                "pad_shear_force_xz_units": "N",
                                "force_projection": "actual XYZW taxel-world rotation followed by inverse actual gel-body-world rotation",
                                "pad_normal_force_axis": [0.0, -1.0, 0.0],
                                "pad_normal_force_sign": "signed outward: dot(F_body,[0,-1,0]); compression is negative",
                                "pad_shear_force_xz_components": ["body_x", "body_z"],
                                "optical_depth_delta_units": "m",
                                "optical_depth_delta_semantics": "no_contact_camera_depth_minus_current_camera_depth",
                                "optical_depth_delta_limit": "RTX depth proxy of visible gel/contact geometry; not FEM gel deformation",
                                "compliant_contact_stiffness": float(self.config.get("compliance_stiffness", 1500.0)),
                                "compliant_contact_damping": float(self.config.get("compliance_damping", 5.0)),
                                "penalty_normal_stiffness": float(self.config.get("normal_stiffness", 1500.0)),
                                "env_id": env_index,
                                "array_size": list(self.config.get("array_size", (12, 16))),
                                "taxel_array_axes": ["sensor_local_x", "sensor_local_z"],
                                "taxel_flatten_order": "C: x outer, z inner",
                                "contact_object_paths": list(self.contact_paths_by_env[env_index]),
                                "contact_object_labels": list(self.config.get("contact_object_labels", [])),
                                "mount_position_link_config_m": list(self.mount_position_link),
                                "mount_translation_link_m": list(
                                    self._mount_translation(-1 if finger == "finger_0" else 1)
                                ),
                                "mount_rotation_link_xyzw": [0.0, 0.0, 0.0, 1.0]
                                if finger == "finger_0"
                                else [0.0, 0.0, 1.0, 0.0],
                                "mount_pose_source": "configured actual USD asset and fixed-joint transform relative to X5 jaw; no runtime pose override",
                                "optical_contact_conditioning": "rgb/depth/optical_depth_delta are actual optical samples when contact_active; otherwise exact zero sentinels; raw_optical_* preserve unmodified samples"
                                if conditioned
                                else None,
                                "raw_optical_depth_units": "m",
                                "raw_optical_depth_semantics": "unmodified distance_to_image_plane",
                            },
                        }
                    )
                result = {"timestamp_ns": timestamp_ns, "physics_frame": frame}
                fields = (
                    "rgb",
                    "depth",
                    "optical_depth_delta",
                    "normal_force",
                    "shear_force",
                    "penetration_depth",
                    "points_pos_w",
                    "force_body_xyz",
                    "pad_normal_force",
                    "pad_shear_force_xz",
                    "taxel_orientation_w_xyzw",
                    "gel_pose_w_xyzw",
                )
                if self.config.get("contact_conditioned_observations"):
                    fields += ("raw_optical_rgb", "raw_optical_depth", "raw_optical_depth_delta", "contact_active")
                for field in fields:
                    values = [sample[field] for sample in samples]
                    result[field] = (
                        torch.stack(values)
                        if all(v is not None for v in values)
                        else (None if all(v is None for v in values) else values)
                    )
                for field in ("provenance", "metadata"):
                    values = [sample[field] for sample in samples]
                    result[field] = values[0] if self.num_envs == 1 else values
                output[side][finger] = result
        if simulation_snapshot(context) != (time_s, frame):
            raise RuntimeError("Physics advanced during tactile acquisition.")
        return output

    def destroy(self):
        self.sensors.clear()
