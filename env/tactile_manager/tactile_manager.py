"""Mount official TacSL elastomers and collect synchronous visuotactile data.

The force field is the official TacSL penalty/SDF model, not a load-cell reading.
Depth is camera distance in metres; penetration_depth is SDF indentation in metres.
All configured rigid task objects participate through independent official SDF views.
"""

from __future__ import annotations

from pathlib import Path


class TactileManager:
    def __init__(self, num_envs, config, robot_manager, device):
        self.num_envs = num_envs
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.robot_manager = robot_manager
        self.device = str(device)
        self.sim = None
        self.sensors = {}
        self.contact_paths = []
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
        self.stage = omni.usd.get_context().get_stage()
        asset = self.asset_root / "gelsight_r15_finger" / "gelsight_r15_finger.usd"
        if not asset.exists():
            raise FileNotFoundError(
                f"Missing official TacSL assets: {asset}. Run reproduction/isaac61/fetch_tacsl_assets.py."
            )
        paths = self.config.get("contact_object_paths")
        if paths:
            self.contact_paths = [str(p).replace("{ENV_REGEX_NS}", "/World/envs/env_.+") for p in paths]
        else:
            self.contact_paths = self._discover_contact_bodies(scene_manager)
        self.force_field_available = bool(self.contact_paths)
        for path in self.contact_paths:
            self._prepare_sdf(path)
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
                for env_index in range(self.num_envs):
                    self._mount(f"/World/envs/env_{env_index}/robot{robot_index}", finger, link, sign, asset)
                base = f"/World/envs/env_.+/robot{robot_index}/tacsl_{finger}"
                self.sensors[side][finger] = self._create_sensors(base)

    def _discover_contact_bodies(self, scene_manager):
        """Each articulated rigid link gets an independent SDF view.

        TacSL chooses the first SDF collider under each body. Keeping expressions
        at body granularity prevents the sensor from silently choosing the first
        link of an articulated object. Non-rigid cloth has visual tactile only.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        objects = dict(scene_manager._rigid_and_dynamic_objects[0])
        objects.update(scene_manager._articulation_objects[0])
        paths = []
        for obj in objects.values():
            root = self.stage.GetPrimAtPath(obj.prim_path)
            bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for body in bodies:
                has_mesh = any(p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(body))
                if has_mesh:
                    paths.append(str(body.GetPath()).replace("/env_0/", "/env_.+/"))
        return list(dict.fromkeys(paths))

    def _prepare_sdf(self, expression):
        import isaaclab.sim as sim_utils
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        prims = sim_utils.find_matching_prims(expression)
        if not prims:
            raise ValueError(f"No TacSL contact object matches {expression}")
        for root in prims:
            meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
            if not meshes:
                raise ValueError(
                    f"TacSL requires a mesh collider under {root.GetPath()}; primitives must be converted to meshes."
                )
            for mesh in meshes:
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("sdf")
                sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh)
                sdf.CreateSdfResolutionAttr(int(self.config.get("sdf_resolution", 128)))

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
        position = Gf.Vec3d(0.035, sign * 0.024, -0.06776)
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
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.0001, 0.0186))

    def _create_sensors(self, base):
        from isaaclab.sensors import CameraCfg
        import isaaclab.sim as sim_utils
        from isaaclab_assets.sensors import GELSIGHT_R15_CFG
        from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor, VisuoTactileSensorCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        render_cfg = GELSIGHT_R15_CFG.copy()
        render_cfg.base_data_path = str(self.asset_root)
        result = []
        for index, path in enumerate(self.contact_paths or [None]):
            camera = (
                CameraCfg(
                    prim_path=base + "/elastomer_tip/cam",
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=2.0342858,
                        horizontal_aperture=2.3977,
                        vertical_aperture=3.19693,
                        clipping_range=(0.0001, 0.0186),
                    ),
                    width=render_cfg.image_width,
                    height=render_cfg.image_height,
                    data_types=["distance_to_image_plane"],
                    update_period=0,
                    renderer_cfg=IsaacRtxRendererCfg(depth_clipping_behavior="max"),
                )
                if index == 0
                else None
            )
            cfg = VisuoTactileSensorCfg(
                prim_path=base + f"/elastomer/sensor_{index}",
                update_period=0,
                enable_camera_tactile=index == 0,
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
            result.append(VisuoTactileSensor(cfg))
        return result

    def reset(self, env_ids=None):
        """Clear histories. Baseline is retained after the initial calibration."""
        for fingers in self.sensors.values():
            for sensors in fingers.values():
                for sensor in sensors:
                    sensor.reset(env_ids=env_ids)

    def calibrate(self):
        """Acquire the initial no-contact RTX depth; call with open empty jaws."""
        if not self.enabled or self._calibrated:
            return
        import torch

        from utils.isaaclab_compat import as_torch

        for fingers in self.sensors.values():
            for sensors in fingers.values():
                primary = sensors[0]
                initial = primary.get_initial_render()
                depth = as_torch(initial["distance_to_image_plane"])
                if not torch.isfinite(depth).all():
                    raise RuntimeError("Non-finite TacSL baseline; verify RTX clipping and gel visibility.")
                for sensor in sensors:
                    if not sensor.cfg.enable_force_field:
                        continue
                    sensor._update_force_field(slice(None))
                    if bool((sensor.data.penetration_depth > 1e-5).any()):
                        raise RuntimeError("TacSL baseline calibration requires open jaws without object contact.")
        self._calibrated = True

    def update(self, dt=0.0, force_recompute=True):
        if not self.enabled:
            return
        self.calibrate()
        for fingers in self.sensors.values():
            for sensors in fingers.values():
                for sensor in sensors:
                    sensor.update(dt, force_recompute=force_recompute)

    def get_observations(self):
        """Return copied tensor batches; update at the current frozen physics frame."""
        if not self.enabled:
            return {}
        self.update()
        context = self.sim.sim
        from utils.isaaclab_compat import simulation_snapshot

        time_s, frame = simulation_snapshot(context)
        timestamp_ns = round(time_s * 1e9)
        output = {}
        for side, fingers in self.sensors.items():
            output[side] = {}
            for finger, sensors in fingers.items():
                primary = sensors[0].data
                normal = shear = penetration = points = None
                if self.force_field_available:
                    normal = primary.tactile_normal_force.clone()
                    shear = primary.tactile_shear_force.clone()
                    penetration = primary.penetration_depth.clone()
                    points = primary.tactile_points_pos_w.clone()
                    for sensor in sensors[1:]:
                        normal += sensor.data.tactile_normal_force
                        shear += sensor.data.tactile_shear_force
                        penetration = penetration.maximum(sensor.data.penetration_depth)
                output[side][finger] = {
                    "rgb": primary.tactile_rgb_image.to(__import__("torch").uint8).clone(),
                    "depth": primary.tactile_depth_image.clone(),
                    "normal_force": normal,
                    "shear_force": shear,
                    "penetration_depth": penetration,
                    "points_pos_w": points,
                    "timestamp_ns": timestamp_ns,
                    "physics_frame": frame,
                    "provenance": {
                        "backend": "isaaclab_tacsl",
                        "model": "official_TacSL_penalty_SDF_and_Taxim",
                        "sensor": "GelSight_R1.5",
                        "force_field_available": self.force_field_available,
                        "force_unavailable_reason": None
                        if self.force_field_available
                        else "No supported rigid mesh SDF contact bodies",
                    },
                    "metadata": {
                        "depth_units": "m",
                        "depth_semantics": "distance_to_image_plane",
                        "penetration_units": "m",
                        "force_units": "N",
                        "array_size": list(self.config.get("array_size", (12, 16))),
                        "contact_object_paths": list(self.contact_paths),
                    },
                }
        if simulation_snapshot(context) != (time_s, frame):
            raise RuntimeError("Physics advanced during tactile acquisition.")
        return output

    def destroy(self):
        self.sensors.clear()
