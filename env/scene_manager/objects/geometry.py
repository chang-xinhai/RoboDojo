import random

from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
import numpy as np
from omegaconf import DictConfig
import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import torch


class GeometryObject(SingleGeometryPrim):
    """Static geometry object with collision support.

    Represents static objects (e.g., rooms, walls, fixed structures) with collision detection,
    configured entirely through the provided configuration.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        inst_config: DictConfig,
        env_origin: torch.Tensor,
        default_pos: tuple,
        default_ori: tuple,
        scale: tuple,
        primitive_type: str = None,
    ):
        """Initialize a static geometry object.

        Args:
            prim_path: USD prim path for the object (format: .../{config_root}/{category}/{instance})
            usd_path: Path to the USD asset file. Can be None for procedurally generated objects.
            config: Global configuration containing object properties
            config_root: Root node name for object settings in config, default is "objects"
            primitive_type: The name of the primitive shape (e.g., "Cube"), if applicable.
        """
        if usd_path:
            prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        else:
            prim = get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            error_message = (
                f"Failed to load USD from {usd_path} to {prim_path}"
                if usd_path
                else f"Failed to find an existing prim at path {prim_path}"
            )
            raise RuntimeError(error_message)
        self.stage = get_current_stage()
        self.env_origin = env_origin
        prim_path_parts = prim_path.split("/")
        self._prim_path = prim_path
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.primitive_type = primitive_type
        self.usd_path = usd_path
        self.usd_prim_path = prim_path
        self.instance_config = inst_config
        self.visual_config = self.instance_config.get("visual", {})
        self.physics_config = self.instance_config.get("physics", {})
        collision_val = self.physics_config.get("collision")
        collision = collision_val if collision_val is not None else False
        track_forces_val = self.physics_config.get("track_contact_forces")
        track_contact_forces = track_forces_val if track_forces_val is not None else False
        prepare_sensor_val = self.physics_config.get("prepare_contact_sensor")
        prepare_contact_sensor = prepare_sensor_val if prepare_sensor_val is not None else False
        disable_stab_val = self.physics_config.get("disable_stablization")
        disable_stablization = disable_stab_val if disable_stab_val is not None else True
        contact_filter_val = self.physics_config.get("contact_filter_prim_paths_expr")
        contact_filter_prim_paths_expr = contact_filter_val if contact_filter_val is not None else []
        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            raise ValueError(f"Could not extract env_id from prim path: {self._prim_path}")
        self.default_pos = default_pos
        self.default_ori = default_ori
        self.scale = scale
        self.physics_material = PhysicsMaterial(
            prim_path=self.usd_prim_path + "/physcis_material",
            static_friction=self.physics_config.get("static_friction", 0.0),
            dynamic_friction=self.physics_config.get("dynamic_friction", 0.0),
            restitution=self.physics_config.get("restitution", 0.5),
        )
        # Mesh cooking happens while SingleGeometryPrim enables collision.
        # Apply a task-requested approximation before that call; changing the
        # USD attribute afterwards leaves PhysX using the authored cooking mode.
        collision_approximation = self.physics_config.get("collision_approximation")
        if collision_approximation:
            self._set_collision_approximation_on_meshes(str(collision_approximation))
        # Respect an explicit collision=false even when an authored USD mesh
        # already carries PhysicsCollisionAPI=true. SingleGeometryPrim's
        # collision flag does not remove child mesh collision schemas, so set
        # every authored collision mesh before PhysX cooking.
        if collision is False:
            self._set_collision_enabled_on_meshes(False)
        super().__init__(
            prim_path=prim_path,
            name=self.instance_name,
            translation=self.default_pos,
            orientation=self.default_ori,
            scale=self.scale,
            visible=self.visual_config.get("visible", True),
            collision=collision,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensor=prepare_contact_sensor,
            disable_stablization=disable_stablization,
            contact_filter_prim_paths_expr=contact_filter_prim_paths_expr,
        )
        # Some task assets contain narrow holes or thin collision walls. The
        # default Isaac Sim cooking approximation can fill those holes and
        # eject a nearby dynamic body on the first physics step. Keep the
        # existing default for all tasks, but allow a task asset to request
        # the authored mesh approximation explicitly.
        self._remove_rigid_body_if_exists(prim_path)
        self._current_color = self.visual_config.get("color", None)
        if self._current_color is not None:
            self._apply_color_material(self._current_color)
        else:
            self.visual_usd_path = self.visual_config.get("visual_usd_path", None)
            if self.visual_usd_path:
                self._apply_visual_material(self.visual_usd_path)
            elif not usd_path:
                self._apply_default_material()

    def _remove_rigid_body_if_exists(self, prim_path):
        prim = get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            print(f"Warning: Cannot find prim at {prim_path} to remove rigid body.")
            return
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = UsdPhysics.RigidBodyAPI.Get(prim.GetStage(), prim.GetPath())
            rb_api.GetRigidBodyEnabledAttr().Set(False)
            attributes_to_remove = [
                "physics:kinematicEnabled",
                "physics:startsAsleep",
                "physics:linearVelocity",
                "physics:angularVelocity",
            ]
            for attr_name in attributes_to_remove:
                if prim.HasAttribute(attr_name):
                    prim.RemoveProperty(attr_name)

    def _set_collision_enabled_on_meshes(self, enabled: bool) -> None:
        """Set collision enabled on authored child meshes before cooking."""
        root = get_prim_at_path(self._prim_path)
        if not root or not root.IsValid():
            return
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(bool(enabled))

    def _set_collision_approximation_on_meshes(self, approximation: str) -> None:
        """Set the PhysX mesh cooking mode for every authored collision mesh."""
        root = get_prim_at_path(self._prim_path)
        if not root or not root.IsValid():
            return
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            api = UsdPhysics.MeshCollisionAPI(prim)
            api.GetApproximationAttr().Set(approximation)

    def _extract_env_id_from_prim_path(self):
        """get env_id from prim_path"""
        try:
            parts = self._prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def _apply_color_material(self, color_rgb):
        """Creates or updates a simple PreviewSurface material with the given color and binds it."""
        if self.color_material_path is None:
            self.color_material_path = find_unique_string_name(
                initial_name=self.usd_prim_path + "/Looks/color_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
        material_prim = get_prim_at_path(self.color_material_path)
        if not material_prim:
            material = PreviewSurface(prim_path=self.color_material_path, color=torch.tensor(color_rgb))
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.prim.GetPath(),
                material_path=self.color_material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )
        else:
            material = PreviewSurface(prim_path=self.color_material_path)
            try:
                material.set_color(np.array(color_rgb))
            except Exception as e:
                print(f"Error setting color for material {self.color_material_path}: {e}")

    def _apply_visual_material(self, material_path: str):
        self.visual_material_path = find_unique_string_name(
            self.usd_prim_path + "/visual_material", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        add_reference_to_stage(usd_path=material_path, prim_path=self.visual_material_path)
        visual_material_ref_prim = prims_utils.get_prim_at_path(self.visual_material_path)
        material_children = prims_utils.get_prim_children(visual_material_ref_prim)
        if not material_children:
            print(f"Warning: Material USD at {material_path} has no child prims to bind.")
            return
        self.material_prim = material_children[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)
        object_prim = self.prim
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=object_prim.GetPath(),
            material_path=self.material_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )
        children_prims = prims_utils.get_prim_children(object_prim)
        for prim in children_prims:
            if prim.GetTypeName() in ["Mesh", "GeomSubset"]:
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=prim.GetPath(),
                    material_path=self.material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

    def _apply_default_material(self, material_name: str = "default_material", prim_path: str = None) -> str:
        """Creates a default PreviewSurface material (no color) and binds it to the prim.

        This ensures the object always has a material bound, even if no color or material
        is specified in the configuration.

        Args:
            material_name: Name for the material (default: "default_material")
            prim_path: Prim path to bind the material to. If None, uses self.prim.GetPath()

        Returns:
            str: The path of the created material
        """
        if prim_path is None:
            prim_path = self.prim.GetPath()
        opaque_mtl_path = f"{self.usd_prim_path}/Looks/{material_name}"
        PreviewSurface(prim_path=opaque_mtl_path)
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=prim_path,
            material_path=opaque_mtl_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )
        return opaque_mtl_path

    def apply_saved_pose(self):
        self.set_local_pose(translation=self.default_pos, orientation=self.default_ori)
        self.set_local_scale(np.array(self.scale))

    def relocate_offscreen(self):
        _FAR_CENTER = (100000.0, 100000.0, 100000)
        _FAR_JITTER = 1000.0
        far_pos = (
            _FAR_CENTER[0] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[1] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[2] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
        )
        self.set_local_pose(translation=far_pos, orientation=self.default_ori)
        self.set_local_scale(np.array(self.scale))

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Return an owned CPU float32 XYZ/WXYZ snapshot of static USD geometry.

        ``is_relative`` subtracts the environment origin once. Static geometry
        uses its authored world transform; scale is removed from the quaternion.
        """
        try:
            world_tf = omni.usd.get_world_transform_matrix(self.prim)
            translation = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
            rot_quat = Gf.Transform(world_tf).GetRotation().GetQuat()
            orientation = np.array([rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(f"Cannot read the static geometry world pose for {self._prim_path}") from error
        if translation.shape != (3,) or orientation.shape != (4,):
            raise ValueError(f"Invalid static geometry pose shape for {self._prim_path}")
        if not np.isfinite(translation).all() or not np.isfinite(orientation).all():
            raise ValueError(f"Non-finite static geometry pose for {self._prim_path}")
        quat_norm = float(np.linalg.norm(orientation))
        if quat_norm <= np.finfo(np.float32).eps:
            raise ValueError(f"Invalid static geometry quaternion for {self._prim_path}")
        orientation /= quat_norm
        root_pose = torch.from_numpy(np.concatenate((translation, orientation)))
        if is_relative:
            origin = self.env_origin
            if isinstance(origin, torch.Tensor):
                env_origin = origin.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            else:
                env_origin = torch.as_tensor(origin, dtype=torch.float32, device="cpu").reshape(-1)
            if env_origin.numel() != 3 or not bool(torch.isfinite(env_origin).all()):
                raise ValueError(f"Environment origin must contain three finite coordinates for {self._prim_path}")
            root_pose[:3] -= env_origin
        return {"root_pose": root_pose}

    def initialize(self):
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        super().initialize(physics_sim_view=self.physics_sim_view)

    def hide_prim(self, prim_path: str):
        """
        Hide a prim by setting its visibility to invisible.
        This will make the prim and all its children invisible and ignored by physics.

        Args:
            prim_path: The prim path to hide
        """
        try:
            path = Sdf.Path(prim_path)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                print(f"Warning: Invalid prim path {prim_path}")
                return
            visibility_attribute = prim.GetAttribute("visibility")
            if visibility_attribute is None:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
            else:
                visibility_attribute.Set("invisible")
        except Exception as e:
            print(f"Warning: Failed to hide prim {prim_path}: {e}")

    def destroy(self):
        self._geometry_prim_view.disable_collision()
        self._geometry_prim_view.set_visibilities([False])
        self.hide_prim(self._prim_path)
