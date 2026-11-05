import warnings

from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
import numpy as np
from omegaconf import DictConfig
import omni.kit.commands
from omni.physx import get_physx_cooking_interface, get_physx_interface, get_physx_simulation_interface
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import PhysxSchema, Usd, UsdGeom, UsdShade, Vt
from scipy.spatial import cKDTree
import torch


class GarmentObject(SingleXFormPrim):
    """Original garment mesh and task interface, simulated by PhysX surface FEM.

    This is a new physical baseline, not a conversion of particle-cloth springs.
    All physics opinions are authored on the current stage, never in asset files.
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
        """
        Initialize the GarmentObject with position, orientation, and configuration.

        Args:
            prim_path: Path to the prim in the stage
            usd_path: Path to the USD asset file for this object
            inst_config: Configuration dictionary containing object properties
            env_origin: Origin position of the environment
            primitive_type: Type of primitive (only 'Plane' is supported)
        """
        if primitive_type is not None and primitive_type != "Plane":
            raise ValueError(
                f"GarmentObject '{prim_path}' only supports the 'Plane' primitive type, but received '{primitive_type}'"
            )

        # Parse prim path components
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.env_name = prim_path_parts[-4]

        # Configuration setup
        self.primitive_type = primitive_type
        self.instance_config = inst_config
        self.stage = get_current_stage()
        self._current_color = None

        self.visual_cfg = self.instance_config.get("visual", {})
        self.physics_cfg = self.instance_config.get("physics", {})

        self.surface_fem_config = {
            "youngs_modulus": 1.0e6,
            "poissons_ratio": 0.3,
            "density": 300.0,
            "surface_thickness": 0.001,
            "friction": 0.5,
            "elasticity_damping": 0.05,
            "bend_damping": 0.05,
            "self_collision": True,
            "solver_position_iteration_count": 32,
            "rest_offset": 0.001,
            "contact_offset": 0.003,
        }
        overrides = self.physics_cfg.get("surface_fem", {})
        unknown = set(overrides) - self.surface_fem_config.keys()
        if unknown:
            raise ValueError(f"Unknown surface_fem settings: {sorted(unknown)}")
        self.surface_fem_config.update(overrides)
        legacy = [key for key in ("garment_config", "particle_material", "particle_system") if key in self.physics_cfg]
        if legacy:
            warnings.warn(
                f"{prim_path}: surface FEM uses a new physics baseline; ignoring legacy {', '.join(legacy)}. "
                "Configure physics.surface_fem instead (see README).",
                UserWarning,
                stacklevel=2,
            )
        if self.instance_config.get("interaction_with_fluid", False):
            raise ValueError("Surface FEM does not support the former shared particle-system fluid interaction.")
        self._surface_view = None

        # USD path configurations
        self.usd_prim_path = prim_path
        self.usd_path = usd_path

        # Initial pose calculation
        self.env_origin = env_origin.detach().cpu().numpy()
        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            raise ValueError(f"Could not extract env_id from prim path: {self.usd_prim_path}")
        self.init_pos = default_pos
        self.init_ori = default_ori
        self.init_scale = scale

        if usd_path:
            add_reference_to_stage(usd_path=usd_path, prim_path=self.usd_prim_path)
            self.mesh_prim_path = self._find_first_mesh_in_hierarchy(self.usd_prim_path)
            if self.mesh_prim_path is None:
                raise RuntimeError(
                    f"Could not find a UsdGeom.Mesh prim under the referenced asset at {self.usd_prim_path}"
                )
        else:
            self.mesh_prim_path = self.usd_prim_path

        mesh = UsdGeom.Mesh.Get(self.stage, self.mesh_prim_path)
        if (
            not mesh
            or self.mesh_prim_path == self.usd_prim_path
            or any(count != 3 for count in mesh.GetFaceVertexCountsAttr().Get())
        ):
            raise ValueError(f"Surface FEM requires a triangle mesh below an Xform root at {self.mesh_prim_path}")
        # Keep point ordering: rewards refer to semantic vertex IDs in the asset.
        self.initial_points_positions = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
        self.triangle_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        if not len(self.initial_points_positions) or not len(self.triangle_indices):
            raise ValueError(f"Empty garment mesh at {self.mesh_prim_path}")
        super().__init__(
            name=self.usd_prim_path,
            scale=self.init_scale,
            prim_path=self.usd_prim_path,
            position=self.init_pos,
            orientation=self.init_ori,
        )
        # Keep the same layout frame as the former mesh-level wrapper: the asset's
        # tiny authored child translation was previously replaced by set_world_pose.
        UsdGeom.Xformable(mesh).ClearXformOpOrder()
        self._setup_surface_fem()

        # --- Visual Material and Visibility Setup ---
        # Set visibility based on config.
        visible = self.visual_cfg.get("visible", True)
        if not visible:
            imageable = UsdGeom.Imageable(self.prim)
            imageable.MakeInvisible()

        self._current_color = self.visual_cfg.get("color", None)
        if self._current_color is not None:
            self._apply_color_material(self._current_color)
        else:
            self.visual_usd_path = self.visual_cfg.get("visual_usd_path", None)
            if self.visual_usd_path:
                self._apply_visual_material(self.visual_usd_path)

        # Set initial pose
        self.set_world_pose(position=self.init_pos, orientation=self.init_ori)

    def _apply_color_material(self, color):
        """Creates or updates the PreviewSurface material with the specified color."""
        material_path = find_unique_string_name(
            initial_name=f"{self.usd_prim_path}/Looks/color_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        material_prim = get_prim_at_path(material_path)
        if not material_prim:
            material = PreviewSurface(prim_path=material_path, color=torch.tensor(color))
            # Bind the new material to the geometry prim and its submeshes
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.mesh_prim_path,  # Bind to the mesh prim
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )
            # Also bind to submeshes if they exist
            mesh_prim_to_bind = prims_utils.get_prim_at_path(self.mesh_prim_path)
            if mesh_prim_to_bind:
                garment_submesh = prims_utils.get_prim_children(mesh_prim_to_bind)
                if len(garment_submesh) > 0:
                    for sub_prim in garment_submesh:
                        if sub_prim.IsA(UsdGeom.Gprim):
                            omni.kit.commands.execute(
                                "BindMaterialCommand",
                                prim_path=sub_prim.GetPath(),
                                material_path=material_path,
                                strength=UsdShade.Tokens.strongerThanDescendants,
                            )
        else:
            material = PreviewSurface(prim_path=material_path)
            material.set_color(np.array(color))

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        """Recursively searches for the first prim of type UsdGeom.Mesh under the given path."""
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None

        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString

        return None

    def _setup_surface_fem(self):
        cfg = self.surface_fem_config
        self.simulation_mesh_path = f"{self.usd_prim_path}/simulation_mesh"
        if not deformableUtils.create_auto_surface_deformable_hierarchy(
            self.stage,
            root_prim_path=self.usd_prim_path,
            simulation_mesh_path=self.simulation_mesh_path,
            cooking_src_mesh_path=self.mesh_prim_path,
            cooking_src_simplification_enabled=False,
            set_visibility_with_guide_purpose=True,
        ):
            raise RuntimeError(f"Failed to author surface FEM at {self.usd_prim_path}")
        get_physx_cooking_interface().cook_auto_deformable_body(self.usd_prim_path)
        simulation_mesh = UsdGeom.Mesh.Get(self.stage, self.simulation_mesh_path)
        self.initial_simulation_points = np.array(simulation_mesh.GetPointsAttr().Get(), dtype=np.float32)
        # Official cooking welds coincident seam vertices. Preserve every original
        # semantic ID through an explicit rest-space map, never a nearest-node guess
        # on a deformed or simplified mesh.
        distances, self.vertex_to_simulation_node = cKDTree(self.initial_simulation_points).query(
            self.initial_points_positions
        )
        if not np.all(distances <= 1.0e-7):
            raise RuntimeError(f"Surface FEM cooking changed rest vertices at {self.mesh_prim_path}")
        prim = self.prim
        prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        prim.GetAttribute("physxDeformableBody:selfCollision").Set(bool(cfg["self_collision"]))
        prim.GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(True)
        prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(
            int(cfg["solver_position_iteration_count"])
        )
        collision = PhysxSchema.PhysxCollisionAPI.Apply(simulation_mesh.GetPrim())
        collision.CreateRestOffsetAttr().Set(float(cfg["rest_offset"]))
        collision.CreateContactOffsetAttr().Set(float(cfg["contact_offset"]))
        self.physics_material_path = f"{self.usd_prim_path}/Looks/surface_fem_material"
        if not deformableUtils.add_surface_deformable_material(
            self.stage,
            self.physics_material_path,
            youngs_modulus=float(cfg["youngs_modulus"]),
            poissons_ratio=float(cfg["poissons_ratio"]),
            density=float(cfg["density"]),
            surface_thickness=float(cfg["surface_thickness"]),
            static_friction=float(cfg["friction"]),
            dynamic_friction=float(cfg["friction"]),
        ):
            raise RuntimeError(f"Failed to author surface FEM material at {self.physics_material_path}")
        material = self.stage.GetPrimAtPath(self.physics_material_path)
        material.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
        material.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(float(cfg["elasticity_damping"]))
        material.GetAttribute("physxDeformableMaterial:bendDamping").Set(float(cfg["bend_damping"]))
        physicsUtils.add_physics_material_to_prim(self.stage, prim, self.physics_material_path)
        # Explicit total mass from the scaled rest surface (not old per-particle mass).
        points = self._points_to_world(self.initial_points_positions)
        triangles = points[self.triangle_indices]
        self.rest_surface_area = float(
            0.5
            * np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
            ).sum()
        )
        self.total_mass = self.rest_surface_area * float(cfg["density"]) * float(cfg["surface_thickness"])
        if not np.isfinite(self.total_mass) or self.total_mass <= 0:
            raise ValueError(f"Invalid surface FEM mass for {self.mesh_prim_path}: {self.total_mass}")
        prim.GetAttribute("omniphysics:mass").Set(self.total_mass)

    def _points_to_world(self, points):
        transform = np.array(UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        return np.asarray(points) @ transform[:3, :3] + transform[3, :3]

    def initialize(self):
        """Bind the native surface FEM tensor view after scene creation."""
        get_physx_simulation_interface().flush_changes()
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        if self.physics_sim_view is None:
            raise RuntimeError("Surface FEM requires an initialized PhysX simulation view")
        # PhysX 110.4's CPU tensor frontend does not implement deformable nodal
        # access. CPU readback uses PhysX's USD publication, with the same GPU FEM
        # solver; CUDA uses the native tensor view directly.
        if self._device == "cpu":
            return
        self._surface_view = self.physics_sim_view.create_surface_deformable_body_view(self.usd_prim_path)
        if self._surface_view.count != 1 or self._surface_view.max_simulation_nodes_per_body != len(
            self.initial_simulation_points
        ):
            raise RuntimeError(f"Surface FEM mesh identity mismatch at {self.mesh_prim_path}")

    def reset(self):
        """Restore the undeformed mesh at the current pose and zero native velocities."""
        if self._device == "cpu":
            mesh = UsdGeom.Mesh.Get(self.stage, self.simulation_mesh_path)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.initial_simulation_points))
            mesh.CreateVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros_like(self.initial_simulation_points)))
            get_physx_simulation_interface().flush_changes()
            return
        positions = self._backend_utils.convert(
            self._points_to_world(self.initial_simulation_points)[None].astype(np.float32), self._device
        )
        indices = self._backend_utils.create_tensor_from_list([0], dtype="int32", device=self._device)
        self._surface_view.set_simulation_nodal_positions(positions, indices)
        self._surface_view.set_simulation_nodal_velocities(positions * 0, indices)

    def apply_saved_pose(self):
        self.set_world_pose(np.asarray(self.init_pos, dtype=np.float32) + self.env_origin, self.init_ori)
        self.set_local_scale(self.init_scale)
        self.reset()

    def sample_mesh_vertices(self):
        """Return current native FEM vertices in original asset order and the layout frame."""
        if self._device == "cpu":
            # Publish native state even after unrendered steps. Never sample the
            # static cooking source or depend on camera rendering for rewards.
            get_physx_interface().update_transformations(False, True, True, True)
            mesh = UsdGeom.Mesh.Get(self.stage, self.simulation_mesh_path)
            local_nodes = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            world_points = self._points_to_world(local_nodes[self.vertex_to_simulation_node])
        else:
            positions = self._surface_view.get_simulation_nodal_positions()
            world_points = positions[0].detach().cpu().numpy()[self.vertex_to_simulation_node].copy()
        transform = np.array(UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        local_points = (world_points - transform[3, :3]) @ np.linalg.inv(transform[:3, :3])
        pos_world, ori_world = self.get_world_pose()
        return world_points, local_points, pos_world, ori_world

    def close(self):
        """Release only this body's view; the scene owns the simulation view."""
        self._surface_view = None
        self.physics_sim_view = None

    def _apply_visual_material(self, material_path: str):
        """Apply a visual material to the garment mesh."""
        self.visual_material_path = find_unique_string_name(
            self.usd_prim_path + "/Looks/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        add_reference_to_stage(usd_path=material_path, prim_path=self.visual_material_path)

        self.visual_material_prim = prims_utils.get_prim_at_path(self.visual_material_path)
        # Check if visual_material_prim is valid and has children
        if not self.visual_material_prim or not self.visual_material_prim.IsValid():
            print(f"Warning: Could not get valid prim at {self.visual_material_path}")
            return
        children = prims_utils.get_prim_children(self.visual_material_prim)
        if not children:
            print(f"Warning: Material prim at {self.visual_material_path} has no children.")
            return

        self.material_prim = children[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)

        mesh_prim_to_bind = prims_utils.get_prim_at_path(self.mesh_prim_path)
        if not mesh_prim_to_bind:
            print(f"Warning: Could not find mesh prim at {self.mesh_prim_path} to bind material.")
            return

        # Apply material to main mesh with strongerThanDescendants
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.mesh_prim_path,
            material_path=self.material_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )

        # Apply material to submeshes if any, also with strongerThanDescendants
        garment_submesh = prims_utils.get_prim_children(mesh_prim_to_bind)
        if len(garment_submesh) > 0:
            for prim in garment_submesh:
                if prim.IsA(UsdGeom.Gprim):
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=prim.GetPath(),
                        material_path=self.material_prim_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )

    def _extract_env_id_from_prim_path(self):
        """get env_id from prim_path"""
        try:
            parts = self.usd_prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def transform_points(self, points, pos, ori, scale):
        """
        Transform local points to world space using position, orientation, and scale.

        Args:
            points: (N, 3) array of local points
            pos: (3,) position vector
            ori: (4,) quaternion orientation
            scale: Scale factor (numpy array)

        Returns:
            (N, 3) array of transformed points in world space
        """
        ori_matrix = quat_to_rot_matrix(ori)  # Expects numpy array, returns numpy array
        scaled_points = points * scale  # element-wise multiplication if scale is numpy array
        transformed_points = scaled_points @ ori_matrix.T + pos
        return transformed_points

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Return the layout frame [position, WXYZ]; deformation is in the vertices."""
        position, orientation = self.get_world_pose()
        position = torch.as_tensor(position, dtype=torch.float32).clone()
        orientation = torch.as_tensor(orientation, dtype=torch.float32).clone()
        if is_relative:
            position -= torch.as_tensor(self.env_origin, dtype=torch.float32, device=position.device)
        return {"root_pose": torch.cat((position, orientation))}
