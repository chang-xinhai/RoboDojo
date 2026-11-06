"""Ground object with material domain randomization support."""

import os
from pathlib import Path
import random
from typing import Dict, List

from isaaclab.sim.utils import get_current_stage
import numpy as np
from omegaconf import DictConfig
from pxr import Gf, Sdf, UsdGeom, UsdPhysics
from env.global_configs import ASSETS_PATH
from .physics_material import PhysicsMaterial
from .scene_object import bind_material, color_material, mdl_material, set_pose


def resolve_mdl_paths(mdl_path_or_folder: str) -> List[str]:
    """Resolve MDL file paths from either a single .mdl file or a folder.

    Args:
        mdl_path_or_folder: Path to a .mdl file or a directory containing .mdl files.

    Returns:
        A list of absolute paths to .mdl files.
    """
    path = Path(mdl_path_or_folder).expanduser()
    if not path.exists():
        return []
    if path.is_file() and path.suffix.lower() == ".mdl":
        return [str(path.resolve())]
    if path.is_dir():
        return [str(p.resolve()) for p in path.glob("**/*.mdl")]
    return []


class Ground:
    """Ground plane with material domain randomization.

    This class creates a ground plane for each parallel environment and supports
    material randomization through MDL files or color changes.

    Args:
        prim_path: USD prim path for the ground
        config: Configuration containing ground parameters including:
            - size: Ground plane size (should match env_spacing)
            - materials: List of material configurations for randomization
            - axis: Ground plane axis (default "Z")
            - position: Ground plane position (default [0, 0, 0])
            - thickness: Ground plane thickness (default 0.1)
            - geometry: Geometry type for ground (default "cube", options: "cube", "plane")
    """

    def __init__(self, prim_path: str, config: DictConfig, env_spacing: float):
        """Initialize the Ground object.

        Args:
            prim_path: USD prim path for the ground
            config: Ground configuration
            env_spacing: Environment spacing (used as ground size)
            env_origin: Environment origin position
        """
        self.prim_path = prim_path
        self.config = config
        self.env_spacing = env_spacing
        self.stage = get_current_stage()

        # Extract configuration parameters
        self.axis = config.get("axis", "Z")
        self.position = config.get("default_pos", [0.0, 0.0, 0.0])
        self.base_color = config.get("base_color", [0.8, 0.8, 0.8])
        self.thickness = config.get("thickness", 0.1)  # Ground thickness
        self.geometry_type = config.get("geometry", "cube")  # cube or plane

        # Material randomization configuration
        self.materials = config.get("materials", None)
        # Process materials to expand folder paths
        if self.materials:
            self.materials = self._process_material_configs(self.materials)

        self.use_material_randomization = self.materials is not None and len(self.materials) > 0

        # Current material state
        self.current_material_path = None
        self.current_visual_material = None

        # Physics material configuration
        self.physics_material_config = config.get("physics_material", {})

        # Geometry wrapper (will be set in create)
        self.geometry_prim = None

        self.create()
        self.initialize()

    def _process_material_configs(self, material_configs: List[Dict]) -> List[Dict]:
        """Process material configurations to expand folder paths into individual files.

        Args:
            material_configs: List of material configurations

        Returns:
            Expanded list of material configurations with individual MDL files
        """
        expanded_configs = []
        material_type = material_configs.get("type", "color")
        if material_type == "color":
            # Color materials don't need expansion
            expanded_configs.append(material_configs)
        elif material_type == "mdl":
            # MDL materials may need folder expansion
            mdl_folder = f"{ASSETS_PATH}/Material"
            if not material_configs.get("random", False):
                file_name = material_configs.get("default")
                mdl_file = os.path.join(mdl_folder, f"{file_name}")
                expanded_configs.append(
                    {
                        "type": "mdl",
                        "mdl_file": mdl_file,
                    }
                )
            else:
                mdls = os.listdir(mdl_folder)
                for mdl_file in mdls:
                    if mdl_file.startswith("material_"):
                        expanded_configs.append(
                            {
                                "type": "mdl",
                                "mdl_file": os.path.join(mdl_folder, mdl_file),
                            }
                        )
        else:
            # Unknown type, keep as is
            expanded_configs.append(material_configs)

        return expanded_configs

    def create(self):
        """Create the ground plane in the USD stage."""
        if self.geometry_type == 'cube':
            self._create_cube_ground(None)
        elif self.geometry_type == 'plane':
            self._create_plane_ground(None)
        else:
            raise ValueError(f'Unsupported ground geometry: {self.geometry_type}')
        self.geometry_prim = self.stage.GetPrimAtPath(self.prim_path)
        UsdPhysics.CollisionAPI.Apply(self.geometry_prim).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(self.geometry_prim).CreateApproximationAttr().Set('none')
        material = PhysicsMaterial(f'{self.prim_path}/physics_material', config=self.physics_material_config)
        bind_material(self.geometry_prim, str(material.material.GetPath()), physics=True)

    def _create_cube_ground(self, physics_material):
        """Create ground using a Mesh primitive (Cube shape) with UVs."""
        # Create mesh instead of Cube to support UVs
        mesh_prim = UsdGeom.Mesh.Define(self.stage, self.prim_path)

        # Calculate size based on axis and env_spacing
        if self.axis.upper() == "Z":
            size_x = self.env_spacing
            size_y = self.env_spacing
            size_z = self.thickness
            position = Gf.Vec3d(
                float(self.position[0]),
                float(self.position[1]),
                float(self.position[2]) - self.thickness / 2,
            )
        elif self.axis.upper() == "Y":
            size_x = self.env_spacing
            size_y = self.thickness
            size_z = self.env_spacing
            position = Gf.Vec3d(
                float(self.position[0]),
                float(self.position[1]) - self.thickness / 2,
                float(self.position[2]),
            )
        elif self.axis.upper() == "X":
            size_x = self.thickness
            size_y = self.env_spacing
            size_z = self.env_spacing
            position = Gf.Vec3d(
                float(self.position[0]) - self.thickness / 2,
                float(self.position[1]),
                float(self.position[2]),
            )
        else:
            print(f"Warning: Unknown axis '{self.axis}', using Z.")
            size_x = self.env_spacing
            size_y = self.env_spacing
            size_z = self.thickness
            position = Gf.Vec3d(
                float(self.position[0]),
                float(self.position[1]),
                float(self.position[2]) - self.thickness / 2,
            )

        # Define a cube of size 2.0 (extents -1 to 1) to match original UsdGeom.Cube(2.0) logic
        points = [
            Gf.Vec3f(-1, -1, -1),  # 0
            Gf.Vec3f(1, -1, -1),  # 1
            Gf.Vec3f(-1, 1, -1),  # 2
            Gf.Vec3f(1, 1, -1),  # 3
            Gf.Vec3f(-1, -1, 1),  # 4
            Gf.Vec3f(1, -1, 1),  # 5
            Gf.Vec3f(-1, 1, 1),  # 6
            Gf.Vec3f(1, 1, 1),  # 7
        ]

        face_vertex_counts = [4] * 6
        face_vertex_indices = [
            4,
            5,
            7,
            6,  # +Z
            1,
            0,
            2,
            3,  # -Z
            2,
            6,
            7,
            3,  # +Y
            0,
            1,
            5,
            4,  # -Y
            5,
            1,
            3,
            7,  # +X
            0,
            4,
            6,
            2,  # -X
        ]

        normals = [
            Gf.Vec3f(0, 0, 1),  # +Z
            Gf.Vec3f(0, 0, -1),  # -Z
            Gf.Vec3f(0, 1, 0),  # +Y
            Gf.Vec3f(0, -1, 0),  # -Y
            Gf.Vec3f(1, 0, 0),  # +X
            Gf.Vec3f(-1, 0, 0),  # -X
        ]

        # UVs (FaceVarying - 4 per face)
        uvs = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
        ] * 6

        # Set attributes
        mesh_prim.GetPointsAttr().Set(points)
        mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
        mesh_prim.GetNormalsAttr().Set(normals)
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.uniform)

        # Set UVs
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        uv_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        uv_primvar.Set(uvs)

        # Apply scale to match desired size (base size is 2.0)
        scale = Gf.Vec3f(size_x / 2.0, size_y / 2.0, size_z / 2.0)
        set_pose(mesh_prim.GetPrim(), position, scale=scale)

    def _create_plane_ground(self, physics_material: PhysicsMaterial):
        """Create a plane mesh.

        Args:
            physics_material: Physics material to apply
        """
        # Create mesh for plane
        mesh_prim = UsdGeom.Mesh.Define(self.stage, self.prim_path)

        # Define plane vertices based on env_spacing
        half_size = self.env_spacing / 2

        if self.axis.upper() == "Z":
            # Ground is XY plane
            vertices = [
                Gf.Vec3f(-half_size, -half_size, 0.0),  # Bottom-left
                Gf.Vec3f(half_size, -half_size, 0.0),  # Bottom-right
                Gf.Vec3f(half_size, half_size, 0.0),  # Top-right
                Gf.Vec3f(-half_size, half_size, 0.0),  # Top-left
            ]
            normals = [Gf.Vec3f(0, 0, 1)] * 4
        elif self.axis.upper() == "Y":
            # Ground is XZ plane
            vertices = [
                Gf.Vec3f(-half_size, 0.0, -half_size),
                Gf.Vec3f(half_size, 0.0, -half_size),
                Gf.Vec3f(half_size, 0.0, half_size),
                Gf.Vec3f(-half_size, 0.0, half_size),
            ]
            normals = [Gf.Vec3f(0, 1, 0)] * 4
        else:  # X
            # Ground is YZ plane
            vertices = [
                Gf.Vec3f(0.0, -half_size, -half_size),
                Gf.Vec3f(0.0, half_size, -half_size),
                Gf.Vec3f(0.0, half_size, half_size),
                Gf.Vec3f(0.0, -half_size, half_size),
            ]
            normals = [Gf.Vec3f(1, 0, 0)] * 4

        # Set mesh attributes
        mesh_prim.GetPointsAttr().Set(vertices)
        mesh_prim.GetFaceVertexCountsAttr().Set([4])  # One quad face
        mesh_prim.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh_prim.GetNormalsAttr().Set(normals)

        # Set UVs
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        uv_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        uv_primvar.Set(uvs)

        # Set position
        position = np.array(
            [
                float(self.position[0]),
                float(self.position[1]),
                float(self.position[2]),
            ]
        )
        mesh_prim.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))

        # Set display color
        color_attr = mesh_prim.GetDisplayColorAttr()
        if not color_attr:
            color_attr = mesh_prim.CreateDisplayColorAttr()
        color_attr.Set(
            [
                Gf.Vec3f(
                    float(self.base_color[0]),
                    float(self.base_color[1]),
                    float(self.base_color[2]),
                )
            ]
        )

    def initialize(self):
        """Initialize the ground with the configuration after the first scene reload."""
        self.reset()

    def reset(self):
        """Apply the saved ground configuration and perform material randomization."""
        if self.use_material_randomization:
            self._randomize_material()

    def _randomize_material(self):
        """Randomize ground material.

        This method supports two types of material randomization:
        1. MDL materials: Apply MDL material files
        2. Color materials: Apply random colors using PreviewSurface
        """
        if not self.materials:
            return
        # Select a random material configuration
        material_config = random.choice(self.materials)
        material_type = material_config.get("type", "color")

        if material_type == "mdl":
            self._apply_mdl_material(material_config)
        elif material_type == "color":
            self._apply_color_material(material_config)
        else:
            print(f"Warning: Unknown material type '{material_type}' for ground. Skipping.")

    def _apply_mdl_material(self, material_config):
        paths = resolve_mdl_paths(material_config.get('mdl_file', ''))
        if not paths:
            raise FileNotFoundError(f'No MDL at {material_config}')
        path = f'{self.prim_path}/Looks/mdl_{Path(paths[0]).stem}'
        self.current_material_path = mdl_material(path, paths[0])
        bind_material(self.geometry_prim, self.current_material_path)

    def _apply_color_material(self, material_config):
        path = f'{self.prim_path}/Looks/color_material'
        self.current_material_path = color_material(path, material_config.get('color', self.base_color))
        bind_material(self.geometry_prim, self.current_material_path)
