"""Real XPBD FEM garments for Isaac Sim 6.1 / Omni Physics 110."""

from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
import numpy as np
import omni.physics.tensors as tensors
from omni.physx.scripts import deformableUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdShade, Vt
import torch

from utils.surface_deformable import surface_material_parameters, triangulate_faces


class GarmentObject(SingleXFormPrim):
    """Surface cloth with live GPU nodal reads and zero-velocity resets.

    Particle cloth was removed in Sim 6.1. This uses the supported FEM solver;
    removed aerodynamics and cloth/fluid particle coupling fail explicitly.
    """

    def __init__(
        self, prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale, primitive_type=None
    ):
        if primitive_type not in (None, "Plane"):
            raise ValueError("GarmentObject only supports the Plane primitive")
        self.usd_prim_path, self.usd_path = prim_path, usd_path
        self.category_name, self.instance_name = prim_path.split("/")[-2:]
        self.env_name = prim_path.split("/")[-4]
        self.primitive_type, self.instance_config = primitive_type, inst_config
        self.stage = get_current_stage()
        self.visual_cfg = inst_config.get("visual", {})
        self.physics_cfg = inst_config.get("physics", {})
        self.env_origin = env_origin.detach().cpu().numpy()
        self.init_pos, self.init_ori, self.init_scale = default_pos, default_ori, scale
        self._surface_view = self._surface_sim_view = None
        if usd_path:
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        meshes = [p for p in Usd.PrimRange(get_prim_at_path(prim_path)) if p.IsA(UsdGeom.Mesh)]
        if len(meshes) != 1:
            raise ValueError(f"Garment '{prim_path}' requires exactly one mesh; found {len(meshes)}")
        prim = meshes[0]
        if prim.IsInstanceProxy():
            raise ValueError("Garment mesh must be editable; disable asset USD instanceable")
        self.mesh_prim_path, self._mesh = str(prim.GetPath()), UsdGeom.Mesh(prim)
        if primitive_type:
            # Primitive factories author physical dimensions in their local scale.
            # Preserve them when applying the layout's additional instance scale.
            native_scale = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation()).GetScale()
            self.init_scale = tuple(np.asarray(native_scale) * np.asarray(scale))
        if self._mesh.GetHoleIndicesAttr().Get():
            raise ValueError("Garment hole faces require explicit asset conversion")
        self._triangulate_mesh()
        self._rest_points = np.asarray(self._mesh.GetPointsAttr().Get(), dtype=np.float32).copy()
        physics = dict(self.physics_cfg)
        physics["interaction_with_fluid"] = inst_config.get(
            "interaction_with_fluid", physics.get("interaction_with_fluid", False)
        )
        self.surface_parameters = surface_material_parameters(physics, len(self._rest_points))
        super().__init__(
            prim_path=prim_path,
            name=prim_path,
            scale=np.asarray(self.init_scale),
            position=np.asarray(default_pos, dtype=np.float32) + self.env_origin,
            orientation=np.asarray(default_ori),
        )
        if not deformableUtils.set_physics_surface_deformable_body(self.stage, Sdf.Path(self.mesh_prim_path)):
            raise RuntimeError(f"Cannot author surface deformable: {self.mesh_prim_path}")
        if not prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI"):
            raise RuntimeError("PhysxSurfaceDeformableBodyAPI unavailable: Isaac Sim 6.1 required")
        self._set_attr(prim, "omniphysics:mass", self.surface_parameters["mass"])
        self._set_attr(prim, "omniphysics:deformableBodyEnabled", True)
        garment, system = physics.get("garment_config", {}), physics.get("particle_system", {})
        for attr, value in {
            "selfCollision": garment.get("self_collision", True),
            "disableGravity": self.surface_parameters["disable_gravity"],
            "solverPositionIterationCount": system.get("solver_position_iteration_count", 16),
            "enableSpeculativeCCD": system.get("enable_ccd", True),
        }.items():
            self._set_attr(prim, f"physxDeformableBody:{attr}", value)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(float(system.get("contact_offset", 0.01)))
        collision.CreateRestOffsetAttr().Set(float(system.get("rest_offset", 0.0)))
        self._create_surface_material()
        if not self.visual_cfg.get("visible", True):
            UsdGeom.Imageable(prim).MakeInvisible()
        self._current_color = self.visual_cfg.get("color")
        if self._current_color is not None:
            self._apply_color_material(self._current_color)
        elif self.visual_cfg.get("visual_usd_path"):
            self._apply_visual_material(self.visual_cfg["visual_usd_path"])
        self.physics_model = "omniphysics_surface_xpbd_fem"

    @staticmethod
    def _set_attr(prim, name, value):
        attr = prim.GetAttribute(name)
        if not attr or not attr.Set(value):
            raise RuntimeError(f"Required surface schema attribute unavailable: {name}")

    def _triangulate_mesh(self):
        counts = list(self._mesh.GetFaceVertexCountsAttr().Get())
        triangles, corners, faces = triangulate_faces(
            counts, self._mesh.GetFaceVertexIndicesAttr().Get(), len(self._mesh.GetPointsAttr().Get())
        )
        if all(n == 3 for n in counts):
            return
        for pv in UsdGeom.PrimvarsAPI(self._mesh).GetPrimvars():
            interpolation = pv.GetInterpolation()
            remap = corners if interpolation == "faceVarying" else faces if interpolation == "uniform" else None
            if remap is not None:
                attr = pv.GetIndicesAttr() if pv.IsIndexed() else pv.GetAttr()
                old = attr.Get()
                attr.Set(type(old)([old[int(i)] for i in remap]))
        normals = self._mesh.GetNormalsAttr().Get()
        if normals and self._mesh.GetNormalsInterpolation() in ("faceVarying", "uniform"):
            remap = corners if self._mesh.GetNormalsInterpolation() == "faceVarying" else faces
            self._mesh.GetNormalsAttr().Set(Vt.Vec3fArray([normals[int(i)] for i in remap]))
        for subset in UsdGeom.Subset.GetAllGeomSubsets(self._mesh):
            if subset.GetElementTypeAttr().Get() == "face":
                old = set(subset.GetIndicesAttr().Get())
                subset.GetIndicesAttr().Set([i for i, face in enumerate(faces) if face in old])
        self._mesh.GetFaceVertexCountsAttr().Set([3] * (len(triangles) // 3))
        self._mesh.GetFaceVertexIndicesAttr().Set(triangles.tolist())
        self._mesh.GetSubdivisionSchemeAttr().Set("none")

    def _create_surface_material(self):
        params = self.surface_parameters
        self.surface_material_path = self.usd_prim_path + "/Looks/surface_physics"
        args = {
            k: params[k]
            for k in (
                "youngs_modulus",
                "poissons_ratio",
                "surface_thickness",
                "surface_bend_stiffness",
                "dynamic_friction",
            )
        }
        if not deformableUtils.add_surface_deformable_material(self.stage, self.surface_material_path, **args):
            raise RuntimeError("Cannot author surface FEM material")
        prim = self.stage.GetPrimAtPath(self.surface_material_path)
        prim.ApplyAPI("PhysxDeformableMaterialAPI")
        prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
        self._set_attr(prim, "physxDeformableMaterial:elasticityDamping", params["elasticity_damping"])
        self._set_attr(prim, "physxDeformableMaterial:bendDamping", params["bend_damping"])
        UsdShade.MaterialBindingAPI.Apply(self._mesh.GetPrim()).Bind(
            UsdShade.Material(prim), UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    def initialize(self):
        # Lab 3's global view uses Warp. Keep this wrapper's frontend explicitly torch.
        self._surface_sim_view = tensors.create_simulation_view("torch")
        self._surface_sim_view.set_subspace_roots("/")
        self._surface_view = self._surface_sim_view.create_surface_deformable_body_view(self.mesh_prim_path)
        if self._surface_view.count != 1 or not self._surface_view.check():
            raise RuntimeError(f"No live surface deformable at {self.mesh_prim_path}")
        self.initial_points_positions = self._surface_view.get_simulation_nodal_positions().clone()

    def _world_points(self, points):
        transform = UsdGeom.Xformable(self._mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.asarray([transform.Transform(Gf.Vec3d(*map(float, p))) for p in points], dtype=np.float32)

    def reset(self):
        if self._surface_view is None:
            self._mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self._rest_points))
            self._mesh.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros_like(self._rest_points)))
            return
        template = self._surface_view.get_simulation_nodal_positions()
        points = torch.as_tensor(
            self._world_points(self._rest_points), device=template.device, dtype=template.dtype
        ).unsqueeze(0)
        indices = torch.zeros(1, device=template.device, dtype=torch.int32)
        self._surface_view.set_simulation_nodal_positions(points, indices)
        self._surface_view.set_simulation_nodal_velocities(torch.zeros_like(points), indices)

    def apply_saved_pose(self):
        self.set_world_pose(
            np.asarray(self.init_pos, dtype=np.float32) + self.env_origin, np.asarray(self.init_ori, dtype=np.float32)
        )
        self.set_local_scale(np.asarray(self.init_scale, dtype=np.float32))
        self.reset()

    def sample_mesh_vertices(self):
        if self._surface_view is not None:
            world = self._surface_view.get_simulation_nodal_positions()[0].detach().cpu().numpy().copy()
            inverse = UsdGeom.Xformable(self._mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            local = np.asarray([inverse.Transform(Gf.Vec3d(*map(float, p))) for p in world], dtype=np.float32)
        else:
            local = np.asarray(self._mesh.GetPointsAttr().Get(), dtype=np.float32)
            world = self._world_points(local)
        pos, ori = self.get_world_pose()
        return world, local, pos, ori

    def get_state(self, is_relative=False):
        world = self.sample_mesh_vertices()[0]
        # Cloth has no rigid root orientation: report live AABB center.
        center = (world.min(axis=0) + world.max(axis=0)) * 0.5
        if is_relative:
            center -= self.env_origin
        return {"root_pose": torch.tensor([*center, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)}

    def _apply_color_material(self, color):
        material = UsdShade.Material.Define(self.stage, self.usd_prim_path + "/Looks/color_material")
        shader = UsdShade.Shader.Define(self.stage, str(material.GetPath()) + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(self._mesh.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        self._current_color = color

    def _apply_visual_material(self, usd_path):
        path = self.usd_prim_path + "/Looks/visual_material"
        add_reference_to_stage(usd_path=usd_path, prim_path=path)
        materials = [p for p in Usd.PrimRange(self.stage.GetPrimAtPath(path)) if p.IsA(UsdShade.Material)]
        if not materials:
            raise ValueError(f"No UsdShade.Material found in {usd_path}")
        UsdShade.MaterialBindingAPI.Apply(self._mesh.GetPrim()).Bind(
            UsdShade.Material(materials[0]), UsdShade.Tokens.strongerThanDescendants
        )
