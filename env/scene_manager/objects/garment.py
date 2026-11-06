"""Native surface FEM garments with the original mesh and reward vertex indices.

This is a new physical baseline: explicit garment surface material defaults replace
PBD spring/particle parameters, which have no dimensionally equivalent mapping.
The object's pose denotes the layout reference frame; deformation is represented
by native world-space nodes, not a rigid orientation fitted to the moving cloth.
"""
from dataclasses import fields

from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim.schemas import apply_collision_properties, apply_surface_deformable_properties
from isaaclab.sim.spawners.materials import spawn_deformable_body_material
from isaaclab.sim.utils import get_current_stage
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyCfg, PhysxSurfaceDeformableBodyCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg
import numpy as np
from omni.physx.scripts import deformableUtils
from pxr import Usd, UsdGeom
import torch

from .scene_object import SceneObject, bind_material, matrix_pose, pose_matrix, reference, world_matrix


def _triangulate_plane(mesh):
    """Triangulate Kit's generated quad grid without changing point IDs or UVs."""
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if np.all(counts == 3):
        return
    if not len(counts) or np.any(counts != 4):
        raise ValueError('Generated garment Plane must contain triangles or quads')
    corners = (4 * np.arange(len(counts))[:, None] + [0, 1, 2, 0, 2, 3]).reshape(-1)
    corner_attributes = []
    if mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying:
        corner_attributes.append(mesh.GetNormalsAttr())
    for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying:
            corner_attributes.append(primvar.GetIndicesAttr() if primvar.IsIndexed() else primvar.GetAttr())
    remapped = []
    for attr in corner_attributes:
        values = attr.Get()
        if values is None or len(values) != 4 * len(counts):
            raise ValueError(f'Invalid generated Plane corner data at {attr.GetPath()}')
        remapped.append((attr, [values[int(i)] for i in corners]))
    indices = deformableUtils.triangulate_mesh(mesh)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetFaceVertexCountsAttr().Set([3] * (len(indices) // 3))
    for attr, values in remapped:
        attr.Set(values)


def _deduplicate_surface_vertices(mesh):
    """Merge exact coincidences required by PhysX; retain original reward IDs."""
    points = np.asarray(mesh.GetPointsAttr().Get())
    _, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    if len(first) == len(points):
        return np.arange(len(points))
    # Existing garments use face-varying normals/UVs. Do not silently merge
    # independently authored per-vertex appearance or velocity data.
    if mesh.GetNormalsAttr().Get() and mesh.GetNormalsInterpolation() in ('vertex', 'varying'):
        raise ValueError('Coincident garment vertices have per-vertex normals')
    for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if primvar.Get() and primvar.GetInterpolation() in ('vertex', 'varying'):
            raise ValueError(f'Coincident garment vertices have per-vertex data: {primvar.GetPath()}')
    if mesh.GetVelocitiesAttr().Get():
        raise ValueError('Coincident garment vertices have authored velocities')
    order = np.argsort(first)
    remap = np.empty(len(first), dtype=np.int64)
    remap[order] = np.arange(len(first))
    original_to_native = remap[inverse]
    triangles = original_to_native[np.asarray(mesh.GetFaceVertexIndicesAttr().Get())].reshape(-1, 3)
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise ValueError('Coincident garment vertices form a degenerate triangle')
    mesh.GetPointsAttr().Set(points[np.sort(first)])
    mesh.GetFaceVertexIndicesAttr().Set(triangles.reshape(-1))
    return original_to_native


class GarmentObject(SceneObject):
    """One native surface body, driven by SceneManager's singleton lifecycle."""

    def __init__(self, prim_path, usd_path, inst_config, env_origin, default_pos,
                 default_ori, scale, primitive_type=None):
        if primitive_type not in (None, 'Plane'):
            raise ValueError(f'Garment only supports the Plane primitive: {primitive_type}')
        if inst_config.get('interaction_with_fluid', False):
            raise ValueError('Surface FEM does not implement the previous shared PBD particle system')
        if usd_path:
            # Own the placement Xform separately: never clear a referenced asset's
            # authored root/ancestor transforms to apply the saved layout pose.
            get_current_stage().DefinePrim(prim_path, 'Xform')
            reference(prim_path + '/Asset', usd_path)
        super().__init__(prim_path, None, inst_config, env_origin, default_pos,
                         default_ori, scale, primitive_type)
        self.usd_path = usd_path
        meshes = [p for p in Usd.PrimRange(self.prim) if p.IsA(UsdGeom.Mesh)]
        if len(meshes) != 1:
            raise ValueError(f'Garment requires exactly one mesh, found {len(meshes)} at {prim_path}')
        mesh = UsdGeom.Mesh(meshes[0])
        if primitive_type == 'Plane':
            _triangulate_plane(mesh)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        if not len(points) or not len(counts) or np.any(counts != 3):
            raise ValueError(f'Garment requires a nonempty triangle mesh: {mesh.GetPath()}')
        self.mesh_prim_path = str(mesh.GetPath())
        transform = world_matrix(mesh.GetPrim())
        self._rest_points_world = points @ transform[:3, :3].T + transform[:3, 3]
        self.vertex_to_native = _deduplicate_surface_vertices(mesh)
        native_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        self._native_rest_points_world = native_points @ transform[:3, :3].T + transform[:3, 3]
        self._simulation_triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        self._reference_pose = matrix_pose(world_matrix(self.prim))
        self._reference_inverse = np.linalg.inv(pose_matrix(*self._reference_pose))

        # Keep the rendered and simulated mesh identical. Exact seam duplicates
        # are expanded back to their original vertex IDs when rewards sample it.
        if not deformableUtils.set_physics_surface_deformable_body(self.stage, mesh.GetPath()):
            raise RuntimeError(f'Could not author surface FEM at {self.mesh_prim_path}')
        native = inst_config.get('physics', {}).get('fem_config', {})
        # Folding requires cloth self-contact; remaining body settings use the
        # current SDK defaults unless an explicit native override is supplied.
        body_options = {'self_collision': True}
        body_options.update(native.get('body', {}))
        body_cfg = PhysxDeformableBodyCfg(**body_options)
        surface_cfg = PhysxSurfaceDeformableBodyCfg(**native.get('surface', {}))
        if not apply_surface_deformable_properties(
                self.mesh_prim_path, [body_cfg, surface_cfg], stage=self.stage):
            raise RuntimeError(f'Could not configure surface FEM at {self.mesh_prim_path}')
        # Automatic offsets leave these thin garments visibly inflated. Keep
        # contact distances explicit and separate from the material thickness.
        collision_options = {'rest_offset': 0.001, 'contact_offset': 0.003}
        collision_options.update(native.get('collision', {}))
        if not apply_collision_properties(
                self.mesh_prim_path, [PhysxCollisionCfg(**collision_options)], stage=self.stage):
            raise RuntimeError(f'Could not configure FEM collisions at {self.mesh_prim_path}')
        # Garments use a 1 mm surface, not Lab's generic 1 cm material default.
        # Zero stiffness overrides retain PhysX's modulus/thickness-derived values.
        material_options = {
            'youngs_modulus': 1.0e6,
            'poissons_ratio': 0.3,
            'density': 300.0,
            'surface_thickness': 0.001,
            'static_friction': 0.5,
            'dynamic_friction': 0.5,
            'elasticity_damping': 0.05,
            'bend_damping': 0.05,
        }
        material_options.update(native.get('material', {}))
        material_cfg = PhysxSurfaceDeformableBodyMaterialCfg(**material_options)
        self.physics_material_path = prim_path + '/Looks/fem_material'
        spawn_deformable_body_material(self.physics_material_path, material_cfg)
        bind_material(mesh.GetPrim(), self.physics_material_path, physics=True)
        self.fem_config = {
            'model': 'surface_fem',
            'material': {f.name: getattr(material_cfg, f.name) for f in fields(material_cfg)
                         if f.name != 'func' and not f.name.startswith('_')},
            'body': body_options,
            'surface': dict(native.get('surface', {})),
            'collision': collision_options,
        }
        self.asset = DeformableObject(DeformableObjectCfg(prim_path=self.mesh_prim_path, spawn=None))
        self.native_assets = [self.asset]

    def post_init(self):
        if not self.asset.is_initialized:
            raise RuntimeError(f'Native garment not initialized: {self.mesh_prim_path}')
        view = self.asset.root_view
        rest = view.get_rest_nodal_positions().numpy()[0]
        triangles = view.get_simulation_element_indices().numpy()[0]
        expected = self._native_rest_points_world
        canonical = lambda faces: sorted(map(tuple, np.sort(faces, axis=1)))
        if (rest.shape != expected.shape or not np.isfinite(rest).all()
                or canonical(triangles) != canonical(self._simulation_triangles)):
            raise RuntimeError(f'FEM nodes do not preserve authored garment vertex indices: {self.mesh_prim_path}')
        # Native rest positions need not share the world frame. Edge lengths
        # verify rest geometry independently of translation/rotation. Warmup has
        # already advanced live nodes, so never compare those to the rest pose.
        edges = self._simulation_triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        lengths = lambda nodes: np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
        if not np.allclose(lengths(rest), lengths(expected), atol=1e-6, rtol=1e-4):
            raise RuntimeError(f'FEM rest geometry differs from authored garment: {self.mesh_prim_path}')
        initial = torch.zeros_like(self.asset.data.default_nodal_state_w.torch)
        if initial.shape != (1, len(expected), 6):
            raise RuntimeError(f'Unexpected FEM nodal state shape: {initial.shape}')
        initial[0, :, :3] = torch.as_tensor(expected, dtype=initial.dtype, device=initial.device)
        self._initial_nodal_state = initial

    def _place(self, offset):
        self.asset.reset([0])
        state = self._initial_nodal_state.clone()
        state[..., :3] += torch.as_tensor(offset, dtype=state.dtype, device=state.device)
        self.asset.write_nodal_state_to_sim_index(state, env_ids=[0])

    def relocate_offscreen(self):
        # Keep cloth below the scene during robot settling. A 1e5m translation
        # quantizes float32 world nodes by millimeters and damages small elements.
        self._place(np.array([0.0, 0.0, -10.0]))

    def sample_mesh_vertices(self):
        """Current native nodes, with fixed layout axes for garment predicates."""
        native_points = self.asset.data.nodal_pos_w.torch[0].detach().cpu().numpy()
        points = native_points[self.vertex_to_native].copy()
        local = points @ self._reference_inverse[:3, :3].T + self._reference_inverse[:3, 3]
        position, orientation = self._reference_pose
        return (points, local.astype(np.float32), position.astype(np.float32),
                orientation.astype(np.float32))
