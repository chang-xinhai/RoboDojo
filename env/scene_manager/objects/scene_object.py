"""Task geometry records backed by public USD authoring and native Lab assets.

Only mesh-to-body transforms are cached. Moving body poses always come from Lab,
including when Fabric deliberately leaves the USD transforms unchanged.
"""
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import get_current_stage
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation
import torch


def pose_matrix(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(quaternion)).as_matrix()
    matrix[:3, 3] = position
    return matrix


def matrix_pose(matrix):
    # Extract rotation independently from scale, including nonuniform asset scale.
    gf = Gf.Matrix4d(np.asarray(matrix).T.tolist())
    quat = gf.RemoveScaleShear().ExtractRotationQuat()
    return np.asarray(gf.ExtractTranslation()), np.array([*quat.GetImaginary(), quat.GetReal()])


def world_matrix(prim):
    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def set_pose(prim, position, quaternion=(0, 0, 0, 1), scale=(1, 1, 1)):
    matrix = pose_matrix(position, quaternion)
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(scale)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    # Reuse this named op on subsequent resets instead of accumulating xform ops.
    attr = prim.GetAttribute('xformOp:transform:robodojo')
    op = UsdGeom.XformOp(attr) if attr else xform.AddTransformOp(opSuffix='robodojo')
    op.Set(Gf.Matrix4d(matrix.T.tolist()))
    xform.SetXformOpOrder([op])


def reference(prim_path, usd_path=None):
    stage = get_current_stage()
    if usd_path:
        prim = stage.DefinePrim(prim_path, 'Xform')
        prim.GetReferences().AddReference(str(usd_path))
    else:
        prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f'No asset at {prim_path}: {usd_path}')
    return prim


def bind_material(prim, material_path, *, physics=False):
    material = UsdShade.Material.Get(prim.GetStage(), material_path)
    if not material:
        raise ValueError(f'Expected a UsdShade.Material at {material_path}')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.strongerThanDescendants,
        'physics' if physics else UsdShade.Tokens.allPurpose,
    )


def mdl_material(path, mdl_path, name=None):
    stage = get_current_stage()
    if not stage.GetPrimAtPath(path):
        cfg = sim_utils.MdlFileCfg(mdl_path=str(mdl_path))
        cfg.func(path, cfg)
    if name:
        UsdShade.Shader.Get(stage, path + '/Shader').SetSourceAssetSubIdentifier(name, 'mdl')
    return path


def color_material(path, color):
    stage = get_current_stage()
    if not stage.GetPrimAtPath(path):
        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(x) for x in color))
        cfg.func(path, cfg)
    else:
        UsdShade.Shader.Get(stage, path + '/Shader').CreateInput(
            'diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    return path


def apply_visual(prim, visual):
    path = str(prim.GetPath()) + '/Looks/override'
    if visual.get('color') is not None:
        bind_material(prim, color_material(path, visual['color']))
    elif visual.get('visual_usd_path'):
        material_root = reference(path, visual['visual_usd_path'])
        materials = [p for p in Usd.PrimRange(material_root) if p.IsA(UsdShade.Material)]
        if not materials:
            raise ValueError(f"No Material in {visual['visual_usd_path']}")
        bind_material(prim, str(materials[0].GetPath()))
    elif visual.get('visual_mdl_path'):
        bind_material(prim, mdl_material(path, visual['visual_mdl_path']))
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeVisible() if visual.get('visible', True) else imageable.MakeInvisible()


class SceneObject:
    """Object identity, saved placement and task geometry; no simulator lifecycle."""

    def __init__(self, prim_path, usd_path, inst_config, env_origin,
                 default_pos, default_ori, scale, primitive_type=None):
        self.prim = reference(prim_path, usd_path)
        self.stage = self.prim.GetStage()
        self.prim_path = self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.category_name, self.instance_name = prim_path.split('/')[-2:]
        self.instance_config = inst_config
        self.env_origin = np.asarray(torch.as_tensor(env_origin).cpu(), dtype=np.float32)
        self.default_pos, self.default_ori = tuple(default_pos), tuple(default_ori)
        self.scale = tuple(scale)
        self.primitive_type = primitive_type
        self.native_assets = []
        self._body_sources = {}
        set_pose(self.prim, self.default_pos, self.default_ori, self.scale)
        apply_visual(self.prim, inst_config.get('visual', {}))

    def _cache_geometry(self):
        """Cache each mesh transform relative to its nearest physics link."""
        self._geometry = []
        self._authored_root = pose_matrix(*matrix_pose(world_matrix(self.prim)))
        for prim in Usd.PrimRange(self.prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            if not points.size:
                continue
            body = prim
            while body and str(body.GetPath()) not in self._body_sources and body != self.prim:
                body = body.GetParent()
            key = str(body.GetPath()) if str(body.GetPath()) in self._body_sources else None
            frame = pose_matrix(*matrix_pose(world_matrix(body))) if key else self._authored_root
            to_frame = np.linalg.inv(frame) @ world_matrix(prim)
            self._geometry.append((key, points, to_frame))

    def _frame(self, key):
        if key is None:
            return pose_matrix(*matrix_pose(world_matrix(self.prim)))
        asset, body_id = self._body_sources[key]
        data = asset.data.root_link_pose_w if body_id is None else asset.data.body_link_pose_w
        pose = data.torch[0] if body_id is None else data.torch[0, body_id]
        pose = pose.detach().cpu().numpy()
        return pose_matrix(pose[:3], pose[3:])

    def _root_frame(self):
        if getattr(self, '_root_body_key', None):
            return self._frame(self._root_body_key) @ self._root_from_body
        return self._frame(None)

    def post_init(self):
        for asset in self.native_assets:
            if not asset.is_initialized:
                raise RuntimeError(f'Native asset not initialized: {asset.cfg.prim_path}')
        self._initial_poses = [a.data.root_link_pose_w.torch.clone() for a in self.native_assets]

    def _get_object_transform(self, device=None):
        pos, quat = matrix_pose(self._root_frame())
        device = device or 'cpu'
        return (torch.as_tensor(pos - self.env_origin, dtype=torch.float32, device=device),
                torch.as_tensor(quat, dtype=torch.float32, device=device), device)

    def get_state(self, is_relative=False):
        pos, quat = matrix_pose(self._root_frame())
        if is_relative:
            pos = pos - self.env_origin
        return {'root_pose': torch.as_tensor(np.r_[pos, quat], dtype=torch.float32)}

    def sample_mesh_vertices(self):
        """Return the historical first-mesh vertices and reference axes.

        Ancestor scale is included in the returned local points, so applying the
        returned pose reproduces the world points without consulting moving USD.
        """
        if not self._geometry:
            raise ValueError(f'No mesh vertices at {self.prim_path}')
        key, points, mesh_to_body = self._geometry[0]
        transform = self._frame(key) @ mesh_to_body
        world = points @ transform[:3, :3].T + transform[:3, 3]
        pos, quat = matrix_pose(transform)
        inverse = np.linalg.inv(pose_matrix(pos, quat))
        local = world @ inverse[:3, :3].T + inverse[:3, 3]
        return (world.astype(np.float32), local.astype(np.float32),
                pos.astype(np.float32), quat.astype(np.float32))

    def get_bbox(self, is_relative=True):
        _, points, pos, quat = self.sample_mesh_vertices()
        return (pos - self.env_origin if is_relative else pos, quat,
                np.r_[points.min(axis=0), points.max(axis=0)])

    def _place(self, offset):
        # The USD root only controls static geometry; moving bodies use writers.
        if not getattr(self, '_root_body_key', None):
            set_pose(self.prim, np.asarray(self.default_pos) + offset, self.default_ori, self.scale)
        for asset, initial in zip(self.native_assets, self._initial_poses):
            asset.reset([0])
            pose = initial.clone()
            pose[:, :3] += torch.as_tensor(offset, dtype=pose.dtype, device=pose.device)
            asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=[0])
            asset.write_root_com_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), dtype=pose.dtype, device=pose.device), env_ids=[0])

    def apply_saved_pose(self):
        self._place(np.zeros(3))

    def relocate_offscreen(self):
        # Intentional robot-reset staging. No live deletion or stale topology.
        self._place(np.full(3, 100000.0))
