"""Rigid coordinate boundary for official TacSL's native SDF queries.

PhysX queries unscaled mesh coordinates, while TacSL supplies rigid-body metres.
This boundary converts the rigid pose and uniform geometry scale, then rotates
gradients and returns distances in metres, leaving force equations intact.
"""

import torch
import warp as wp
import numpy as np


def shape_affine_in_body(mesh, body):
    """Return row-vector mesh-to-rigid-body-metre affine without authoring USD.

    Positive orthogonal axis scales are supported. Reflections, shear and
    degenerate frames are rejected rather than approximated by one scale.
    A scaled body is allowed only when its own scale is uniform, as before.
    """
    from pxr import Gf, UsdGeom

    cache = UsdGeom.XformCache()
    matrices = [cache.GetLocalToWorldTransform(p) for p in (mesh, body)]
    for index, matrix in enumerate(matrices):
        a = np.asarray(matrix, dtype=np.float64)
        if not np.isfinite(a).all() or not np.allclose(a[:3, 3], 0, atol=1e-12) or abs(a[3, 3]-1) > 1e-12:
            raise ValueError("TacSL requires finite affine shape/body transforms")
        axes = a[:3, :3]
        lengths = np.linalg.norm(axes, axis=1)
        if min(lengths) <= 1e-12 or np.linalg.det(axes) <= 0:
            raise ValueError("TacSL rejects reflected or degenerate shape/body transforms")
        normalized = axes / lengths[:, None]
        if not np.allclose(normalized @ normalized.T, np.eye(3), atol=1e-6, rtol=0):
            raise ValueError("TacSL rejects sheared shape/body transforms")
        if index == 1 and np.ptp(lengths) > 1e-5 * max(lengths):
            raise ValueError("TacSL requires positive uniform scale on the rigid body")
    transform = Gf.Transform(matrices[1])
    rigid = Gf.Matrix4d(1).SetRotate(transform.GetRotation())
    rigid.SetTranslateOnly(matrices[1].ExtractTranslation())
    return np.asarray(matrices[0] * rigid.GetInverse(), dtype=np.float64)


def shape_pose_in_body(mesh, body):
    """Resolve the physical shape's rigid pose, retaining uniform scale metadata."""
    from pxr import Gf, UsdGeom

    cache = UsdGeom.XformCache()
    rigid_world = []
    scales = []
    for prim in (mesh, body):
        matrix = cache.GetLocalToWorldTransform(prim)
        transform = Gf.Transform(matrix)
        scale = tuple(float(v) for v in transform.GetScale())
        if min(scale) <= 0 or max(scale) - min(scale) > 1e-5 * max(scale):
            raise ValueError(f"TacSL SDF frame requires positive uniform scale: {prim.GetPath()}, got {scale}")
        rigid = Gf.Matrix4d(1).SetRotate(transform.GetRotation())
        rigid.SetTranslateOnly(matrix.ExtractTranslation())
        rigid_world.append(rigid)
        scales.append(scale[0])
    relative = rigid_world[0] * rigid_world[1].GetInverse()
    rotation = [[float(relative[i][j]) for j in range(3)] for i in range(3)]
    translation = tuple(float(v) for v in relative.ExtractTranslation())
    return rotation, translation, scales[0]


class BodyFrameSdfView:
    """Delegate to the real native SDF view with body/shape frame conversion."""

    def __init__(self, native_view, rotation, translation, device, geometry_scale=1.0):
        self.native_view = native_view
        self.rotation = torch.tensor(rotation, dtype=torch.float32, device=device)
        self.translation = torch.tensor(translation, dtype=torch.float32, device=device)
        self.geometry_scale = geometry_scale

    def __getattr__(self, name):
        return getattr(self.native_view, name)

    def get_sdf_and_gradients(self, body_points):
        points = wp.to_torch(body_points) if isinstance(body_points, wp.array) else body_points
        shape_points = (((points - self.translation) @ self.rotation.T) / self.geometry_scale).contiguous()
        raw = self.native_view.get_sdf_and_gradients(wp.from_torch(shape_points))
        values = wp.to_torch(raw).clone()
        # Official TacSL uses native [gradient_xyz, distance] buffer layout.
        values[..., :3] = values[..., :3] @ self.rotation
        # Uniform scale cancels in the distance derivative; gradient orientation
        # changes, while the unscaled native distance converts back to metres.
        values[..., -1] *= self.geometry_scale
        return wp.from_torch(values)
