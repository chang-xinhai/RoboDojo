"""Compose static USD geometry with live PhysX/Fabric poses for task rewards.

USD world transforms do not follow DirectGPU bodies when readback is suppressed.
Native tensor transforms use XYZW; public RoboDojo poses retain WXYZ.
"""

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _physics_view(owner):
    view = getattr(owner, "physics_sim_view", None)
    if view is None:
        from isaacsim.core.simulation_manager import SimulationManager

        view = SimulationManager.get_physics_sim_view()
    return view


def get_live_world_transform(owner, prim):
    """Return a descendant's world affine using its nearest live rigid body.

    Mesh-to-body placement and body scale remain authored USD data. The moving
    body's position and orientation come from the same PhysX scene as the
    owner's native tensor view. Pure Xform assets read Fabric explicitly.
    """
    factory = _physics_view(owner)
    if factory is None:
        # Authoring occurs before the first warmup. There is no live solver
        # state yet, so the authored USD placement is authoritative.
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body = prim
    while body.IsValid() and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
        body = body.GetParent()
    if body.IsValid() and not body.IsPseudoRoot():
        path = str(body.GetPath())
        cache = getattr(owner, "_live_rigid_views", None)
        identity = (id(factory), id(owner.stage))
        if cache is None or getattr(owner, "_live_rigid_view_identity", None) != identity:
            cache = owner._live_rigid_views = {}
            owner._live_rigid_view_identity = identity
        if path not in cache:
            view = factory.create_rigid_body_view(path)
            if view.count != 1:
                raise RuntimeError(f"Expected exactly one native rigid body at {path}, found {view.count}")
            cache[path] = view
        pose = _numpy(cache[path].get_transforms())[0]
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise RuntimeError(f"Invalid native rigid body transform at {path}: {pose}")
        quaternion = Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6])))
        position = pose[:3]
        anchor = body
    else:
        anchor = owner.stage.GetPrimAtPath(owner._prim_path)
        view = getattr(owner, "_xform_prim_view", None)
        if view is None:
            raise RuntimeError(f"No native rigid body or Fabric Xform view for {prim.GetPath()}")
        positions, orientations = view.get_world_poses(usd=False)
        position = _numpy(positions)[0]
        wxyz = _numpy(orientations)[0]
        quaternion = Gf.Quatd(float(wxyz[0]), Gf.Vec3d(*map(float, wxyz[1:])))
    rest_anchor = UsdGeom.Xformable(anchor).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rest_descendant = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    relative = rest_descendant * rest_anchor.GetInverse()
    live = Gf.Transform()
    live.SetScale(Gf.Transform(rest_anchor).GetScale())
    live.SetRotation(Gf.Rotation(quaternion))
    live.SetTranslation(Gf.Vec3d(*map(float, position)))
    return relative * live.GetMatrix()
