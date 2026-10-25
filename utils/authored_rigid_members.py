"""Map a legacy rigid carrier to its authored native assembly without recooking it."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = position
    return matrix


def matrix_pose(matrix):
    matrix = np.asarray(matrix, float)
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or np.linalg.det(matrix[:3, :3]) <= 0
    ):
        raise ValueError("Authored member pose requires a positive rigid transform")
    position = matrix[:3, 3].copy()
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return position, quaternion


def inspect_authored_members(root):
    from pxr import Usd, UsdGeom, UsdPhysics

    bodies = [p for p in Usd.PrimRange(root) if p != root and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    roots = [p for p in bodies if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not roots:
        return None
    if len(roots) != 1:
        raise ValueError("Ambiguous authored rigid assembly roots")
    cache = UsdGeom.XformCache()
    world = np.asarray(cache.GetLocalToWorldTransform(root)).T
    members = {
        str(p.GetPath()): np.linalg.inv(world) @ np.asarray(cache.GetLocalToWorldTransform(p)).T
        for p in bodies
    }
    for matrix in members.values():
        matrix_pose(matrix)
    return dict(
        asset_root=str(root.GetPath()),
        canonical_body=str(roots[0].GetPath()),
        members=members,
        world_from_parent=np.asarray(cache.GetLocalToWorldTransform(root.GetParent())).T,
    )


def array(value):
    import warp as wp

    if isinstance(value, wp.array):
        value = wp.to_torch(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def backend(value, reference, indices=False):
    import torch
    import warp as wp

    if isinstance(reference, wp.array):
        return wp.array(
            value,
            dtype=wp.uint32 if indices else wp.float32,
            device=reference.device,
        )
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(
            value,
            dtype=torch.int32 if indices else torch.float32,
            device=reference.device,
        )
    return np.asarray(value, dtype=np.uint32 if indices else np.float32)


class AuthoredRigidMembers:
    def __init__(self, spec, physics_view):
        self.spec = spec
        self.views = {path: physics_view.create_rigid_body_view([path]) for path in spec["members"]}
        if any(view.count != 1 for view in self.views.values()):
            raise RuntimeError("Authored body not uniquely mapped")
        self.art = physics_view.create_articulation_view([spec["canonical_body"]])
        if self.art.count != 1:
            raise RuntimeError("Authored articulation root not uniquely mapped")
        self.initial_dofs = (
            array(self.art.get_dof_positions()).copy() if self.art.max_dofs else None
        )

    def world_pose(self):
        body = array(self.views[self.spec["canonical_body"]].get_transforms())[0]
        world = pose_matrix(body[:3], body[[6, 3, 4, 5]]) @ np.linalg.inv(
            self.spec["members"][self.spec["canonical_body"]]
        )
        return matrix_pose(world)

    def local_pose(self):
        position, quaternion = self.world_pose()
        return matrix_pose(
            np.linalg.inv(self.spec["world_from_parent"]) @ pose_matrix(position, quaternion)
        )

    def reset(self, position, quaternion_wxyz):
        world = self.spec["world_from_parent"] @ pose_matrix(position, quaternion_wxyz)
        for path, view in self.views.items():
            transform_position, transform_quaternion = matrix_pose(
                world @ self.spec["members"][path]
            )
            transform = np.r_[transform_position, transform_quaternion[[1, 2, 3, 0]]].astype(
                np.float32
            )
            if path == self.spec["canonical_body"]:
                reference = self.art.get_root_transforms()
                indices = backend([0], reference, indices=True)
                self.art.set_root_transforms(backend(transform, reference), indices)
                velocities = self.art.get_root_velocities()
                self.art.set_root_velocities(
                    backend(np.zeros((1, 6)), velocities), indices
                )
                if self.initial_dofs is not None:
                    self.art.set_dof_positions(
                        backend(self.initial_dofs, reference), indices
                    )
            else:
                reference = view.get_transforms()
                indices = backend([0], reference, indices=True)
                view.set_transforms(backend(transform, reference), indices)
                velocities = view.get_velocities()
                view.set_velocities(backend(np.zeros((1, 6)), velocities), indices)
