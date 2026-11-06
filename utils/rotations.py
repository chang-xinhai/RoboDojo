"""Rotation helpers. RoboDojo quaternions are XYZW; angles are radians unless named otherwise."""

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def quat_wxyz_to_xyzw(quaternion):
    """Convert a scalar-first external quaternion, preserving array type and batch axes."""
    q = quaternion if isinstance(quaternion, torch.Tensor) else np.asarray(quaternion)
    return q[..., [1, 2, 3, 0]]


def quat_xyzw_to_wxyz(quaternion):
    """Convert an internal quaternion for a scalar-first external API."""
    q = quaternion if isinstance(quaternion, torch.Tensor) else np.asarray(quaternion)
    return q[..., [3, 0, 1, 2]]


def quat_multiply(q1, q2):
    """Hamilton product in XYZW order (apply q2, then q1)."""
    if isinstance(q1, torch.Tensor):
        q2 = torch.as_tensor(q2, dtype=q1.dtype, device=q1.device)
        a, b = torch.broadcast_tensors(q1, q2)
        vector = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.linalg.cross(a[..., :3], b[..., :3])
        scalar = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(dim=-1, keepdim=True)
        return torch.cat((vector, scalar), dim=-1)
    a, b = np.broadcast_arrays(np.asarray(q1), np.asarray(q2))
    vector = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + np.cross(a[..., :3], b[..., :3])
    scalar = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(axis=-1, keepdims=True)
    return np.concatenate((vector, scalar), axis=-1)


def quat_inverse(q):
    """Multiplicative inverse of a nonzero XYZW quaternion."""
    if isinstance(q, torch.Tensor):
        norm_sq = (q * q).sum(dim=-1, keepdim=True)
        if torch.any(norm_sq == 0):
            raise ValueError("Cannot invert a zero quaternion")
        return q * q.new_tensor([-1, -1, -1, 1]) / norm_sq
    q = np.asarray(q)
    norm_sq = (q * q).sum(axis=-1, keepdims=True)
    if np.any(norm_sq == 0):
        raise ValueError("Cannot invert a zero quaternion")
    return q * [-1, -1, -1, 1] / norm_sq


def quat_rotate(vector, quaternion):
    """Actively rotate vector(s) by XYZW quaternion(s)."""
    if isinstance(quaternion, torch.Tensor):
        vector = torch.as_tensor(vector, dtype=quaternion.dtype, device=quaternion.device)
        pure = torch.cat((vector, torch.zeros_like(vector[..., :1])), dim=-1)
    else:
        vector = np.asarray(vector)
        pure = np.concatenate((vector, np.zeros_like(vector[..., :1])), axis=-1)
    return quat_multiply(quat_multiply(quaternion, pure), quat_inverse(quaternion))[..., :3]


def quat_to_rot_matrix(quaternion):
    """Convert XYZW quaternion(s) to active rotation matrices (NumPy)."""
    return Rotation.from_quat(quaternion).as_matrix()


def rot_matrix_to_quat(matrix):
    """Convert active rotation matrix/matrices to XYZW (NumPy)."""
    return Rotation.from_matrix(matrix).as_quat()


def euler_angles_to_quat(euler_angles, degrees=False, extrinsic=False, device=None):
    """Convert XYZ intrinsic (default) or xyz extrinsic Euler angles to XYZW tensors.

    Preserves the historical float32 return type and explicit output-device argument.
    """
    if isinstance(euler_angles, torch.Tensor):
        angles = euler_angles.detach().cpu().numpy()
    else:
        angles = np.asarray(euler_angles)
    result = Rotation.from_euler("xyz" if extrinsic else "XYZ", angles, degrees=degrees).as_quat()
    return torch.as_tensor(result, dtype=torch.float32, device=device)


def euler_to_quat(euler: list[float], order="xyz") -> list[float]:
    """Convert Euler angles in degrees to an XYZW quaternion."""
    return Rotation.from_euler(order, euler, degrees=True).as_quat().tolist()
