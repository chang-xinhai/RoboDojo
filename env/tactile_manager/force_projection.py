"""Project recorded official TacSL force components into the actual R1.5 pad.

The official values are retained separately. Native IsaacLab 3 PhysX poses and
TacSL taxel orientations use XYZW quaternions. No force law or sign is changed.
"""

import torch


def _unit_quaternion(value):
    if value.shape[-1] != 4 or not torch.isfinite(value).all():
        raise ValueError("Finite XYZW force-frame quaternions are required")
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(torch.abs(norm - 1) > 1e-3):
        raise ValueError("Force-frame quaternion must have unit length")
    return value / norm


def _rotate_xyzw(quaternion, vector):
    quaternion = quaternion.expand(*vector.shape[:-1], 4)
    imaginary = quaternion[..., :3]
    cross = torch.cross(imaginary, vector, dim=-1)
    return vector + 2 * (quaternion[..., 3:] * cross + torch.cross(imaginary, cross, dim=-1))


def project_force_to_pad(normal_force, shear_force, taxel_orientation_w_xyzw, gel_orientation_w_xyzw):
    """Return body-frame XYZ, signed outward −Y and tangential XZ components.

    Positive pad normal points outward from the gel; compression into the pad
    is negative. Signed original fields stay unchanged. Orientations are actual
    native body/taxel observations, not a guessed constant mounting rotation.
    """
    if shear_force.shape != (*normal_force.shape, 2):
        raise ValueError("Official TacSL normal and XY shear shapes must match")
    if not torch.isfinite(normal_force).all() or not torch.isfinite(shear_force).all():
        raise ValueError("Force projection requires finite actual TacSL estimates")
    original = torch.cat((shear_force, normal_force.unsqueeze(-1)), dim=-1)
    taxel = _unit_quaternion(taxel_orientation_w_xyzw)
    gel = _unit_quaternion(gel_orientation_w_xyzw)
    world_force = _rotate_xyzw(taxel, original)
    gel_inverse = torch.cat((-gel[..., :3], gel[..., 3:]), dim=-1)
    body_force = _rotate_xyzw(gel_inverse, world_force)
    return body_force, -body_force[..., 1], body_force[..., [0, 2]]
