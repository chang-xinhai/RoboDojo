"""Independent rotation and signed-force contracts; no simulation is claimed."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from env.tactile_manager.force_projection import project_force_to_pad


def test_two_nontrivial_native_rotations_match_independent_body_frame():
    normal = torch.tensor([5.0], dtype=torch.float64)
    shear = torch.tensor([[-2.0, 3.0]], dtype=torch.float64)
    taxel = torch.tensor(Rotation.from_euler("z", 90, degrees=True).as_quat()[None])
    gel = torch.tensor(Rotation.from_euler("x", 90, degrees=True).as_quat())
    body, pad_normal, pad_shear = project_force_to_pad(normal, shear, taxel, gel)
    torch.testing.assert_close(body, torch.tensor([[-3.0, 5.0, 2.0]], dtype=torch.float64))
    torch.testing.assert_close(pad_normal, torch.tensor([-5.0], dtype=torch.float64))
    torch.testing.assert_close(pad_shear, torch.tensor([[-3.0, 2.0]], dtype=torch.float64))
    assert normal.tolist() == [5]
    assert shear.tolist() == [[-2, 3]]


def test_actual_quaternion_projection_preserves_norm_and_roundtrips():
    rng = np.random.default_rng(73)
    original = rng.normal(size=(500, 3))
    taxel_rotation = Rotation.random(500, random_state=rng)
    gel_rotation = Rotation.random(random_state=rng)
    expected = gel_rotation.inv().apply(taxel_rotation.apply(original))
    values = torch.tensor(original)
    body, normal, tangential = project_force_to_pad(
        values[:, 2], values[:, :2], torch.tensor(taxel_rotation.as_quat()), torch.tensor(gel_rotation.as_quat())
    )
    np.testing.assert_allclose(body.numpy(), expected, atol=1e-12)
    torch.testing.assert_close(torch.linalg.vector_norm(body, dim=-1), torch.linalg.vector_norm(values, dim=-1))
    torch.testing.assert_close(normal.square() + tangential.square().sum(-1), values.square().sum(-1))
    roundtrip = taxel_rotation.inv().apply(gel_rotation.apply(body.numpy()))
    np.testing.assert_allclose(roundtrip, original, atol=1e-12)


def test_r15_outward_minus_y_keeps_compression_negative():
    body, normal, tangential = project_force_to_pad(
        torch.tensor([0.0]),
        torch.tensor([[0.0, -4.0]]),
        torch.tensor([[0.0, 0.0, -1.0, 0.0]]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(body, torch.tensor([[0.0, 4.0, 0.0]]))
    assert normal.item() == -4
    assert tangential.abs().sum().item() == 0


def test_invalid_frame_and_nonfinite_force_fail_clearly():
    with pytest.raises(ValueError, match="unit length"):
        project_force_to_pad(torch.ones(1), torch.ones(1, 2), torch.zeros(1, 4), torch.tensor([0.0, 0, 0, 1]))
    with pytest.raises(ValueError, match="finite actual"):
        project_force_to_pad(torch.tensor([float("nan")]), torch.ones(1, 2), torch.ones(1, 4), torch.ones(4))
