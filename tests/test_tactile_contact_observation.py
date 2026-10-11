"""Physical contact conditioning preserves raw optics and rejects hidden faults."""

from types import SimpleNamespace

import pytest
import torch

from env.tactile_manager.contact_observation import condition_contact_observation


def optical_buffers():
    rgb = torch.arange(18, dtype=torch.uint8).reshape(2, 3, 3)
    depth = torch.full((2, 3, 1), 0.018, dtype=torch.float32)
    # Actual renderer residuals must remain visible independently of contact.
    delta = torch.tensor([0.0, 0.0001, -0.0002, 0.0, 0.0003, 0.0]).reshape(2, 3, 1)
    return rgb, depth, delta


def test_no_contact_is_exact_zero_but_raw_optics_and_inputs_are_unchanged():
    rgb, depth, delta = optical_buffers()
    result = condition_contact_observation(rgb, depth, delta, torch.zeros(4), torch.zeros(4), torch.zeros(4, 2))
    assert result["contact_active"].dtype == torch.bool
    assert result["contact_active"].shape == torch.Size([])
    assert not result["contact_active"].item()
    for key in ("rgb", "depth", "optical_depth_delta"):
        assert torch.count_nonzero(result[key]) == 0
    for key, original in zip(("rgb", "depth", "optical_depth_delta"), (rgb, depth, delta), strict=True):
        assert torch.equal(result["raw_optical_" + ("depth_delta" if key == "optical_depth_delta" else key)], original)
        assert result[key].data_ptr() != original.data_ptr()
    assert torch.count_nonzero(delta) == 3
    assert torch.count_nonzero(rgb) == 17


def test_one_real_penetrating_taxel_preserves_full_actual_optical_sample():
    rgb, depth, delta = optical_buffers()
    penetration = torch.tensor([0.0, 0.0, 0.0001, 0.0])
    normal = torch.tensor([0.0, 0.0, -0.1, 0.0])
    result = condition_contact_observation(rgb, depth, delta, penetration, normal, torch.zeros(4, 2))
    assert result["contact_active"].item()
    for key, original in zip(("rgb", "depth", "optical_depth_delta"), (rgb, depth, delta), strict=True):
        assert torch.equal(result[key], original)
    assert normal[2] == -0.1


@pytest.mark.parametrize(
    "normal,shear",
    [(torch.tensor([0.0, 0.001]), torch.zeros(2, 2)), (torch.zeros(2), torch.tensor([[0.0, 0.001], [0.0, 0.0]]))],
)
def test_nonzero_force_without_penetration_is_not_silently_masked(normal, shear):
    with pytest.raises(RuntimeError, match="refusing to mask"):
        condition_contact_observation(*optical_buffers(), torch.zeros(2), normal, shear)


@pytest.mark.parametrize("penetration", [torch.tensor([-1e-8]), torch.tensor([float("nan")])])
def test_invalid_penetration_fails(penetration):
    with pytest.raises(ValueError, match="penetration"):
        condition_contact_observation(*optical_buffers(), penetration, torch.zeros(1), torch.zeros(1, 2))


def test_unavailable_force_cannot_claim_contact_conditioning():
    with pytest.raises(ValueError, match="available SDF"):
        condition_contact_observation(*optical_buffers(), None, None, None)


def test_manager_batches_contact_and_free_environment_without_leaking_optical_residuals():
    import warp as wp

    from env.tactile_manager import TactileManager

    wp.init()
    manager = TactileManager(
        2,
        {
            "enabled": True,
            "contact_object_labels": ["tube0", "tube1", "tube2"],
            "contact_conditioned_observations": True,
        },
        None,
        "cpu",
    )
    manager.contact_paths_by_env = [[f"/env_{i}/tube0/collision"] for i in range(2)]
    manager.sim = SimpleNamespace(
        sim=SimpleNamespace(get_physics_step_count=lambda: 24, get_physics_dt=lambda: 1 / 120)
    )
    manager.update = lambda: None
    manager.synchronize_contact_objects = lambda: None
    rgb, depth, delta = optical_buffers()
    environments = {}
    for env_id in range(2):
        actual_pose = wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
        data = SimpleNamespace(
            tactile_normal_force=torch.tensor([[0.0, 0.2 * env_id]]),
            tactile_shear_force=torch.zeros(1, 2, 2),
            penetration_depth=torch.tensor([[0.0, 0.0001 * env_id]]),
            tactile_points_pos_w=torch.ones(1, 2, 3),
            tactile_points_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]),
            tactile_rgb_image=rgb.unsqueeze(0),
            tactile_depth_image=depth.unsqueeze(0),
        )
        sensor = SimpleNamespace(
            data=data,
            _nominal_tactile={"distance_to_image_plane": (depth + delta).unsqueeze(0)},
            _elastomer_body_view=SimpleNamespace(get_transforms=lambda pose=actual_pose: pose),
            contact_query_backend="physx_native_sdf",
            contact_query_metadata={},
        )
        environments[env_id] = [sensor]
    manager.sensors = {"left": {"finger_0": environments}}
    sample = manager.get_observations()["left"]["finger_0"]
    assert sample["contact_active"].tolist() == [False, True]
    assert sample["contact_active"].dtype == torch.bool
    assert torch.count_nonzero(sample["rgb"][0]) == 0
    assert torch.count_nonzero(sample["depth"][0]) == 0
    assert torch.count_nonzero(sample["optical_depth_delta"][0]) == 0
    assert torch.equal(sample["rgb"][1], rgb)
    assert torch.equal(sample["raw_optical_rgb"][0], rgb)
    assert torch.equal(sample["raw_optical_depth"][0], depth)
    assert torch.count_nonzero(sample["raw_optical_depth_delta"][0]) == 3
    assert sample["normal_force"][1, 1] == 0.2
    assert sample["points_pos_w"][0].sum() == 6  # Geometry state is not a contact signal.
    assert sample["timestamp_ns"] == 200000000
    assert sample["physics_frame"] == 24
    assert sample["provenance"][0]["contact_conditioned_observations"] is True
