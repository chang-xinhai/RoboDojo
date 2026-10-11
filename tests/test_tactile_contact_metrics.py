"""Test contact measurements without generating or changing sensor images."""

import numpy as np
import pytest

from env.tactile_manager.contact_metrics import contact_frame_metrics


def sample_for_mask(mask):
    delta = np.zeros((*mask.shape, 1), dtype=np.float32)
    delta[..., 0][mask] = 0.0005
    rgb = np.full((*mask.shape, 3), 80, dtype=np.uint8)
    rgb[mask] = 100
    return {"optical_depth_delta": delta, "rgb": rgb, "normal_force": np.array([0, 0.01, 0.02])}


def test_measured_annular_contact_has_closed_hole_and_retains_source_arrays():
    rows, columns = np.indices((80, 60))
    radius = np.hypot(rows - 40, columns - 30)
    mask = (radius < 22) & (radius > 10)
    sample = sample_for_mask(mask)
    before = {name: value.copy() for name, value in sample.items()}
    result = contact_frame_metrics(sample, baseline_rgb=np.full((80, 60, 3), 80, dtype=np.uint8))
    assert result["closed_contour_hole_count"] == 1
    assert result["contact_component_count"] == 1
    assert result["contact_pixel_fraction"] == pytest.approx(mask.mean())
    assert result["largest_component_centroid_row_col"] == pytest.approx([40, 30])
    assert result["optical_depth_delta_positive_p95_m"] == pytest.approx(0.0005)
    assert result["normal_force_max_N"] == pytest.approx(0.02)
    assert result["rgb_mean_abs_change_0_255"] == pytest.approx(20 * mask.mean())
    for name in sample:
        assert np.array_equal(sample[name], before[name])


def test_filled_contact_and_released_contact_are_not_reported_as_annular():
    rows, columns = np.indices((80, 60))
    mask = np.hypot(rows - 40, columns - 30) < 20
    pressed = contact_frame_metrics(sample_for_mask(mask))
    released = contact_frame_metrics(sample_for_mask(np.zeros_like(mask)))
    assert pressed["closed_contour_hole_count"] == 0
    assert pressed["largest_component_pixel_fraction"] > 0.2
    assert released["contact_pixel_fraction"] == 0
    assert released["largest_component_bbox_row_col"] is None
    assert released["contact_component_count"] == 0
    assert released["optical_depth_delta_positive_p95_m"] == 0


def test_small_depth_noise_and_negative_delta_do_not_become_positive_contact():
    sample = sample_for_mask(np.zeros((80, 60), dtype=bool))
    sample["optical_depth_delta"][20:40] = 10e-6
    sample["optical_depth_delta"][40:60] = -0.0002
    sample["normal_force"] = None
    result = contact_frame_metrics(sample)
    assert result["contact_pixel_fraction"] == 0
    assert result["optical_depth_delta_min_m"] == pytest.approx(-0.0002)
    assert result["force_field_available"] is False
    assert "normal_force_max_N" not in result


def test_metrics_reject_invalid_recorded_sensor_arrays():
    sample = sample_for_mask(np.zeros((80, 60), dtype=bool))
    sample["optical_depth_delta"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite HxW"):
        contact_frame_metrics(sample)
