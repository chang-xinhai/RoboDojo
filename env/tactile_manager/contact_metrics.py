"""Measure contact contours from actual recorded TacSL depth/RGB arrays.

Metrics do not modify sensor output. Optical depth delta is the measured
camera-depth change, a geometry proxy rather than FEM gel deformation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def contact_frame_metrics(sample, baseline_rgb=None, depth_threshold_m=25e-6, minimum_component_pixels=64):
    """Measure raw positive depth-change support and closed contour topology.

    The threshold rejects small depth noise; it never changes saved depth or
    RGB. Hole counts use the largest measured component, excluding exterior
    background and tiny disconnected pixels. Values retain physical metres.
    """
    if depth_threshold_m <= 0 or minimum_component_pixels < 1:
        raise ValueError("Contact metric thresholds must be positive")
    delta = _array(sample["optical_depth_delta"])
    if delta.ndim == 3 and delta.shape[-1] == 1:
        delta = delta[..., 0]
    if delta.ndim != 2 or not np.isfinite(delta).all():
        raise ValueError("Contact metrics require finite HxW optical depth delta")
    mask = delta > depth_threshold_m
    labels, count = ndimage.label(mask)
    areas = np.bincount(labels.reshape(-1))
    areas[0] = 0
    component = labels == areas.argmax() if count else np.zeros_like(mask)
    largest_area = int(component.sum())
    holes = []
    bbox = None
    centroid = None
    if largest_area:
        rows, columns = np.nonzero(component)
        bbox = [int(rows.min()), int(columns.min()), int(rows.max()), int(columns.max())]
        centroid = [float(rows.mean()), float(columns.mean())]
        background_labels, _ = ndimage.label(~component)
        exterior = set(
            np.unique(
                np.concatenate(
                    (background_labels[0], background_labels[-1], background_labels[:, 0], background_labels[:, -1])
                )
            )
        )
        hole_areas = np.bincount(background_labels.reshape(-1))
        holes = [
            int(area)
            for index, area in enumerate(hole_areas)
            if index and index not in exterior and area >= minimum_component_pixels
        ]
    metrics = {
        "optical_depth_delta_units": "m",
        "depth_threshold_m": float(depth_threshold_m),
        "minimum_component_pixels": minimum_component_pixels,
        "contact_pixel_fraction": float(mask.mean()),
        "contact_component_count": int((areas >= minimum_component_pixels).sum()),
        "largest_component_pixel_fraction": largest_area / delta.size,
        "largest_component_bbox_row_col": bbox,
        "largest_component_centroid_row_col": centroid,
        "closed_contour_hole_count": len(holes),
        "closed_contour_hole_areas_pixels": holes,
        "optical_depth_delta_max_m": float(delta.max()),
        "optical_depth_delta_min_m": float(delta.min()),
        "optical_depth_delta_mean_abs_m": float(np.abs(delta).mean()),
        "optical_depth_delta_positive_p95_m": float(np.quantile(delta[mask], 0.95)) if mask.any() else 0.0,
        "physical_model_note": "Optical camera-depth change; not FEM deformation or a force measurement",
    }
    if baseline_rgb is not None:
        rgb = _array(sample["rgb"]).astype(np.float32)
        baseline = _array(baseline_rgb).astype(np.float32)
        if (
            rgb.shape != baseline.shape
            or rgb.shape[:2] != delta.shape
            or not np.isfinite(rgb).all()
            or not np.isfinite(baseline).all()
        ):
            raise ValueError("Contact RGB and baseline must have matching finite image shapes")
        metrics["rgb_mean_abs_change_0_255"] = float(np.abs(rgb - baseline).mean())
    normal = sample.get("normal_force")
    if normal is not None:
        force = _array(normal)
        if not np.isfinite(force).all():
            raise ValueError("Nonfinite actual TacSL force field")
        metrics.update(
            {
                "normal_force_max_N": float(force.max()),
                "normal_force_min_N": float(force.min()),
                "normal_force_sample_sum_N": float(force.sum()),
                "normal_force_positive_taxel_fraction": float((force > 0).mean()),
                "force_note": "Official penalty estimates; sample sum is not a calibrated net load",
            }
        )
    else:
        metrics["force_field_available"] = False
    if sample.get("pad_normal_force") is not None:
        pad_normal = _array(sample["pad_normal_force"])
        if not np.isfinite(pad_normal).all():
            raise ValueError("Nonfinite projected actual pad force")
        metrics["pad_normal_force_max_N"] = float(pad_normal.max())
        metrics["pad_normal_force_min_N"] = float(pad_normal.min())
        metrics["pad_normal_force_note"] = "signed outward -Y from actual body/taxel rotation; compression negative"
    return metrics
