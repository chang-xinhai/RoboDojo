"""Explicit contact-conditioned observations, retaining original optical data."""

from __future__ import annotations

import torch


def condition_contact_observation(rgb, depth, optical_delta, penetration, normal, shear):
    """Return optical contact channels gated only by actual registered-body SDF.

    An inactive channel is an explicit zero sentinel, not a rendered black gel.
    Raw optical buffers remain independently available, including mechanical
    occlusion, camera residuals and the normal coloured unloaded gel texture.
    Forces are never repaired: nonzero force without penetration fails loudly.
    """
    if any(value is None for value in (penetration, normal, shear)):
        raise ValueError("Contact-conditioned TacSL requires actual available SDF force data")
    if not torch.isfinite(penetration).all() or bool((penetration < 0).any()):
        raise ValueError("Invalid actual TacSL penetration for contact conditioning")
    if not torch.isfinite(normal).all() or not torch.isfinite(shear).all():
        raise ValueError("Invalid actual TacSL force for contact conditioning")
    active = (penetration > 0).any()
    if not bool(active) and (bool((normal != 0).any()) or bool((shear != 0).any())):
        raise RuntimeError("Actual TacSL force is nonzero without registered-body SDF penetration; refusing to mask it")
    return {
        "rgb": rgb.clone() if bool(active) else torch.zeros_like(rgb),
        "depth": depth.clone() if bool(active) else torch.zeros_like(depth),
        "optical_depth_delta": optical_delta.clone() if bool(active) else torch.zeros_like(optical_delta),
        "raw_optical_rgb": rgb.clone(),
        "raw_optical_depth": depth.clone(),
        "raw_optical_depth_delta": optical_delta.clone(),
        "contact_active": active.clone(),
    }
