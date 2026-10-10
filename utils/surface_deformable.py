"""CPU geometry and configuration checks for Omni Physics 110 surface cloth."""

import warnings

import numpy as np


def triangulate_faces(counts, indices, num_points):
    """Triangulate triangle/quad panels, preserving face/corner remapping.

    Garment panels must be convex; arbitrary polygons require an asset conversion.
    Corner and face maps also preserve UVs, face normals and material subsets.
    """
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if counts.ndim != 1 or indices.ndim != 1 or counts.size == 0:
        raise ValueError("Surface garment requires a nonempty mesh topology")
    if np.any((counts != 3) & (counts != 4)):
        raise ValueError("Surface garment accepts triangles and convex quad panels only")
    if counts.sum() != indices.size or np.any(indices < 0) or np.any(indices >= num_points):
        raise ValueError("Invalid garment face vertex indices")
    triangles, corner_map, face_map = [], [], []
    offset = 0
    for face, count in enumerate(counts):
        for triangle in ((0, 1, 2),) if count == 3 else ((0, 1, 2), (0, 2, 3)):
            corners = [offset + i for i in triangle]
            vertices = indices[corners]
            if len(set(vertices)) != 3:
                raise ValueError("Garment mesh contains a degenerate triangle")
            triangles.extend(vertices)
            corner_map.extend(corners)
            face_map.append(face)
        offset += count
    return np.asarray(triangles), np.asarray(corner_map), np.asarray(face_map)


def surface_material_parameters(physics_cfg, num_points):
    """Resolve FEM parameters and reject removed physical features explicitly."""
    legacy = physics_cfg.get("garment_config", {})
    material = physics_cfg.get("surface_material", {})
    particles = physics_cfg.get("particle_material", {})
    system = physics_cfg.get("particle_system", {})
    if physics_cfg.get("interaction_with_fluid", False):
        raise NotImplementedError("Omni Physics 110 surface cloth cannot share a PBD fluid particle system")
    for key in ("lift", "drag", "adhesion", "cohesion", "viscosity", "surface_tension", "vorticity_confinement"):
        value = particles.get(key)
        if value is not None and float(value) != 0:
            raise NotImplementedError(f"Particle cloth {key} has no surface deformable equivalent in Omni Physics 110")
    wind = system.get("wind")
    if wind is not None and np.any(np.asarray(wind) != 0):
        raise NotImplementedError("Surface deformables do not support particle cloth wind")
    gravity_scale = float(particles.get("gravity_scale", 1.0))
    if gravity_scale not in (0.0, 1.0):
        raise NotImplementedError("Surface cloth gravity_scale must be 0 or 1")
    if any(key in legacy for key in ("stretch_stiffness", "shear_stiffness", "bend_stiffness", "spring_damping")):
        warnings.warn(
            "Particle spring coefficients do not map exactly to XPBD FEM cloth. "
            "Configure physics.surface_material to calibrate Young's modulus, Poisson ratio, "
            "thickness, bend stiffness and damping. Legacy spring coefficients are not reused.",
            UserWarning,
            stacklevel=2,
        )
    poisson = float(material.get("poissons_ratio", 0.3))
    if not 0 <= poisson < 0.5:
        raise ValueError("Surface cloth poissons_ratio must be in [0, 0.5)")
    result = {
        "mass": float(legacy.get("mass", float(legacy.get("particle_mass", 1e-2)) * num_points)),
        "youngs_modulus": float(material.get("youngs_modulus", 1e5)),
        "poissons_ratio": poisson,
        "surface_thickness": float(material.get("surface_thickness", 1e-3)),
        "surface_bend_stiffness": float(
            material.get(
                "surface_bend_stiffness",
                float(material.get("youngs_modulus", 1e5)) / (12.0 * (1.0 - poisson**2)),
            )
        ),
        "dynamic_friction": float(material.get("dynamic_friction", particles.get("friction", 0.5))),
        "elasticity_damping": float(material.get("elasticity_damping", 0.005)),
        "bend_damping": float(material.get("bend_damping", 0.005)),
        "disable_gravity": gravity_scale == 0,
    }
    for key in ("mass", "youngs_modulus", "surface_thickness"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Surface cloth {key} must be finite and positive")
    for key in ("surface_bend_stiffness", "dynamic_friction", "elasticity_damping", "bend_damping"):
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"Surface cloth {key} must be finite and nonnegative")
    return result
