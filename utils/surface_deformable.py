"""CPU geometry and configuration checks for Omni Physics 110 surface cloth."""

from copy import deepcopy
from pathlib import PurePosixPath
import warnings

import numpy as np

_ROBO_DOJO_TOP_LONG = {"00001", "00004", "00006", "00007", "00009", "00011"}


def weld_exact_surface_seams(points, triangles):
    """Join exactly coincident seam vertices and retain every source vertex ID.

    Returns compact simulation points/indices, an original-to-simulation map,
    and source representatives. No distance tolerance or geometric remeshing is
    used. Callers must opt in for an asset whose duplicate vertices are seams.
    """
    points = np.asarray(points, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Surface seam welding requires finite XYZ points")
    if triangles.size == 0 or np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("Surface seam welding requires valid triangle indices")
    # np.unique sorts coordinates. Preserve first-occurrence source order so
    # identity meshes keep their original node ordering.
    _, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    representatives = first[order]
    sorted_to_stable = np.empty(len(first), dtype=np.int64)
    sorted_to_stable[order] = np.arange(len(first))
    original_to_simulation = sorted_to_stable[inverse]
    indices = original_to_simulation[triangles]
    simulation_points = points[representatives].copy()
    a, b, c = np.moveaxis(simulation_points[indices], 1, 0)
    if np.any(np.linalg.norm(np.cross(b - a, c - a), axis=1) == 0):
        raise ValueError("Welding coincident surface seams would create degenerate triangles")
    if len(np.unique(np.sort(indices, axis=1), axis=0)) != len(indices):
        raise ValueError("Welding coincident surface seams would create duplicate triangles")
    edges = np.sort(np.concatenate([indices[:, [0, 1]], indices[:, [1, 2]], indices[:, [2, 0]]]), axis=1)
    if np.any(np.unique(edges, axis=0, return_counts=True)[1] > 2):
        raise ValueError("Welding coincident surface seams would create a nonmanifold edge")
    return simulation_points, indices.reshape(-1), original_to_simulation, representatives


def migrate_known_garment_physics(physics_cfg, usd_path):
    """Explicitly migrate the bundled Top_Long particle-cloth material.

    This is a named asset calibration, not a claim that dry friction replicates
    adhesion. Unknown assets and modified nonzero adhesion still fail the strict
    surface configuration checks. The caller records this provenance with data.
    """
    physics = deepcopy(physics_cfg)
    parts = PurePosixPath(str(usd_path or "")).parts
    known_asset = (
        len(parts) >= 5
        and parts[-5:-2] == ("RoboDojo", "Garment", "Top_Long")
        and parts[-2] in _ROBO_DOJO_TOP_LONG
        and parts[-1] == "object.usdz"
    )
    if not known_asset:
        return physics, {}
    particles = physics.setdefault("particle_material", {})
    material = physics.setdefault("surface_material", {})
    provenance = {
        "asset": "/".join(parts[-5:]),
        "model": "omniphysics_surface_xpbd_fem",
        "benchmark_equivalent": False,
    }
    adhesion = particles.get("adhesion")
    if adhesion is not None and float(adhesion) == 0.1:
        particles["adhesion"] = 0.0
        material.setdefault("dynamic_friction", float(particles.get("friction", 0.5)))
        provenance["adhesion_removed"] = {
            "legacy_adhesion": 0.1,
            "replacement": "dry_surface_contact",
            "dynamic_friction": material["dynamic_friction"],
            "equivalent": False,
        }
        for key in ("adhesion_offset_scale", "particle_adhesion_scale", "particle_friction_scale"):
            if key in particles:
                provenance.setdefault("particle_contact_scaling_removed", {})[key] = particles[key]
                particles[key] = 0.0
    gravity_scale = particles.get("gravity_scale", 1.0)
    if float(gravity_scale) == 2.0:
        particles["gravity_scale"] = 1.0
        material.setdefault("gravity_scale", 2.0)
        provenance["gravity_scale"] = {
            "legacy": 2.0,
            "surface": material["gravity_scale"],
            "implementation": "pre_step_nodal_velocity",
        }
    warnings.warn(
        f"Migrating bundled garment {provenance['asset']} to FEM surface contact; "
        "particle adhesion is removed when present and the physical models differ. "
        "See instance_config.physics_migration for recorded changes.",
        UserWarning,
        stacklevel=2,
    )
    return physics, provenance


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
    for key in (
        "lift",
        "drag",
        "adhesion",
        "cohesion",
        "viscosity",
        "surface_tension",
        "vorticity_confinement",
        "adhesion_offset_scale",
        "particle_adhesion_scale",
        "particle_friction_scale",
    ):
        value = particles.get(key)
        if value is not None and float(value) != 0:
            raise NotImplementedError(f"Particle cloth {key} has no surface deformable equivalent in Omni Physics 110")
    wind = system.get("wind")
    if wind is not None and np.any(np.asarray(wind) != 0):
        raise NotImplementedError("Surface deformables do not support particle cloth wind")
    legacy_gravity_scale = float(particles.get("gravity_scale", 1.0))
    if legacy_gravity_scale not in (0.0, 1.0):
        raise NotImplementedError("Legacy surface cloth gravity_scale must be 0 or 1; use an explicit FEM calibration")
    gravity_scale = float(material.get("gravity_scale", legacy_gravity_scale))
    if not np.isfinite(gravity_scale) or gravity_scale < 0:
        raise ValueError("Surface cloth gravity_scale must be finite and nonnegative")
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
        "gravity_scale": gravity_scale,
    }
    for key in ("mass", "youngs_modulus", "surface_thickness"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Surface cloth {key} must be finite and positive")
    for key in ("surface_bend_stiffness", "dynamic_friction", "elasticity_damping", "bend_damping"):
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"Surface cloth {key} must be finite and nonnegative")
    return result
