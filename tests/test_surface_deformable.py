"""Geometry/material migration checks independent of a running Kit instance."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.surface_deformable import surface_material_parameters, triangulate_faces


def test_quad_triangulation_preserves_corners_and_material_faces():
    triangles, corners, faces = triangulate_faces([4, 3], [0, 1, 2, 3, 1, 4, 2], 5)
    np.testing.assert_array_equal(triangles, [0, 1, 2, 0, 2, 3, 1, 4, 2])
    np.testing.assert_array_equal(corners, [0, 1, 2, 0, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(faces, [0, 0, 1])


@pytest.mark.parametrize("counts,indices", [([2], [0, 1]), ([3], [0, 1, 9]), ([3], [0, 0, 1]), ([4], [0, 1, 2])])
def test_invalid_topology_rejected(counts, indices):
    with pytest.raises(ValueError):
        triangulate_faces(counts, indices, 4)


def test_mass_preserved_and_bending_stiffness_derived():
    params = surface_material_parameters({"garment_config": {"particle_mass": 0.002}}, 100)
    assert params["mass"] == 0.2
    assert params["surface_bend_stiffness"] > 0
    assert params["surface_thickness"] == 0.001


def test_spring_migration_warns_and_explicit_material_wins():
    with pytest.warns(UserWarning, match="do not map exactly"):
        params = surface_material_parameters(
            {"garment_config": {"stretch_stiffness": 1e8}, "surface_material": {"youngs_modulus": 1234}}, 10
        )
    assert params["youngs_modulus"] == 1234


@pytest.mark.parametrize(
    "cfg",
    [
        {"particle_system": {"wind": [0, 0, 1]}},
        {"particle_material": {"drag": 1}},
        {"particle_material": {"adhesion": 0.1}},
        {"interaction_with_fluid": True},
        {"particle_material": {"gravity_scale": 0.5}},
    ],
)
def test_removed_features_fail_explicitly(cfg):
    with pytest.raises(NotImplementedError):
        surface_material_parameters(cfg, 10)


@pytest.mark.parametrize(
    "cfg",
    [
        {"surface_material": {"youngs_modulus": -1}},
        {"surface_material": {"surface_thickness": 0}},
        {"surface_material": {"poissons_ratio": 0.5}},
        {"garment_config": {"mass": float("nan")}},
    ],
)
def test_invalid_physical_parameters_fail(cfg):
    with pytest.raises(ValueError):
        surface_material_parameters(cfg, 10)


def fluid_sampler():
    # Run the actual sampler without importing Isaac-only object wrappers.
    source = Path(__file__).resolve().parents[1] / "env/scene_manager/objects/fluid.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_particles_in_convex_mesh"
    )
    Delaunay = pytest.importorskip("scipy.spatial").Delaunay
    namespace = {"np": np, "Delaunay": Delaunay, "Gf": SimpleNamespace(Vec3f=lambda *x: x)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def test_fluid_sampler_interior_no_input_mutation():
    vertices = np.asarray([[x, y, z] for x in (0, 0.1) for y in (0, 0.1) for z in (0, 0.1)])
    original = vertices.copy()
    positions, velocities = fluid_sampler()(vertices, 0.025)
    assert len(positions) == 125
    np.testing.assert_array_equal(vertices, original)
    assert np.count_nonzero(velocities) == 0


@pytest.mark.parametrize("spacing", [0, -1, float("nan")])
def test_fluid_sampler_rejects_bad_spacing(spacing):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        fluid_sampler()(vertices, spacing)


def test_fluid_sampler_rejects_translated_planar_vertices():
    vertices = np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    with pytest.raises(ValueError, match="nonzero volume"):
        fluid_sampler()(vertices, 0.1)
