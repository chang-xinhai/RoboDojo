"""Independent collider validity checks for the particle-readback TacSL path."""

import pytest

from env.tactile_manager.triangle_mesh_sdf import WarpTriangleMeshSdfView


@pytest.mark.parametrize(
    "points,face",
    [
        ([[0, 0, 0]] * 3, [0, 1, 2]),
        ([[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0]], [0, 1, 2]),
        ([[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0]], [0, 1, 1]),
    ],
)
def test_degenerate_collision_geometry_cannot_produce_finite_tactile_distance(points, face):
    with pytest.raises(ValueError, match="degenerate|area|triangle"):
        WarpTriangleMeshSdfView(points, [face], "cpu")
