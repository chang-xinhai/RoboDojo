"""Geometry/material migration checks independent of a running Kit instance."""

import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from utils.physx_callback import invoke_physx_callback
from utils.surface_deformable import migrate_known_garment_physics, surface_material_parameters, triangulate_faces


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
        {"particle_material": {"particle_adhesion_scale": 0.2}},
        {"particle_material": {"particle_friction_scale": 0.6}},
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


@pytest.mark.parametrize("asset_id", ["00001", "00004", "00006", "00007", "00009", "00011"])
def test_bundled_garments_have_explicit_non_equivalent_migration(asset_id):
    root = Path(__file__).resolve().parents[1]
    folder = root / "Assets/Object/RoboDojo/Garment/Top_Long" / asset_id
    metadata_path = folder / "metadata.json"
    # Config checks remain runnable before the large asset bundle is downloaded.
    metadata = json.loads(
        (
            metadata_path if metadata_path.exists() else root / "tests/fixtures/garment_top_long_metadata.json"
        ).read_text()
    )
    original = deepcopy(metadata["physics"])
    with pytest.warns(UserWarning, match="physical models differ"):
        physics, migration = migrate_known_garment_physics(metadata["physics"], folder / "object.usdz")
    assert metadata["physics"] == original
    assert migration["adhesion_removed"]["legacy_adhesion"] == 0.1
    assert migration["adhesion_removed"]["equivalent"] is False
    assert migration["benchmark_equivalent"] is False
    assert migration["particle_contact_scaling_removed"]["particle_friction_scale"] == 0.6
    with pytest.warns(UserWarning, match="do not map exactly"):
        params = surface_material_parameters(physics, metadata["geometry"]["vertices"])
    assert params["gravity_scale"] == 2.0
    assert params["dynamic_friction"] == 0.5
    assert params["mass"] == 0.01 * metadata["geometry"]["vertices"]


def test_unknown_adhesion_and_modified_bundled_adhesion_remain_strict():
    for asset, adhesion in [("/Unknown/object.usdz", 0.1), ("RoboDojo/Garment/Top_Long/00001/object.usdz", 0.2)]:
        with pytest.warns(UserWarning) if adhesion == 0.2 else nullcontext():
            physics, _ = migrate_known_garment_physics({"particle_material": {"adhesion": adhesion}}, asset)
        with pytest.raises(NotImplementedError, match="adhesion"):
            surface_material_parameters(physics, 10)


def test_explicit_surface_gravity_calibration_is_validated():
    assert surface_material_parameters({"surface_material": {"gravity_scale": 0.5}}, 10)["gravity_scale"] == 0.5
    for scale in (-1, float("nan")):
        with pytest.raises(ValueError, match="gravity_scale"):
            surface_material_parameters({"surface_material": {"gravity_scale": scale}}, 10)


@pytest.mark.parametrize("scale,direction", [(2.0, [0, 0, -1]), (0.5, [0, -2, 0])])
def test_additional_gravity_uses_scene_axis_and_preserves_nodal_velocity(scale, direction):
    torch = pytest.importorskip("torch")
    source = Path(__file__).resolve().parents[1] / "env/scene_manager/objects/garment.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GarmentObject")
    function = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_gravity_velocity_delta"
    )
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    velocities = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    stored = []
    garment = SimpleNamespace(
        surface_parameters={"gravity_scale": scale},
        _gravity_scene=SimpleNamespace(
            GetGravityDirectionAttr=lambda: SimpleNamespace(Get=lambda: direction),
            GetGravityMagnitudeAttr=lambda: SimpleNamespace(Get=lambda: 9.81),
        ),
        _surface_view=SimpleNamespace(
            get_simulation_nodal_velocities=lambda: velocities,
            set_simulation_nodal_velocities=lambda values, indices: stored.append((values, indices)),
        ),
    )
    namespace[function.name](garment, 0.1)
    # Reference is Newtonian acceleration over the dt, including arbitrary gravity
    # axes. Existing velocity components must be retained by the real tensor write.
    expected_delta = 9.81 * 0.1 * (scale - 1) * np.asarray(direction) / np.linalg.norm(direction)
    np.testing.assert_allclose(stored[0][0].numpy(), velocities.numpy() + expected_delta, atol=1e-7)
    assert stored[0][1].dtype == torch.int32


@pytest.mark.parametrize("lab3_loaded", [False, True])
def test_native_physics_callback_errors_are_preserved_for_the_step(monkeypatch, lab3_loaded):
    stored = []
    module_name = "isaaclab_physx.physics.physx_manager"
    if lab3_loaded:
        monkeypatch.setitem(
            sys.modules,
            module_name,
            SimpleNamespace(PhysxManager=SimpleNamespace(store_callback_exception=stored.append)),
        )
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    failure = RuntimeError("Native surface view was invalidated")

    def failing_native_call():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        invoke_physx_callback(failing_native_call)
    assert caught.value is failure
    assert stored == ([failure] if lab3_loaded else [])


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


def test_fluid_sampler_rejects_empty_coarse_grid_in_a_real_volume():
    vertices = np.concatenate((np.eye(3), -np.eye(3)))
    with pytest.raises(ValueError, match="no interior particles"):
        fluid_sampler()(vertices, 3.0)


def scene_manager_method(name):
    import __future__

    source = Path(__file__).resolve().parents[1] / "env/scene_manager/scene_manager.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SceneManager")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"PRIMITIVE_MAP": {"Plane": object()}, "torch": pytest.importorskip("torch")}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("kind", ["garment", "fluid"])
def test_cpu_physics_request_fails_instead_of_disappearing(kind):
    manager = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    with pytest.raises(NotImplementedError, match="CUDA PhysX"):
        scene_manager_method("create_scene_object")(
            manager, 0, "Plane", "/World/cloth", {"physics": {"type": kind}}, (0, 0, 1), (1, 0, 0, 0), (1, 1, 1)
        )


def test_cuda_dynamic_contact_topology_exists_before_sensor_initialization():
    spawned = []
    types = ["rigid", "dynamic", "articulation", "garment", "fluid", "geometry"]
    manager = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        num_envs=1,
        config={},
        spawnable_object_types=types,
        layout_manager=SimpleNamespace(load_saved_layout=lambda _i: {}),
        _setup_background=lambda: None,
        _set_env_seed=lambda _i: None,
        spawn_scene_objects=lambda **kwargs: spawned.append(kwargs),
    )
    scene_manager_method("initialize")(manager, SimpleNamespace(scene=SimpleNamespace(env_origins=[0])))
    assert set(types) - set(spawned[0]["exclude_types"]) == {"rigid", "dynamic", "articulation"}


def test_cuda_reload_preserves_dynamic_contacts():
    excluded = []
    manager = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        setup_scene=True,
        reload_lights=lambda _i: None,
        reload_ground=lambda _i: None,
        reload_room=lambda _i: None,
        reload_table=lambda _i: None,
        relocate_stale_objects=lambda *_args: None,
        clear_scene_objects=lambda _i, exclude_types: excluded.append(exclude_types),
        spawn_scene_objects=lambda _i, exclude_types: excluded.append(exclude_types),
    )
    scene_manager_method("reload_env_scene")(manager, 0)
    assert all(set(types) == {"rigid", "dynamic", "articulation"} for types in excluded)


def test_new_deformables_are_consumed_by_physx_before_tensor_view_creation():
    calls = []
    manager = SimpleNamespace(
        _background=None,
        device="cpu",
        num_envs=0,
        _tables=[],
        setup_scene=True,
        _set_env_seed=lambda _i: None,
        reload_env_scene=lambda _i: None,
        pending_initialization=[SimpleNamespace(initialize=lambda: calls.append("initialize"))],
        sim=SimpleNamespace(sim_step=lambda **_kwargs: calls.append("physics_step")),
    )
    scene_manager_method("reload_envs")(manager, [])
    assert calls == ["physics_step", "initialize", "physics_step"]
    assert manager.pending_initialization == []
