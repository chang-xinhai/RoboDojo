"""Exercise Isaac Lab's actual ProxyArray camera boundary without Kit/GPU physics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from env.tactile_manager.camera_boundary import TacSLCameraBoundary, tactile_camera_depth


def runtime_arrays():
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    wp.init()
    path = (
        Path(__file__).resolve().parents[1] / "third_party/IsaacLab3/source/isaaclab/isaaclab/utils/warp/proxy_array.py"
    )
    spec = importlib.util.spec_from_file_location("official_tacsl_test_proxy_array", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tensor = torch.full((1, 4, 3, 1), 0.0186, dtype=torch.float32)
    proxy = module.ProxyArray(wp.from_torch(tensor))
    return torch, tensor, proxy


@pytest.mark.parametrize("key", ["depth/distance_to_image_plane", "distance_to_image_plane", "depth"])
def test_camera_depth_resolves_actual_official_proxy(key):
    torch, tensor, proxy = runtime_arrays()
    assert torch.equal(tactile_camera_depth({key: proxy}), tensor)


def test_missing_camera_depth_is_explicit():
    with pytest.raises(RuntimeError, match="did not produce image-plane depth"):
        tactile_camera_depth({"rgb": object()})


def test_baseline_owns_copy_and_depth_delta_reaches_renderer():
    torch, tensor, proxy = runtime_arrays()
    calls = []
    rendered = []
    camera = SimpleNamespace(
        data=SimpleNamespace(output={"depth/distance_to_image_plane": proxy}),
        update=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    boundary = TacSLCameraBoundary()
    boundary.cfg = SimpleNamespace(enable_camera_tactile=True)
    boundary._camera_sensor = camera
    boundary._sim_physics_dt = 1 / 120
    boundary._data = SimpleNamespace(
        tactile_depth_image=torch.zeros_like(tensor),
        tactile_rgb_image=torch.zeros((1, 4, 3, 3), dtype=torch.uint8),
    )

    def render(depth_delta):
        rendered.append(depth_delta.clone())
        return torch.full((1, 4, 3, 3), 42, dtype=torch.uint8)

    boundary._tactile_rgb_render = SimpleNamespace(render=render)
    baseline = boundary.get_initial_render()["distance_to_image_plane"].clone()
    tensor -= 0.001
    boundary._update_camera_tactile(slice(None))
    assert torch.equal(boundary._nominal_tactile["distance_to_image_plane"], baseline)
    assert torch.equal(boundary._data.tactile_depth_image, tensor)
    assert torch.allclose(rendered[0], torch.full((1, 4, 3), 0.001), atol=1e-7)
    assert torch.all(boundary._data.tactile_rgb_image == 42)
    assert all(kwargs["force_recompute"] for _, kwargs in calls)
