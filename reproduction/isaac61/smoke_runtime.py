"""Validate the real CUDA / PhysX / RTX stack, independent of task assets."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from utils.isaaclab_compat import add_app_launcher_args, as_torch

parser = argparse.ArgumentParser()
add_app_launcher_args(parser, AppLauncher)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app
output = Path(__file__).resolve().parent / "runtime-smoke.json"
output.unlink(missing_ok=True)
success = False
try:
    from isaaclab.sensors import Camera, CameraCfg
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    import numpy as np
    import torch

    assert metadata.version("isaacsim") == "6.1.0.0"
    assert torch.cuda.is_available()
    assert torch.ones(8, device="cuda").sum().item() == 8
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0", physics=PhysxCfg()))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=1500)
    light.func("/World/Light", light)
    cube = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    )
    cube.func("/World/Cube", cube, translation=(0, 0, 0.5))
    cfg = CameraCfg(
        prim_path="/World/Camera",
        width=128,
        height=128,
        data_types=["rgb", "depth/distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 10.0)),
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    camera = Camera(cfg)
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[1.0, 0.0, 1.0]], device="cuda"), torch.tensor([[0.0, 0.0, 0.2]], device="cuda")
    )
    for _ in range(20):
        sim.step()
        camera.update(sim.get_physics_dt(), force_recompute=True)
    rgb = as_torch(camera.data.output["rgb"])
    depth = as_torch(camera.data.output["depth/distance_to_image_plane"])
    finite = torch.isfinite(depth) & (depth > 0)
    assert tuple(rgb.shape[:3]) == (1, 128, 128), rgb.shape
    assert finite.any().item(), "No valid RTX depth"
    result = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "packages": {name: metadata.version(name) for name in ("isaacsim", "isaaclab", "torch", "numpy", "warp-lang")},
        "isaaclab_release": "v3.0.0-EA",
        "isaaclab_source_commit": "ae37b028ea415c91ea2bc32609efcd759ed2b974",
        "physics": "physx",
        "renderer": "isaac_rtx",
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "valid_depth_pixels": int(finite.sum().item()),
        "checks": ["CUDA tensor operation", "PhysX stepping", "RTX RGBD capture"],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output.with_suffix(".npz"), rgb=rgb.cpu().numpy(), depth_gt_m=depth.cpu().numpy())
    print(json.dumps(result), flush=True)
    success = True
except BaseException:
    import traceback

    traceback.print_exc()
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=0 if success else 1)
