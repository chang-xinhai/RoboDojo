"""Explicit Isaac Lab 3 array and quaternion boundaries.

Isaac Lab 3 uses ProxyArray and XYZW quaternions. RoboDojo's YAML, cuRobo,
transforms3d, policy observations and Isaac Sim legacy prim wrappers use WXYZ.
Convert only at the Isaac Lab boundary; converting legacy prims again is wrong.
This module deliberately imports no Isaac/Kit modules and is usable before launch.
"""

import numpy as np


def add_app_launcher_args(parser, launcher_class):
    """Register 3.x CLI options and keep RoboDojo's existing launch scripts usable.

    Isaac Lab 3 still accepts the headless/enable_cameras keyword arguments,
    but no longer adds their CLI flags to downstream argument parsers.
    """
    launcher_class.add_app_launcher_args(parser)
    for option, help_text in (
        ("--headless", "Run Isaac Sim without a window."),
        ("--enable_cameras", "Enable offscreen camera capture."),
    ):
        if option not in parser._option_string_actions:
            parser.add_argument(option, action="store_true", help=help_text)


def as_torch(value):
    """Return the zero-copy torch view of a ProxyArray, or a tensor unchanged."""
    return getattr(value, "torch", value)


def enable_legacy_isaacsim_extensions(extension_manager=None):
    """Enable actual RoboDojo prim/camera wrappers after AppLauncher starts Kit.

    Isaac Lab 3 launches a minimal experience which omits these Isaac Sim
    extensions. Enable them before importing legacy RoboDojo managers. This
    does not create a second SimulationContext or change the physics backend.
    The optional manager supports checking the exact extension requirements
    without launching Kit.
    """
    if extension_manager is None:
        from importlib.metadata import distribution
        from pathlib import Path

        import isaacsim
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        # Sim 6 ships the legacy APIs in extsDeprecated; the minimal Lab 3
        # experience searches only the main extension folders by default.
        package_paths = list(isaacsim.__path__) + [distribution("isaacsim").locate_file("isaacsim")]
        for package_path in package_paths:
            legacy_path = Path(package_path) / "extsDeprecated"
            if legacy_path.is_dir():
                extension_manager.add_path(str(legacy_path))
        extension_manager.process_and_apply_all_changes()
    for name in ("isaacsim.core.api", "isaacsim.sensors.camera", "isaacsim.replicator.behavior"):
        if extension_manager.is_extension_enabled(name):
            continue
        extension_manager.set_extension_enabled_immediate(name, True)
        if not extension_manager.is_extension_enabled(name):
            raise RuntimeError(f"Isaac Sim extension failed to initialize: {name}")


def simulation_snapshot(context):
    """Return (simulation seconds, monotonic physics frame) without stepping.

    Isaac Lab 3 exposes get_physics_step_count/get_physics_dt. The configured
    Isaac Sim PhysX backend advances one dt per call, so their product is a
    deterministic acquisition clock, anchored to this context's creation.
    Legacy 2.x contexts retain public time properties.
    """
    if hasattr(context, "get_physics_step_count"):
        frame = int(context.get_physics_step_count())
        return frame * float(context.get_physics_dt()), frame
    return float(context.current_time), int(context.current_time_step_index)


def _reorder_quaternion(value, indices):
    value = as_torch(value)
    if isinstance(value, (tuple, list)):
        value = np.asarray(value)
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0 or shape[-1] != 4:
        raise ValueError(f"Expected quaternion (..., 4), received {shape}")
    return value[..., indices]


def wxyz_to_xyzw(value):
    return _reorder_quaternion(value, [1, 2, 3, 0])


def xyzw_to_wxyz(value):
    return _reorder_quaternion(value, [3, 0, 1, 2])


def pose_xyzw_to_wxyz(value):
    """Copy an Isaac Lab pose (..., 7) to RoboDojo's [XYZ, WXYZ] contract."""
    value = as_torch(value)
    if isinstance(value, (tuple, list)):
        value = np.asarray(value)
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0 or shape[-1] != 7:
        raise ValueError(f"Expected pose (..., 7), received {shape}")
    return value[..., [0, 1, 2, 6, 3, 4, 5]]


def set_joint_target(articulation, target, *, kind="position", joint_ids=None, env_ids=None):
    """Use Isaac Lab 3 actuator commands, retaining 2.x tensor compatibility."""
    command = getattr(getattr(articulation, "actuators", None), "target_command", None)
    if command is not None:
        getattr(command, f"set_{kind}_index")(value=as_torch(target), joint_ids=joint_ids, env_ids=env_ids)
    else:
        getattr(articulation, f"set_joint_{kind}_target")(as_torch(target), joint_ids=joint_ids, env_ids=env_ids)
