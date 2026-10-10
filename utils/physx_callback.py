"""Surface native callback errors through the running Lab 3 physics manager."""

import sys


def invoke_physx_callback(callback, *args):
    """Retain the original error when Carb catches an event callback exception.

    Lab 3 checks its stored callback error after event dispatch. Standalone
    Isaac Sim callers still receive/log the original exception. Checking loaded
    modules avoids importing or creating a second physics lifecycle.
    """
    try:
        return callback(*args)
    except Exception as error:
        manager_module = sys.modules.get("isaaclab_physx.physics.physx_manager")
        if manager_module is not None:
            manager_module.PhysxManager.store_callback_exception(error)
        raise
