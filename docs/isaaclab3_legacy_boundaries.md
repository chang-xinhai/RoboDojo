# Isaac Lab 3 and legacy Isaac Sim boundaries

RoboDojo's task object wrappers and RGBD camera manager use the real Isaac Sim
prim and camera APIs. Isaac Lab 3 launches a minimal experience that omits their
extensions. After `AppLauncher` starts Kit, call
`utils.isaaclab_compat.enable_legacy_isaacsim_extensions()` before importing
RoboDojo environments. It registers the installed `extsDeprecated` folder,
processes extension discovery synchronously, and enables `isaacsim.core.api`,
`isaacsim.sensors.camera`, and `isaacsim.replicator.behavior` with their declared
dependencies. Failed extension initialization raises an error.

Isaac Lab's PhysX manager owns simulation stepping, reset, lifecycle callbacks,
and the official Warp tensor view used by its robot assets and TacSL. Its
compatibility export does not implement every deprecated prim requirement, and
its Warp view cannot be consumed by legacy Torch helpers.

`utils.legacy_prim_adapter.install_legacy_prim_adapter()` runs after the PhysX
manager initializes and before task objects spawn. It binds a facade only to
the loaded legacy prim/camera modules and RoboDojo object/camera modules. The
facade creates a native Torch (or NumPy for CPU) tensor view for the same USD
stage and PhysX scene. It delegates lifecycle callbacks to the official manager
and maps legacy event enums to their actual native event values. Its view cache
refreshes when the stage or official view changes. The official manager and
Warp view are not modified. No additional `SimulationContext` is created.

The native camera time mapping and physics flush interfaces come from the
installed Isaac Sim extension. Unrecognized manager APIs raise an error instead
of inventing behavior. Public RoboDojo poses, cuRobo, configuration, and legacy
prim wrappers retain WXYZ quaternion order. Isaac Lab robot poses cross an
explicit XYZW-to-WXYZ boundary in `RobotManager`.

PBD fluid tasks require host particle readback. Before the first PhysX warmup,
their environment disables `/physics/suppressReadback` and enables particle USD
updates while retaining CUDA device and GPU dynamics. `FluidObject` flushes
actual solver transformations after physics steps. Surface deformable garment
nodal tensors require a different pipeline; mixed garment/fluid tasks raise an
explicit error and must use separate simulation contexts.

Run the canonical scene smoke before policy evaluation. The fluid integration
smoke additionally checks both X5 CUDA articulation buffers and joint motion,
native particle motion, exact particle reset, three RGBD views, and official
TacSL SDF output:

```bash
bash reproduction/isaac61/run.sh task-smoke --task stack_bowls
source reproduction/isaac61/environment.sh
bash reproduction/isaac61/run.sh python reproduction/isaac61/smoke_fluid_task.py --viz none --kit_args "$ROBODOJO_KIT_ARGS"
```

The fluid report records the installed runtime and pinned Isaac Lab source
commit. Its `--allow-isaac60-debug` option labels a 6.0.1 result as
`debug_passed`; it does not satisfy the Isaac Sim 6.1 acceptance requirement.
