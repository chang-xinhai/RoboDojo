# Physics migration to Isaac Sim 6.1

`GarmentObject` now simulates a real Omni Physics 110 **surface deformable**.
Isaac Sim 6.1 removed particle cloth and its tensor views. The replacement uses
XPBD FEM corotational elasticity, not the previous particle spring model. The
same garment USD mesh is the visual, simulation and collision surface. GPU nodal
positions feed reward queries and resets; resets restore positions at the current
layout pose and clear all nodal velocities.

Triangle and convex quad panel meshes are supported. Quad conversion retains
face-varying UVs, indexed primvars, normals and material subset membership. A
single editable mesh is required per garment; other mesh topologies must first be
converted to suitable triangulated assets. Primitive dimensions are retained.

Configure calibrated FEM properties explicitly:

```yaml
physics:
  type: garment
  garment_config:
    mass: 0.10
    self_collision: true
  surface_material:
    youngs_modulus: 100000.0
    poissons_ratio: 0.3
    surface_thickness: 0.001
    surface_bend_stiffness: 9157.509
    dynamic_friction: 0.5
    elasticity_damping: 0.005
    bend_damping: 0.005
    gravity_scale: 1.0
  particle_system: # compatible legacy collision/solver names
    contact_offset: 0.01
    rest_offset: 0.0
    enable_ccd: true
    solver_position_iteration_count: 16
```

Without an explicit mass, total mass remains `particle_mass * vertex_count`.
Legacy spring stiffness/damping settings issue a visible warning: spring constants
are not FEM Young's modulus or bending coefficients. Defaults use a thickness of
1 mm, Young's modulus 100 kPa and Poisson ratio 0.3. The default bending coefficient
is `E / (12 * (1 - nu**2))`; the solver incorporates thickness cubed separately.
These defaults provide an executable replacement and require task-specific
calibration before comparing trajectories against old particle-cloth results.

The six bundled `RoboDojo/Garment/Top_Long/{00001,00004,00006,00007,00009,00011}`
assets have an explicit runtime migration for their original metadata. The old
adhesion of 0.1 is removed; the new surface uses dry contact with the asset's
friction of 0.5. Dry friction is not adhesive contact. The migration is recorded
in `instance_config.physics_migration` and the mesh's
`robodojo:physicsMigration` custom data, with `adhesion_removed` and
`benchmark_equivalent: false`. Unknown assets and modified adhesion values still
raise the strict unsupported-feature error.

Those bundled assets' gravity scale of 2 is implemented using actual PhysX
nodal velocity writes before every physics step: the callback adds
`(scale - 1) * scene_gravity * dt` and PhysX integrates normal scene gravity. This
retains the real surface solver, elastic constraints and contact response.
Explicit new `surface_material.gravity_scale` values support finite nonnegative
scaling; arbitrary nonzero legacy scaling is not implicitly accepted.

The following old behaviors have no direct replacement and raise explicit
`NotImplementedError`: nonzero wind/lift/drag; cloth PBD adhesion, cohesion,
viscosity, vorticity and surface tension; uncalibrated legacy gravity scaling; shared
cloth/fluid particle-system interaction. Rigid colliders physically contact the
new cloth. TacSL can render the resulting cloth geometry; official TacSL SDF
force fields support rigid mesh objects, so cloth-only scenes must mark those
fields unavailable. No cloth force values are synthesized.

Non-cloth PBD particle fluids remain supported in Omni Physics 110. `FluidObject`
retains that physics implementation, removes the deleted helper's `lift`/`drag`
keywords, applies visual scale once, includes environment origin, and clears
particle velocities on reset. Its convex volume sampler rejects invalid spacing,
non-volume source meshes and excessive particle grids instead of generating empty
or nondeterministic volumes.

PBD particle state has no supported GPU tensor readback. Fluid tasks therefore
keep GPU dynamics and the CUDA simulation device, but enable host readback
(`/physics/suppressReadback = false`) before physics initialization. Real solver
positions are published to USD after steps for rendering and rewards. Surface
deformable tensor reads require the GPU pipeline in Omni Physics 110, so a task
requesting both fluid and garment is explicitly rejected. These modes must be
validated separately; moving a fluid's rendered USD points manually would not
establish fluid physics support.

Run CPU geometry/configuration tests with:

```sh
.venv-isaac61/bin/python -m pytest tests/test_surface_deformable.py tests/test_surface_reset_contract.py -q
```

Run actual GPU motion/reset checks with:

```sh
.venv-isaac61/bin/python scripts/check_scene_physics_live.py \
  --output reproduction/isaac61/outputs/scene-cloth
.venv-isaac61/bin/python scripts/check_scene_physics_live.py \
  --fluid-only --cpu-readback --output reproduction/isaac61/outputs/scene-fluid
.venv-isaac61/bin/python scripts/check_scene_physics_live.py \
  --gravity-scale 2 --output reproduction/isaac61/outputs/scene-gravity2
.venv-isaac61/bin/python scripts/check_scene_physics_live.py \
  --garment-asset Assets/Object/RoboDojo/Garment/Top_Long/00001/object.usdz \
  --output reproduction/isaac61/outputs/scene-garment-00001
```

The live checks require GPU dynamics. They check actual cloth motion, physical
contact with a rigid ground collider, nodal gravity acceleration, reset,
rendering changes and environment offsets, or PBD
fluid motion/reset. They export the stage and record the measured particle/nodal
positions. Reports identify the actual package and PhysX extension versions.
A passing CPU test alone does not establish runtime physics validity.

Verified against the installed `omni.physx-110.1.13` implementation of
`deformableUtils.set_physics_surface_deformable_body`,
`deformableUtils.add_surface_deformable_material`, `particleUtils`, and official
Isaac Lab 3 EA `isaaclab_physx` deformable tensor-view code.

Official references:

- [Omni Physics 110.1 deformable migration guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_migration.html)
- [Omni Physics 110.1 deformable bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_bodies.html)
