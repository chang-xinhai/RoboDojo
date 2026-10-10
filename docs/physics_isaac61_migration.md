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

The following old behaviors have no direct replacement and raise explicit
`NotImplementedError`: nonzero wind/lift/drag; cloth PBD adhesion, cohesion,
viscosity, vorticity and surface tension; fractional gravity scaling; shared
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

Run CPU geometry/configuration tests with:

```sh
.venv-isaac61/bin/python -m pytest tests/test_surface_deformable.py -q
```

Run actual GPU motion/reset checks with:

```sh
.venv-isaac61/bin/python scripts/check_scene_physics_live.py \
  --output outputs/validation/scene_physics
```

The live check requires GPU dynamics. It checks actual cloth motion and reset,
PBD fluid motion and reset, exports the stage and records the measured nodal
positions. A passing CPU test alone does not establish runtime physics validity.

Verified against the installed `omni.physx-110.1.13` implementation of
`deformableUtils.set_physics_surface_deformable_body`,
`deformableUtils.add_surface_deformable_material`, `particleUtils`, and official
Isaac Lab 3 EA `isaaclab_physx` deformable tensor-view code.

Official references:

- [Omni Physics 110.1 deformable migration guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_migration.html)
- [Omni Physics 110.1 deformable bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_bodies.html)
