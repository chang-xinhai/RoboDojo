"""PhysX PBD fluid using the current public particle schemas and USD readback.

PhysX exposes no particle-set tensor view in this SDK. Particle USD writeback is
explicitly enabled so particle reward data remain current with Fabric rendering.
"""
import numpy as np
from omni.physx.scripts import particleUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, Vt
from scipy.spatial import Delaunay

from .scene_object import SceneObject, bind_material, mdl_material, world_matrix


class FluidObject(SceneObject):
    def __init__(self, prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale=(1, 1, 1)):
        super().__init__(prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale)
        self.physics_cfg = inst_config.get('physics', {})
        config = self.physics_cfg.get('particle_system', {})
        material_cfg = self.physics_cfg.get('particle_material', {})
        for removed in ('drag', 'lift'):
            if material_cfg.get(removed) is not None:
                raise ValueError(f'PhysX 6.2 removed particle material {removed}; nonempty values need a model decision')
        env_name = next(part for part in prim_path.split('/') if part.startswith('env_'))
        # World-space particle coordinates have an identity ancestor. The source
        # mesh remains under the environment root for metadata/geometry only.
        particle_root = f'/World/ParticleSets/{env_name}/{self.category_name}/{self.instance_name}'
        UsdGeom.Xform.Define(self.stage, particle_root)
        self.particle_system_path = particle_root + '/system'
        system = PhysxSchema.PhysxParticleSystem.Define(self.stage, self.particle_system_path)
        self.particle_system = system
        attributes = {
            'ParticleContactOffset': ('particle_contact_offset', 0.025),
            'ContactOffset': ('contact_offset', 0.025),
            'RestOffset': ('rest_offset', 0.0225),
            'FluidRestOffset': ('fluid_rest_offset', 0.0135),
            'SolidRestOffset': ('solid_rest_offset', 0.0225),
            'MaxVelocity': ('max_velocity', 2.5),
        }
        for name, (key, default) in attributes.items():
            getattr(system, f'Create{name}Attr')().Set(float(config.get(key, default)))
        scenes = [p.GetPath() for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes) != 1:
            raise ValueError(f'Expected one native physics scene, got {scenes}')
        system.CreateSimulationOwnerRel().SetTargets(scenes)
        for key, schema, default in [
            ('smoothing', PhysxSchema.PhysxParticleSmoothingAPI, False),
            ('anisotropy', PhysxSchema.PhysxParticleAnisotropyAPI, False),
            ('isosurface', PhysxSchema.PhysxParticleIsosurfaceAPI, True),
        ]:
            if config.get(key, default):
                schema.Apply(system.GetPrim())
        meshes = [p for p in Usd.PrimRange(self.prim) if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise ValueError(f'Fluid source has no mesh: {usd_path}')
        root_inverse = np.linalg.inv(world_matrix(self.prim))
        points = []
        for mesh in meshes:
            source = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
            transform = root_inverse @ world_matrix(mesh)
            points.append(source @ transform[:3, :3].T + transform[:3, 3])
            UsdGeom.Imageable(mesh).MakeInvisible()
        # Preserve the benchmark's authored cloud: visual scale participates in
        # both grid generation and final placement. Sampling before instance
        # scale preserves particle count and mass for historical saved layouts.
        visual_scale = np.asarray(inst_config.get('visual', {}).get('scale', [1, 1, 1]))
        local = np.concatenate(points) * visual_scale
        transform = world_matrix(self.prim)
        rest = float(config.get('fluid_rest_offset', 0.0135))
        positions, velocities = generate_particles_in_convex_mesh(local, 2.0 * rest)
        if not positions:
            raise ValueError(f'Fluid mesh produced no particles: {prim_path}')
        positions = (np.asarray(positions) * visual_scale) @ transform[:3, :3].T + transform[:3, 3]
        self.init_position = positions.astype(np.float32)
        positions = [Gf.Vec3f(*map(float, p)) for p in self.init_position]
        self.particle_point_instancer_path = Sdf.Path(particle_root + '/particles')
        particleUtils.add_physx_particleset_pointinstancer(
            stage=self.stage, path=self.particle_point_instancer_path,
            positions=Vt.Vec3fArray(positions), velocities=Vt.Vec3fArray(velocities),
            particle_system_path=self.particle_system_path, self_collision=True,
            fluid=True, particle_group=0, particle_mass=0.0005, density=0.0,
        )
        self.point_instancer = UsdGeom.PointInstancer.Get(self.stage, self.particle_point_instancer_path)
        prototype = UsdGeom.Sphere.Get(self.stage, self.particle_point_instancer_path.AppendChild('particlePrototype0'))
        prototype.CreateRadiusAttr().Set(rest)
        if config.get('isosurface', True):
            UsdGeom.Imageable(prototype).MakeInvisible()
        mdl_path = inst_config.get('visual', {}).get('visual_mdl_path', './Assets/Material/Fluid/Linen_Blue.mdl')
        material_path = mdl_material(particle_root + '/Looks/fluid', mdl_path)
        kwargs = {key: value for key, value in material_cfg.items() if key not in ('drag', 'lift')}
        particleUtils.add_pbd_particle_material(stage=self.stage, path=material_path, **kwargs)
        bind_material(system.GetPrim(), material_path)
        bind_material(system.GetPrim(), material_path, physics=True)
        bind_material(prototype.GetPrim(), material_path)
        system.GetPrim().CreateAttribute('primvars:doNotCastShadows', Sdf.ValueTypeNames.Bool).Set(True)

    def post_init(self):
        # SceneManager owns the task-scoped public USD-writeback settings.
        pass

    def get_particle_positions(self):
        """Return live PhysX particle positions in world metres (USD writeback)."""
        positions = np.asarray(self.point_instancer.GetPositionsAttr().Get(), dtype=np.float32)
        if positions.shape != self.init_position.shape:
            raise RuntimeError(f'Particle count changed for {self.prim_path}: {positions.shape}')
        return positions, None, None

    def set_particle_positions(self, positions):
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != self.init_position.shape:
            raise ValueError(f'Expected particle positions {self.init_position.shape}, got {positions.shape}')
        self.point_instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
        self.point_instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros_like(positions)))

    def apply_saved_pose(self):
        self.set_particle_positions(self.init_position)

    def relocate_offscreen(self):
        self.set_particle_positions(self.init_position + np.full(3, 100000, dtype=np.float32))


def generate_particles_in_convex_mesh(vertices: np.ndarray, sphere_diameter: float):
    """
    Generate particles within a convex mesh using Delaunay triangulation.

    Args:
        vertices: Vertices of the convex mesh
        sphere_diameter: Diameter of particles to generate

    Returns:
        List of particle positions and velocities (zero-initialized)
    """
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)

    if vertices.shape[0] < 4:
        print("Warning: Need at least 4 vertices for Delaunay triangulation. Returning empty.")
        return [], []

    try:
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)

        if np.linalg.matrix_rank(vertices) < 3:
            vertices += np.random.rand(*vertices.shape) * 1e-6

        hull = Delaunay(vertices)
    except Exception as e:
        print(f"Error during Delaunay triangulation: {e}. Vertices shape: {vertices.shape}. Returning empty.")
        return [], []

    epsilon = sphere_diameter * 0.01
    x_vals = np.arange(min_bound[0], max_bound[0] + epsilon, sphere_diameter)
    y_vals = np.arange(min_bound[1], max_bound[1] + epsilon, sphere_diameter)
    z_vals = np.arange(min_bound[2], max_bound[2] + epsilon, sphere_diameter)

    if x_vals.size == 0 or y_vals.size == 0 or z_vals.size == 0:
        print("Warning: Empty dimension range for particle grid. Returning empty.")
        return [], []

    samples = np.stack(np.meshgrid(x_vals, y_vals, z_vals, indexing="ij"), axis=-1).reshape(-1, 3)

    inside_mask = hull.find_simplex(samples, tol=1e-6) >= 0
    inside_points = samples[inside_mask]

    velocity = np.zeros_like(inside_points)

    positions_gf = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in inside_points]
    velocities_gf = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in velocity]

    return positions_gf, velocities_gf
