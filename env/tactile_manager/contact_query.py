"""Distance-query choice independent of the task's physical collision model."""

from .triangle_mesh_sdf import WarpTriangleMeshSdfView


def validate_query_backend(backend):
    if backend not in ("auto", "physx_native_sdf", "warp_triangle_mesh"):
        raise ValueError(f"Unsupported TacSL contact_query_backend: {backend!r}")
    return backend


def find_warp_contact_components(stage, path):
    """Resolve one enabled original collider and its nearest rigid ancestor.

    Unlike the official native-SDF selector, this requires no SDF API and
    authors nothing. The caller provides an exact per-environment USD path.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(path)
    if not root.IsValid():
        raise ValueError(f"No actual Warp TacSL contact prim: {path}")
    meshes = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdGeom.Mesh)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]
    if len(meshes) != 1:
        raise ValueError(f"Warp TacSL requires one actual enabled collision mesh at {path}, found {len(meshes)}")
    mesh = meshes[0]
    body = mesh
    while body.IsValid() and str(body.GetPath()) != "/":
        if body.HasAPI(UsdPhysics.RigidBodyAPI):
            return mesh, body
        body = body.GetParent()
    raise ValueError(f"Warp TacSL collision mesh has no rigid-body ancestor: {mesh.GetPath()}")


def create_contact_query_view(physics_view, mesh, point_count, device, requested_backend="auto"):
    """Use original mesh triangles explicitly, or the existing native SDF path.

    This never authors USD collision APIs or changes cooking approximations.
    Both paths consume actual rigid-body poses through the caller's native view.
    """
    from pxr import UsdPhysics

    from utils.particle_readback_bridge import ReadbackTensorView

    requested_backend = validate_query_backend(requested_backend)
    host_readback = isinstance(physics_view, ReadbackTensorView)
    if host_readback and requested_backend == "physx_native_sdf":
        raise ValueError("PhysX host-readback views do not implement native SDF shape queries")
    metadata = {
        "requested_query_backend": requested_backend,
        "physical_collision_approximation": UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(),
        "physical_collision_preserved": requested_backend == "warp_triangle_mesh",
    }
    if requested_backend == "warp_triangle_mesh" or host_readback:
        view = WarpTriangleMeshSdfView.from_usd(mesh, device)
        metadata.update(
            triangle_count=view.triangle_count,
            discarded_degenerate_triangles=view.discarded_degenerate_triangles,
            query_geometry="original USD collision mesh triangles, not PhysX cooked convex shapes",
        )
        return view, view.query_backend, metadata
    view = physics_view.create_sdf_shape_view(str(mesh.GetPath()), point_count)
    metadata["query_geometry"] = "PhysX native SDF shape"
    return view, "physx_native_sdf", metadata
