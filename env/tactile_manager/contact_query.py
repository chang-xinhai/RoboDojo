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


def create_contact_query_view(physics_view, mesh, point_count, device, requested_backend="auto", *, body=None):
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
    approximation = UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get()
    schemas_before = list(mesh.GetAppliedSchemas())
    if requested_backend == "physx_native_sdf" and approximation != "sdf":
        raise ValueError("Native TacSL SDF requires an authored SDF collider; use auto to preserve physical cooking")
    effective_warp = requested_backend == "warp_triangle_mesh" or host_readback or (requested_backend == "auto" and approximation != "sdf")
    metadata = {
        "requested_query_backend": requested_backend,
        "physical_collision_approximation": UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(),
        "physical_collision_preserved": True,
        "physical_collision_preservation_basis": "No preparation mutation; source approximation and applied schemas retained",
        "physical_collision_schemas_before_query": schemas_before,
    }
    if body is not None:
        import numpy as np
        from .sdf_boundary import shape_affine_in_body

        affine = shape_affine_in_body(mesh, body)
        scales = np.linalg.norm(affine[:3, :3], axis=1)
        if np.ptp(scales) > 1e-5 * max(scales):
            # A local SDF + gradient does not encode exact Euclidean distance
            # after anisotropic scaling, especially at edges. Bake the actual
            # affine into query vertices instead; physical USD is untouched.
            view = WarpTriangleMeshSdfView.from_usd(mesh, device, point_transform=affine)
            view.native_unscaled_reference = (
                physics_view.create_sdf_shape_view(str(mesh.GetPath()), point_count)
                if not effective_warp else None
            )
            view.query_backend = "warp_triangle_mesh_affine_body_metric"
            metadata.update(
                query_coordinate_frame="rigid_body_metres", mesh_to_body_affine=affine.tolist(),
                physical_collision_preserved=True, triangle_count=view.triangle_count,
                discarded_degenerate_triangles=view.discarded_degenerate_triangles,
                query_geometry="authored collision triangles affine-baked into body metres; separate from PhysX cooked approximation",
                metric="Euclidean signed triangle-surface distance in metres; unit body-frame gradient",
                reason="native local SDF cannot supply exact anisotropically scaled distance",
            )
            return view, view.query_backend, metadata
    if effective_warp:
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
