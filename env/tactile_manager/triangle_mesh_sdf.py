"""Real Warp triangle-mesh distance queries for PhysX particle host readback.

The native CPU PhysX tensor view has no SDF-shape implementation. This backend
queries the actual USD collision triangles, retaining TacSL's distance/gradient
contract. It is a geometry query, not a PhysX SDF or a fluid pressure sensor.
"""

import numpy as np
import torch
import warp as wp


@wp.kernel
def _signed_mesh_distance(mesh: wp.uint64, points: wp.array3d(dtype=float), output: wp.array3d(dtype=float)):
    batch, index = wp.tid()
    point = wp.vec3(points[batch, index, 0], points[batch, index, 1], points[batch, index, 2])
    query = wp.mesh_query_point_sign_winding_number(mesh, point, 1.0e10)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        delta = point - closest
        distance = wp.length(delta)
        gradient = wp.normalize(delta) * query.sign
        # Subtracting nearly equal float32 coordinates at a surface produces
        # an arbitrary normalized roundoff vector. Use the closest face normal
        # only within eight coordinate ULPs; retain the measured distance.
        roundoff = 8.0 * 1.1920928955078125e-7 * wp.max(wp.length(point), wp.length(closest))
        if distance <= wp.max(1.0e-12, roundoff):
            gradient = wp.mesh_eval_face_normal(mesh, query.face)
        output[batch, index, 0] = gradient[0]
        output[batch, index, 1] = gradient[1]
        output[batch, index, 2] = gradient[2]
        output[batch, index, 3] = distance * query.sign
    else:
        for component in range(4):
            output[batch, index, component] = wp.nan


class WarpTriangleMeshSdfView:
    """One immutable real collider mesh in its unscaled local coordinates."""

    count = 1
    query_backend = "warp_triangle_mesh_signed_distance"

    def __init__(self, vertices, triangles, device):
        wp.init()
        points = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(triangles, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("TacSL mesh query requires finite Nx3 vertices")
        if faces.ndim != 2 or faces.shape[1] != 3 or not faces.size:
            raise ValueError("TacSL mesh query requires nonempty triangle faces")
        if faces.min() < 0 or faces.max() >= len(points):
            raise ValueError("TacSL collision mesh contains invalid vertex indices")
        # A zero-area triangle has no physical contact surface or well-defined
        # normal. Remove only exactly degenerate faces, retaining all finite
        # nonzero faces rather than applying an arbitrary geometry tolerance.
        triangles = points[faces].astype(np.float64)
        area_vector = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        valid = np.any(area_vector != 0, axis=1)
        self.discarded_degenerate_triangles = int((~valid).sum())
        faces = faces[valid]
        self.triangle_count = len(faces)
        if not self.triangle_count:
            raise ValueError("TacSL collision mesh has no nondegenerate triangle surface")
        self.points = wp.array(points, dtype=wp.vec3, device=device)
        self.indices = wp.array(faces.reshape(-1), dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(points=self.points, indices=self.indices, support_winding_number=True)
        self.device = device

    @classmethod
    def from_usd(cls, prim, device, *, point_transform=None):
        from pxr import UsdGeom

        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        if not len(counts) or not np.isin(counts, (3, 4)).all():
            raise ValueError(f"TacSL readback mesh requires triangles or planar convex quads: {prim.GetPath()}")
        if mesh.GetHoleIndicesAttr().Get():
            raise ValueError(f"TacSL readback mesh holes require explicit collision triangles: {prim.GetPath()}")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        faces, offset = [], 0
        for count in counts:
            face = indices[offset:offset+count]
            offset += count
            if count == 4:
                p = points[face]
                normal = np.cross(p[1]-p[0], p[2]-p[0])
                extent = np.linalg.norm(np.ptp(p, axis=0))
                if (np.linalg.norm(normal) == 0 or extent == 0
                        or abs(np.dot(p[3]-p[0], normal)) > 1e-6 * extent * np.linalg.norm(normal)
                        or any(np.dot(np.cross(p[(i+1)%4]-p[i], p[(i+2)%4]-p[(i+1)%4]), normal) < 0
                               for i in range(4))):
                    raise ValueError(f"TacSL rejects nonplanar/nonconvex query quad: {prim.GetPath()}")
            faces.extend([face[[0, i, i+1]] for i in range(1, count-1)])
        faces = np.asarray(faces, dtype=np.int32)
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            faces = faces[:, ::-1].copy()
        if point_transform is not None:
            matrix = np.asarray(point_transform, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("TacSL requires a finite4x4 query transform")
            points = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
        return cls(points, faces, device)

    def get_sdf_and_gradients(self, points):
        tensor = wp.to_torch(points) if isinstance(points, wp.array) else points
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != 3:
            raise ValueError(f"TacSL mesh query expects 1xNx3 points, got {tuple(tensor.shape)}")
        query = wp.from_torch(tensor.contiguous())
        output = torch.empty((*tensor.shape[:-1], 4), dtype=torch.float32, device=tensor.device)
        wp.launch(
            _signed_mesh_distance,
            dim=tensor.shape[:2],
            inputs=[self.mesh.id, query, wp.from_torch(output)],
            device=self.device,
        )
        if not torch.isfinite(output).all():
            raise RuntimeError("TacSL real collision mesh distance query failed")
        return wp.from_torch(output)
