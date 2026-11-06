import copy

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from shapely.geometry import MultiPoint, Polygon

from utils.rotations import quat_multiply, quat_to_rot_matrix


def quat_to_mat(q):
    """Active rotation matrix for an XYZW quaternion."""
    return quat_to_rot_matrix(q)


def cal_two_axis_angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def pose_to_matrix(pose):
    """Convert 7D pose (x,y,z,qx,qy,qz,qw) to 4x4 homogeneous transformation matrix"""
    x, y, z, qx, qy, qz, qw = pose
    rotation = R.from_quat([qx, qy, qz, qw])
    rot_matrix = rotation.as_matrix()
    matrix = np.eye(4)
    matrix[0:3, 0:3] = rot_matrix
    matrix[0:3, 3] = [x, y, z]
    return matrix


def matrix_to_pose(matrix):
    """Convert 4x4 homogeneous matrix back to 7D pose (x,y,z,qx,qy,qz,qw)"""
    position = matrix[0:3, 3]
    rot_matrix = matrix[0:3, 0:3]
    rotation = R.from_matrix(rot_matrix)
    quaternion = rotation.as_quat()
    pose = np.concatenate([position, quaternion])
    return pose


def calculate_target_pose(real_pose, set_pose, relative_real_pose):
    """
    C relative to A = D relative to B
    Compute relative pose transformation between frames
    """
    T_A = pose_to_matrix(real_pose)
    T_B = pose_to_matrix(set_pose)
    T_C = pose_to_matrix(relative_real_pose)

    R_A = T_A[0:3, 0:3]
    t_A = T_A[0:3, 3]
    inv_A = np.eye(4)
    inv_A[0:3, 0:3] = R_A.T
    inv_A[0:3, 3] = -R_A.T @ t_A
    T_relative = inv_A @ T_C
    T_D = T_B @ T_relative
    relative_set_pose = matrix_to_pose(T_D)

    return relative_set_pose


def cal_quat_dis(quat1, quat2):
    """Shortest angular distance (radians) between XYZW rotations, invariant to sign."""
    q1, q2 = np.asarray(quat1, dtype=float), np.asarray(quat2, dtype=float)
    norm = np.linalg.norm(q1) * np.linalg.norm(q2)
    if norm == 0:
        raise ValueError("Angular distance requires nonzero quaternions")
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)) / norm, 0.0, 1.0))


def calc_polygon(
    origin_pose: np.ndarray,
    origin_bbox_points: np.ndarray,
    margin: float = 0.0,
) -> tuple[Polygon, float]:
    """
    Transform object bounding box vertices to world frame,
    use XY convex hull as footprint polygon,
    return polygon and min/max Z values.
    """
    origin_pose = np.asarray(origin_pose, dtype=float).reshape(7)
    origin_bbox_points = np.asarray(origin_bbox_points, dtype=float).reshape(-1, 3)

    rot_mat = quat_to_mat(origin_pose[3:])
    bbox_points_world = origin_bbox_points @ rot_mat.T + origin_pose[:3]

    hull = MultiPoint(bbox_points_world[:, :2]).convex_hull
    polygon = hull.buffer(margin)
    z_min = bbox_points_world[:, 2].min()
    z_max = bbox_points_world[:, 2].max()
    return polygon, z_min, z_max


def axis_angle_to_quat_xyzw(axis, angle_deg):
    """
    Convert rotation axis and angle (degrees) to quaternion [x, y, z, w]
    axis: rotation axis in world frame, shape (3,)
    angle_deg: rotation angle in degrees
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("axis norm is too small")

    axis = axis / norm
    angle_rad = np.deg2rad(angle_deg)
    half = angle_rad / 2.0

    w = np.cos(half)
    xyz = axis * np.sin(half)

    return np.array([xyz[0], xyz[1], xyz[2], w], dtype=np.float64)


def rotate_quat_about_world_axis(q_xyzw, axis_world, angle_deg, normalize=True):
    """
    Rotate quaternion q_xyzw around world axis by angle_deg degrees

    Parameters:
        q_xyzw: current quaternion [x, y, z, w]
        axis_world: rotation axis in world frame [x, y, z]
        angle_deg: rotation angle in degrees
        normalize: whether to normalize output

    Return:
        new quaternion [x, y, z, w]
    """
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    q_delta = axis_angle_to_quat_xyzw(axis_world, angle_deg)

    # Rotate around world axis => left multiplication
    q_new = quat_multiply(q_delta, q_xyzw)

    if normalize:
        q_new = q_new / np.linalg.norm(q_new)

    return q_new


def check_1d(data):
    """
    Check whether the input data is 1-dimensional.

    Supports common input types such as:
    - list
    - tuple
    - numpy.ndarray
    - ListConfig-like objects that can be converted by np.asarray

    Args:
        data: Input data of any array-like type.

    Returns:
        bool: True if the input is 1D, otherwise False.
    """
    try:
        # Convert input to a NumPy array only for shape inspection.
        # This does not modify the original input data type.
        arr = np.asarray(data)
        return arr.ndim == 1
    except Exception:
        # Return False if the input cannot be interpreted as array-like data.
        return False


def check_2d(data):
    """
    Check whether the input data is 2-dimensional.

    This version allows ragged 2D data, e.g.:
        [[1, 2, 3], [4, 5]]
    where each row can have different length.

    Returns:
        bool: True if data is regular 2D or ragged 2D.
    """
    # First handle normal ndarray directly.
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            return True

        # Handle object ndarray that may contain ragged rows.
        if data.ndim == 1 and data.dtype == object:
            return _check_ragged_2d(data)

        return False

    # Handle list / tuple / ListConfig-like objects.
    return _check_ragged_2d(data)


def _check_ragged_2d(data):
    """
    Check ragged 2D structure:
        outer dimension exists;
        each element in outer dimension is 1D array-like.
    """
    try:
        rows = list(data)
    except Exception:
        return False

    # Empty list [] should be treated as not 2D.
    if len(rows) == 0:
        return False

    for row in rows:
        # Scalar means this is only 1D, e.g. [1, 2, 3]
        if np.isscalar(row):
            return False

        try:
            row_arr = np.asarray(row)
        except Exception:
            return False

        # Each row should be 1D.
        # Allow different lengths, but not nested 2D rows.
        if row_arr.ndim != 1:
            return False

    return True


def safe_deepcopy_keep_callable(obj):
    """
    Deep-copy an object while keeping callable objects unchanged.

    This function recursively copies dictionaries, lists, tuples, and sets.
    If an element is callable, it is returned as-is instead of being deep-copied.
    For other objects, deepcopy is attempted first; if it fails, the original
    object is returned.
    """
    if callable(obj):
        return obj
    elif isinstance(obj, dict):
        return {k: safe_deepcopy_keep_callable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_deepcopy_keep_callable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(safe_deepcopy_keep_callable(v) for v in obj)
    elif isinstance(obj, set):
        return {safe_deepcopy_keep_callable(v) for v in obj}
    else:
        try:
            return copy.deepcopy(obj)
        except Exception:
            return obj


def _get_link_matrix_from_usd(usd_path: str, link_name: str) -> np.ndarray:
    """Return the 4x4 transform of `link_name` relative to the asset root
    at default joint positions, read directly from the USD file.
    Results are cached to avoid repeated file I/O.
    """
    usd_link_matrix_cache = {}
    cache_key = (usd_path, link_name)
    if cache_key in usd_link_matrix_cache:
        return usd_link_matrix_cache[cache_key]

    fallback = np.eye(4, dtype=np.float32)
    try:
        from pxr import Usd, UsdGeom

        def _extract_matrix(xformable):
            tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos = np.array(tf.ExtractTranslation(), dtype=np.float32)
            q = tf.ExtractRotationQuat()
            pose7 = np.array(
                [
                    pos[0],
                    pos[1],
                    pos[2],
                    q.GetImaginary()[0],
                    q.GetImaginary()[1],
                    q.GetImaginary()[2],
                    q.GetReal(),
                ],
                dtype=np.float32,
            )
            return pose_to_matrix(pose7)

        stage = Usd.Stage.Open(usd_path)
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            root_matrix = _extract_matrix(UsdGeom.Xformable(default_prim))
        else:
            root_matrix = np.eye(4, dtype=np.float32)

        for prim in stage.Traverse():
            if prim.GetName() == link_name:
                link_matrix = _extract_matrix(UsdGeom.Xformable(prim))
                result = (np.linalg.inv(root_matrix) @ link_matrix).astype(np.float32)
                usd_link_matrix_cache[cache_key] = result
                return result

        print(f"Warning: link '{link_name}' not found in {usd_path}, using identity")
    except Exception as e:
        print(f"Warning: could not read link transform from {usd_path}: {e}")

    usd_link_matrix_cache[cache_key] = fallback
    return fallback


def is_point_in_3d_bbox_vertices(point, bbox_vertices, atol=1e-6):
    point = np.asarray(point, dtype=float).reshape(-1)[:3]
    bbox_vertices = np.asarray(bbox_vertices, dtype=float).reshape(-1, 3)
    if bbox_vertices.shape[0] < 4:
        return False

    bbox_hull = ConvexHull(bbox_vertices)
    return bool(np.all(bbox_hull.equations[:, :3] @ point + bbox_hull.equations[:, 3] <= atol))
