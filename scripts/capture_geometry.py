"""Geometry/pose math for the D455 capture pipeline.

Quaternion<->axis conversion, camera-frame projection/back-projection, and
orbit-pose synthesis -- all pure, offline-unit-testable, no simulator or USD
library imports. _camera_frame()/_arm_base_world_xyz() are the two
exceptions: they read live USD/runtime state.
"""

import math

import numpy as np

from capture_constants import ARM_BASE_PATH


def _quaternion_wxyz_to_axes(w, x, y, z):
    """Convert a scalar-first (wxyz) quaternion to right/up/back world axes.

    Args:
        w, x, y, z: Quaternion components.

    Returns:
        (right, up, back), each a unit 3-vector in world coordinates.

    Note:
        back is the local +Z axis expressed in world coordinates; USD/Isaac
        Sim cameras look down local -Z.
    """
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation[:, 0], rotation[:, 1], rotation[:, 2]


def _rotation_axes_to_quaternion_wxyz(right, up, back):
    """Inverse of _quaternion_wxyz_to_axes(): world axes to quaternion.

    Args:
        right, up, back: Orthonormal, right-handed world axes (as columns
            of a rotation matrix).

    Returns:
        Scalar-first (wxyz) quaternion.

    Note:
        Uses Shepperd's method.
    """
    m00, m10, m20 = right
    m01, m11, m21 = up
    m02, m12, m22 = back
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def orbit_pose(pivot_world, radius_m, pitch_deg, yaw_deg, up_world=(0, 0, 1)):
    """A camera pose orbiting pivot_world at radius_m.

    pitch_deg=0 is straight overhead (forward = -up_world); pitch_deg tilts
    off top-down toward the side; yaw_deg sweeps in azimuth.

    Args:
        pivot_world: World xyz the camera orbits and looks at.
        radius_m: Orbit radius, metres.
        pitch_deg: Tilt off top-down, degrees.
        yaw_deg: Azimuth, degrees.
        up_world: World up direction.

    Returns:
        dict with position_m, quaternion_wxyz (w,x,y,z), forward, right, up.

    Note:
        Uses an explicit pitch/yaw spherical parametrization, not a
        cross(forward, up_world) look-at construction, because that
        construction is degenerate at pitch=0 -- exactly the case this
        function must get right.
    """
    pivot = np.asarray(pivot_world, dtype=np.float64)
    up_n = np.asarray(up_world, dtype=np.float64)
    up_n = up_n / np.linalg.norm(up_n)
    reference = np.array([1.0, 0.0, 0.0]) if abs(np.dot(up_n, [1.0, 0.0, 0.0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up_n, reference)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up_n, e1)

    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    radial = math.cos(yaw) * e1 + math.sin(yaw) * e2
    direction_from_pivot = math.cos(pitch) * up_n + math.sin(pitch) * radial

    position = pivot + radius_m * direction_from_pivot
    forward = -direction_from_pivot  # unit: position -> pivot, by construction
    right = -math.sin(yaw) * e1 + math.cos(yaw) * e2  # horizontal, always orthogonal to up_n and to radial
    up = np.cross(right, forward)
    back = -forward
    quaternion_wxyz = _rotation_axes_to_quaternion_wxyz(right, up, back)

    return {
        "position_m": position.tolist(),
        "quaternion_wxyz": quaternion_wxyz.tolist(),
        "forward": forward.tolist(),
        "right": right.tolist(),
        "up": up.tolist(),
    }


def _intrinsics_from_fov(width, height, horizontal_fov_rad):
    """Pinhole intrinsics from image size and horizontal FOV.

    Args:
        width: Image width, px.
        height: Image height, px.
        horizontal_fov_rad: Horizontal field of view, radians.

    Returns:
        dict with fx, fy, cx, cy, source.

    Note:
        Assumes square pixels (fx == fy); uses no calibration data.
    """
    fx = (width / 2.0) / math.tan(horizontal_fov_rad / 2.0)
    return {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0, "source": "geometric"}


def _pixel_depth_to_camera_xyz(u, v, depth_m, intrinsics):
    """Back-project one pixel and its axial depth into camera-frame xyz.

    Args:
        u, v: Pixel coordinates.
        depth_m: Axial depth at (u, v), metres.
        intrinsics: dict with fx, fy, cx, cy.

    Returns:
        [x_right, y_up, z_forward] in the camera frame.

    Note:
        forward is the optical axis; u grows rightward (same sign as
        right); v grows downward (opposite sign from up).
    """
    x_right = (u - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y_up = -(v - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    return [x_right, y_up, float(depth_m)]


def _camera_to_world_xyz(camera_xyz, camera_frame):
    """Transform a camera-frame xyz to world xyz.

    Args:
        camera_xyz: [x_right, y_up, z_forward] in the camera frame.
        camera_frame: dict with position_m, right, up, forward (a
            _camera_frame() return value).

    Returns:
        World xyz.
    """
    position = np.asarray(camera_frame["position_m"], dtype=np.float64)
    right = np.asarray(camera_frame["right"], dtype=np.float64)
    up = np.asarray(camera_frame["up"], dtype=np.float64)
    forward = np.asarray(camera_frame["forward"], dtype=np.float64)
    x_right, y_up, z_forward = camera_xyz
    world = position + x_right * right + y_up * up + z_forward * forward
    return world.tolist()


def _world_xyz_to_camera_xyz(world_xyz, camera_frame):
    """Inverse of _camera_to_world_xyz(): transform world xyz to camera-frame xyz.

    Args:
        world_xyz: World xyz.
        camera_frame: dict with position_m, right, up, forward (a
            _camera_frame() return value).

    Returns:
        [x_right, y_up, z_forward] in the camera frame.
    """
    position = np.asarray(camera_frame["position_m"], dtype=np.float64)
    vector = np.asarray(world_xyz, dtype=np.float64) - position
    right = np.asarray(camera_frame["right"], dtype=np.float64)
    up = np.asarray(camera_frame["up"], dtype=np.float64)
    forward = np.asarray(camera_frame["forward"], dtype=np.float64)
    return [float(np.dot(vector, right)), float(np.dot(vector, up)), float(np.dot(vector, forward))]


def _camera_xyz_to_pixel(camera_xyz, intrinsics):
    """Inverse of _pixel_depth_to_camera_xyz(): project camera-frame xyz to a pixel.

    Args:
        camera_xyz: [x_right, y_up, z_forward] in the camera frame.
        intrinsics: dict with fx, fy, cx, cy.

    Returns:
        (u, v) pixel coordinates, or None if behind the camera
        (z_forward <= 0); the caller decides what "outside FOV" means.
    """
    x_right, y_up, z_forward = camera_xyz
    if z_forward <= 0:
        return None
    u = intrinsics["cx"] + x_right * intrinsics["fx"] / z_forward
    v = intrinsics["cy"] - y_up * intrinsics["fy"] / z_forward
    return u, v


def _displacement(from_xyz, to_xyz):
    """World-frame vector from from_xyz to to_xyz, and its norm.

    Args:
        from_xyz: World xyz.
        to_xyz: World xyz.

    Returns:
        (vector, norm).
    """
    vector = np.asarray(to_xyz, dtype=np.float64) - np.asarray(from_xyz, dtype=np.float64)
    return vector.tolist(), float(np.linalg.norm(vector))


def _camera_frame(stage, camera_path):
    """World pose of a camera prim: position and right/up/forward axes.

    Args:
        stage: USD stage.
        camera_path: Camera prim path.

    Returns:
        dict with position_m, right, up, forward.
    """
    import omni.timeline

    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_path}")
    if omni.timeline.get_timeline_interface().is_playing():
        from isaacsim.core.experimental.prims import XformPrim

        positions, quaternions = XformPrim(camera_path).get_world_poses()
        position = positions.numpy()[0].astype(np.float64)
        w, x, y, z = quaternions.numpy()[0].astype(np.float64)
        right, up, back = _quaternion_wxyz_to_axes(w, x, y, z)
    else:
        from pxr import Usd, UsdGeom

        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        position = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
        authored_rotation = matrix.ExtractRotationMatrix()
        axes = np.array(
            [
                [authored_rotation[row][column] for column in range(3)]
                for row in range(3)
            ],
            dtype=np.float64,
        )
        right, up, back = axes[0], axes[1], axes[2]
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    forward = -back / np.linalg.norm(back)
    return {
        "position_m": position.tolist(),
        "right": right.tolist(),
        "up": up.tolist(),
        "forward": forward.tolist(),
    }


def _arm_base_world_xyz(stage):
    """World position of ARM_BASE_PATH.

    Args:
        stage: USD stage.

    Returns:
        World xyz.

    Note:
        Reads runtime/Fabric state while Play is active, not authored
        values.
    """
    import omni.timeline

    prim = stage.GetPrimAtPath(ARM_BASE_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"Arm base prim not found: {ARM_BASE_PATH}")
    if omni.timeline.get_timeline_interface().is_playing():
        from isaacsim.core.experimental.prims import XformPrim

        positions, _ = XformPrim(ARM_BASE_PATH).get_world_poses()
        position = positions.numpy()[0].astype(np.float64)
    else:
        from pxr import Usd, UsdGeom

        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        position = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
    return position.tolist()
