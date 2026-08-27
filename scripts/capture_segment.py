"""Tabletop object segmentation for the D455 capture pipeline.

Depth back-projection, RANSAC plane fit, connected-region segmentation, and
green-hue checks -- all numpy over saved depth/RGB arrays, no omni/USD
access. Offline-unit-testable against saved .npy/.png fixtures.
"""

import numpy as np

import yaw_estimate
from capture_constants import SELF_BODY_MAX_CAMERA_DISTANCE_M
from capture_geometry import _displacement

BORDER_MARGIN_PX = 2
_PLANE_TOLERANCE_M = 0.005  # RANSAC inlier tolerance; also region_overlay's TABLE band


def _select_target_surface_pixel(depth_axial, self_body_max_m=SELF_BODY_MAX_CAMERA_DISTANCE_M):
    """Find the nearest valid depth pixel beyond the self-body envelope.

    Args:
        depth_axial: Axial depth array.
        self_body_max_m: Depth below this counts as the robot's own body.

    Returns:
        dict with target_pixel_px ([u, v] or None), self_body_excluded_px,
        self_body_max_m, valid_px, remaining_px, nearest_remaining_depth_m,
        nearest_excluded_depth_m.

    Note:
        No object identification -- just the nearest thing depth saw that
        isn't the robot's own gripper. target_pixel_px is None (never a
        fallback to a self-body pixel) if nothing valid remains past the
        exclusion.
    """
    valid = np.isfinite(depth_axial) & (depth_axial > 0)
    excluded = valid & (depth_axial < self_body_max_m)
    remaining = valid & ~excluded
    remaining_depths = depth_axial[remaining]
    excluded_depths = depth_axial[excluded]
    stats = {
        "target_pixel_px": None,
        "self_body_excluded_px": int(np.count_nonzero(excluded)),
        "self_body_max_m": self_body_max_m,
        "valid_px": int(np.count_nonzero(valid)),
        "remaining_px": int(np.count_nonzero(remaining)),
        "nearest_remaining_depth_m": float(remaining_depths.min()) if remaining_depths.size else None,
        "nearest_excluded_depth_m": float(excluded_depths.min()) if excluded_depths.size else None,
    }
    if remaining_depths.size:
        masked = np.where(remaining, depth_axial, np.inf)
        v, u = np.unravel_index(np.argmin(masked), masked.shape)
        stats["target_pixel_px"] = [int(u), int(v)]
    return stats


def _backproject_depth_to_world(depth_axial, intrinsics, camera_frame):
    """Back-project every pixel's depth into world xyz.

    Args:
        depth_axial: Axial depth array.
        intrinsics: dict with fx, fy, cx, cy.
        camera_frame: Camera pose (a _camera_frame() return value).

    Returns:
        (H, W, 3) world xyz array. Values at invalid pixels are
        meaningless and must be masked by the caller.
    """
    height, width = depth_axial.shape
    v_idx, u_idx = np.mgrid[0:height, 0:width]
    depth = depth_axial.astype(np.float64)
    x_right = (u_idx - intrinsics["cx"]) * depth / intrinsics["fx"]
    y_up = -(v_idx - intrinsics["cy"]) * depth / intrinsics["fy"]
    position = np.asarray(camera_frame["position_m"], dtype=np.float64)
    right = np.asarray(camera_frame["right"], dtype=np.float64)
    up = np.asarray(camera_frame["up"], dtype=np.float64)
    forward = np.asarray(camera_frame["forward"], dtype=np.float64)
    return (
        position
        + x_right[..., None] * right
        + y_up[..., None] * up
        + depth[..., None] * forward
    )


def _fit_dominant_plane(points_world, iterations=200, tolerance_m=_PLANE_TOLERANCE_M, seed=0):
    """Fit the dominant plane in a world-frame point cloud.

    Args:
        points_world: World xyz points, any shape ending in 3.
        iterations: RANSAC iterations.
        tolerance_m: Inlier distance tolerance, metres.
        seed: Random seed.

    Returns:
        dict with normal, offset_m, inlier_px, inlier_fraction. None if
        fewer than 3 points are given, or no valid plane is found.

    Note:
        RANSAC finds the inlier set; the final normal and offset are
        refined by SVD over that set. Normal is oriented toward +Z
        (tabletop faces up).
    """
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    count = points.shape[0]
    if count < 3:
        return None
    rng = np.random.default_rng(seed)
    best_inliers, best_count = None, -1
    for _ in range(iterations):
        p0, p1, p2 = points[rng.choice(count, size=3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = float(np.dot(normal, p0))
        inliers = np.abs(points @ normal - offset) <= tolerance_m
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count > best_count:
            best_inliers, best_count = inliers, inlier_count
    if best_inliers is None or best_count < 3:
        return None
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    if normal[2] < 0:
        normal = -normal
    return {
        "normal": normal.tolist(),
        "offset_m": float(np.dot(normal, centroid)),
        "inlier_px": best_count,
        "inlier_fraction": float(best_count / count),
    }


def _mask_from_rows_cols(rows, cols, shape):
    """Build a boolean mask from row/column index arrays.

    Args:
        rows: Row indices.
        cols: Column indices.
        shape: Output mask shape.

    Returns:
        Boolean array of the given shape.
    """
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return mask


def _tabletop_object_regions(
    depth_axial,
    points_world,
    plane,
    intrinsics,
    self_body_max_m=SELF_BODY_MAX_CAMERA_DISTANCE_M,
    above_plane_min_m=0.010,
    above_plane_max_m=0.500,
    min_px=80,
    depth_step_m=0.02,
):
    """Segment pixels above the table plane into connected regions.

    Args:
        depth_axial: Axial depth array.
        points_world: World xyz per pixel, from _backproject_depth_to_world().
        plane: Table plane, with "normal" and "offset_m".
        intrinsics: Unused; kept for signature compatibility.
        self_body_max_m: Depth below this counts as the robot's own body.
        above_plane_min_m: Minimum height above the plane to be considered.
        above_plane_max_m: Maximum height above the plane to be considered.
        min_px: Minimum region pixel count to keep.
        depth_step_m: Maximum depth jump within a connected region.

    Returns:
        List of region dicts, each with label, pixel_count, bbox_px,
        height_above_plane_m, depth_median_m, touches_border, mask_rows,
        mask_cols.

    Note:
        Regions are 8-connected; a depth step greater than depth_step_m
        breaks connectivity. intrinsics is unused because points_world is
        already back-projected.
    """
    del intrinsics
    height, width = depth_axial.shape
    valid = np.isfinite(depth_axial) & (depth_axial > 0)
    self_body_mask = valid & (depth_axial < self_body_max_m)
    normal = np.asarray(plane["normal"], dtype=np.float64)
    signed_height = np.dot(points_world, normal) - plane["offset_m"]
    above_mask = (
        valid
        & ~self_body_mask
        & (signed_height >= above_plane_min_m)
        & (signed_height <= above_plane_max_m)
    )
    visited = np.zeros((height, width), dtype=bool)
    regions = []
    for start_v in range(height):
        for start_u in range(width):
            if not above_mask[start_v, start_u] or visited[start_v, start_u]:
                continue
            stack = [(start_v, start_u)]
            visited[start_v, start_u] = True
            rows, cols = [], []
            while stack:
                v, u = stack.pop()
                rows.append(v)
                cols.append(u)
                current_depth = float(depth_axial[v, u])
                for dv in (-1, 0, 1):
                    for du in (-1, 0, 1):
                        nv, nu = v + dv, u + du
                        if dv == 0 and du == 0:
                            continue
                        if not (0 <= nv < height and 0 <= nu < width):
                            continue
                        if not above_mask[nv, nu] or visited[nv, nu]:
                            continue
                        if abs(float(depth_axial[nv, nu]) - current_depth) <= depth_step_m:
                            visited[nv, nu] = True
                            stack.append((nv, nu))
            if len(rows) < min_px:
                continue
            rows_arr, cols_arr = np.asarray(rows), np.asarray(cols)
            bbox = [int(cols_arr.min()), int(rows_arr.min()), int(cols_arr.max()), int(rows_arr.max())]
            touches_border = bool(
                bbox[0] <= BORDER_MARGIN_PX
                or bbox[1] <= BORDER_MARGIN_PX
                or bbox[2] >= width - 1 - BORDER_MARGIN_PX
                or bbox[3] >= height - 1 - BORDER_MARGIN_PX
            )
            regions.append(
                {
                    "label": len(regions) + 1,
                    "pixel_count": int(len(rows)),
                    "bbox_px": bbox,
                    "height_above_plane_m": float(np.median(signed_height[rows_arr, cols_arr])),
                    "depth_median_m": float(np.median(depth_axial[rows_arr, cols_arr])),
                    "touches_border": touches_border,
                    "mask_rows": rows_arr,
                    "mask_cols": cols_arr,
                }
            )
    return regions


def _region_surface_center(region_mask, points_world, depth_axial):
    """Find a region's representative surface point and world extent.

    Args:
        region_mask: Boolean mask of the region, same shape as the image.
        points_world: World xyz per pixel, from _backproject_depth_to_world().
        depth_axial: Unused; kept for signature compatibility.

    Returns:
        dict with surface_centroid_world_xyz_m (mean of the region's world
        points), representative_pixel_px and surface_center_world_xyz_m
        (the region's actual point closest to that centroid),
        centroid_to_surface_offset_m (distance between the two), and
        surface_extent_world_m (per-axis world-space bounding size).
    """
    del depth_axial
    ys, xs = np.nonzero(region_mask)
    points = points_world[ys, xs]
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    best = int(np.argmin(distances))
    surface_center = points[best]
    extent = points.max(axis=0) - points.min(axis=0)
    return {
        "surface_centroid_world_xyz_m": centroid.tolist(),
        "representative_pixel_px": [int(xs[best]), int(ys[best])],
        "surface_center_world_xyz_m": surface_center.tolist(),
        "centroid_to_surface_offset_m": float(np.linalg.norm(centroid - surface_center)),
        "surface_extent_world_m": {
            "x": float(extent[0]),
            "y": float(extent[1]),
            "z": float(extent[2]),
        },
    }


def _rgb_to_hsv(rgb):
    """Convert an RGB array to HSV.

    Args:
        rgb: RGB array, 0-255 range.

    Returns:
        (hue, saturation, value). hue in degrees [0, 360); saturation and
        value on the 0-1 scale.
    """
    array = rgb.astype(np.float64) / 255.0
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    cmax = array.max(axis=-1)
    cmin = array.min(axis=-1)
    delta = cmax - cmin
    safe_delta = np.where(delta == 0, 1.0, delta)
    is_r = (cmax == r) & (delta != 0)
    is_g = (cmax == g) & (delta != 0) & ~is_r
    is_b = (cmax == b) & (delta != 0) & ~is_r & ~is_g
    hue = np.zeros_like(cmax)
    hue = np.where(is_r, 60.0 * (((g - b) / safe_delta) % 6.0), hue)
    hue = np.where(is_g, 60.0 * (((b - r) / safe_delta) + 2.0), hue)
    hue = np.where(is_b, 60.0 * (((r - g) / safe_delta) + 4.0), hue)
    saturation = np.where(cmax == 0, 0.0, delta / np.where(cmax == 0, 1.0, cmax))
    return hue, saturation, cmax


# A dark, low-saturation green material can cross a hue-only green gate
# without being the (high-saturation, bright) unhidden IK target; gating on
# saturation and value too is how the two are told apart.
_GREEN_SATURATION_MIN = 0.45
_GREEN_VALUE_MIN = 0.35


def _region_green_fraction(rgb_left, region_mask):
    """Fraction of a region's pixels that are green (IK-target-colored).

    Args:
        rgb_left: Left-eye RGB array.
        region_mask: Boolean mask of the region, same shape as the image.

    Returns:
        (green_fraction, green_hue_only_fraction). green_hue_only_fraction
        is a hue-only definition, kept for comparison.

    Note:
        green_fraction requires hue in [90,150] deg AND saturation >= 0.45
        AND value >= 0.35.
    """
    hue, sat, val = _rgb_to_hsv(rgb_left)
    hue, sat, val = hue[region_mask], sat[region_mask], val[region_mask]
    if hue.size == 0:
        return 0.0, 0.0
    hue_only = (hue >= 90.0) & (hue <= 150.0)
    green = hue_only & (sat >= _GREEN_SATURATION_MIN) & (val >= _GREEN_VALUE_MIN)
    return (
        float(np.count_nonzero(green) / hue.size),
        float(np.count_nonzero(hue_only) / hue.size),
    )


def _build_tabletop_objects(
    depth_axial,
    rgb_left,
    intrinsics,
    camera_frame,
    arm_base_world_xyz_m,
    self_body_max_m=SELF_BODY_MAX_CAMERA_DISTANCE_M,
):
    """Detect tabletop objects: plane fit, region segmentation, per-region
    surface center and green check.

    Args:
        depth_axial: Axial depth array.
        rgb_left: Left-eye RGB array.
        intrinsics: dict with fx, fy, cx, cy.
        camera_frame: Camera pose (a _camera_frame() return value).
        arm_base_world_xyz_m: Arm base world position.
        self_body_max_m: Depth below this counts as the robot's own body.

    Returns:
        dict with plane, objects (list, each also carrying yaw_estimate
        fields and an internal "mask" array), failures, table_mask,
        points_world.
    """
    points_world = _backproject_depth_to_world(depth_axial, intrinsics, camera_frame)
    valid = np.isfinite(depth_axial) & (depth_axial > 0)
    self_body_mask = valid & (depth_axial < self_body_max_m)
    plane = _fit_dominant_plane(points_world[valid & ~self_body_mask])
    if plane is None:
        return {
            "plane": None, "objects": [], "failures": ["table plane fit failed -- insufficient points"],
            "table_mask": np.zeros(depth_axial.shape, dtype=bool), "points_world": points_world,
        }
    normal = np.asarray(plane["normal"], dtype=np.float64)
    signed_height = np.dot(points_world, normal) - plane["offset_m"]
    table_mask = valid & ~self_body_mask & (np.abs(signed_height) <= _PLANE_TOLERANCE_M)
    regions = _tabletop_object_regions(depth_axial, points_world, plane, intrinsics, self_body_max_m=self_body_max_m)
    objects, failures = [], []
    for region in regions:
        mask = _mask_from_rows_cols(region["mask_rows"], region["mask_cols"], depth_axial.shape)
        center = _region_surface_center(mask, points_world, depth_axial)
        # The >0.5 green-flag failure message is generated in
        # capture_run._collect_failures(), not here -- that's the one place
        # that also has green_audit, needed to tell a genuine unhidden
        # target apart from a green-flag/green_audit inconsistency.
        green_fraction, green_hue_only_fraction = _region_green_fraction(rgb_left, mask)
        cam_vec, cam_norm = _displacement(camera_frame["position_m"], center["surface_center_world_xyz_m"])
        arm_vec, _ = _displacement(arm_base_world_xyz_m, center["surface_center_world_xyz_m"])
        yaw_info = yaw_estimate.region_yaw_estimate(mask, points_world, plane, region["height_above_plane_m"])
        objects.append(
            {
                "id": region["label"],
                "pixel_count": region["pixel_count"],
                "bbox_px": region["bbox_px"],
                "height_above_plane_m": region["height_above_plane_m"],
                "depth_median_m": region["depth_median_m"],
                "touches_border": region["touches_border"],
                "center_reliability": "truncated_by_image_border" if region["touches_border"] else "ok",
                "camera_to_surface_center_distance_m": cam_norm,
                "vector_camera_to_object_m": cam_vec,
                "vector_armbase_to_object_m": arm_vec,
                "green_hue_fraction": green_fraction,
                "green_hue_only_fraction": green_hue_only_fraction,
                "mask": mask,
                **center,
                **yaw_info,
            }
        )
    return {"plane": plane, "objects": objects, "failures": failures, "table_mask": table_mask, "points_world": points_world}
