"""Shape-general grasp-axis estimation from a tabletop region's top-face
points. Pure geometry, no simulator dependency -- stdlib + numpy only,
testable offline.
"""

import math

import numpy as np

ROUND_FILL_RATIO_MAX = 0.89

ELONGATED_ASPECT_RATIO_MIN = 1.2


def convex_hull_2d(points_xy):
    """Convex hull of a 2D point set, via the monotone chain algorithm.

    Args:
        points_xy: (N, 2) array-like of points.

    Returns:
        Hull vertices in CCW order as a list of (x, y) tuples. Fewer than
        3 distinct input points is returned as-is (degenerate, no area).
    """
    pts = sorted(set(map(tuple, np.asarray(points_xy).tolist())))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def rotating_calipers_min_rect(hull_xy):
    """Minimum-area bounding rectangle of a convex polygon, via rotating
    calipers.

    The minimum-area rectangle always has one side collinear with a hull
    edge, so trying every edge's angle and keeping the smallest
    axis-aligned bbox in that edge's rotated frame is an exact solution,
    not a heuristic.

    Args:
        hull_xy: Convex hull vertices, (N, 2) array-like.

    Returns:
        (angle_rad, side_a_m, side_b_m) of the winning orientation.
    """
    hull = np.asarray(hull_xy, dtype=np.float64)
    best, best_area = (0.0, 0.0, 0.0), None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        angle = math.atan2(edge[1], edge[0])
        c, s = math.cos(-angle), math.sin(-angle)
        rotated = hull @ np.array([[c, -s], [s, c]]).T
        span = rotated.max(axis=0) - rotated.min(axis=0)
        area = float(span[0] * span[1])
        if best_area is None or area < best_area:
            best_area, best = area, (angle, float(span[0]), float(span[1]))
    return best


def _polygon_area(hull_xy):
    """Area of a simple polygon via the shoelace formula.

    Args:
        hull_xy: Polygon vertices in order (CW or CCW), (N, 2) array-like.

    Returns:
        Area as a float. Fewer than 3 points returns 0.0.
    """
    pts = np.asarray(hull_xy, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def region_yaw_estimate(mask, points_world, plane, height_above_plane_m, band_m=0.005):
    """Estimate the grasp axis and shape class of one tabletop region.

    Projects the region's top-face points (within +-band_m of their median
    height) onto the world XY plane, fits a minimum-area rectangle with
    rotating calipers, and derives the grasp axis from the rectangle's
    sides and fill ratio.

    Args:
        mask: Boolean mask of the region, same shape as the image.
        points_world: Per-pixel world coordinates, shape (H, W, 3).
        plane: Table plane; must contain "normal" and "offset_m".
        height_above_plane_m: Region top-face height above the table, metres.
        band_m: Height tolerance for selecting top-face points, metres.

    Returns:
        dict with:
          grip_yaw_deg        Direction of the rectangle's short side, degrees
                              mod 180. The gripper's finger axis aligns to it.
          grip_width_mm       Short side length: the width the gripper spans.
          object_length_mm    Long side length.
          symmetry_period_deg 90 (square-like), 180 (elongated), None (round).
          shape_class         square_like / elongated / round / unknown.
          fill_ratio          Convex hull area / min-rect area.
          yaw_deg_estimated   Legacy, mod 90. No discriminative power for
                              non-square objects; do not use.
        All numeric fields are None when fewer than 3 top-face points exist.
        No guessed defaults are ever returned.

    Note:
        Do not switch to PCA: a square's covariance matrix is isotropic, so
        the eigenvector direction is numerical noise, not signal.
    """
    normal = np.asarray(plane["normal"], dtype=np.float64)
    ys, xs = np.nonzero(mask)
    pts = points_world[ys, xs]
    signed_height = np.dot(pts, normal) - plane["offset_m"]
    top_xy = pts[np.abs(signed_height - height_above_plane_m) <= band_m][:, :2]
    hull = convex_hull_2d(top_xy) if len(top_xy) >= 3 else list(map(tuple, top_xy))
    if len(hull) < 3:
        return {
            "yaw_deg_estimated": None, "rect_side_a_mm": None, "rect_side_b_mm": None,
            "rect_aspect_ratio": None, "hull_point_count": int(len(hull)),
            "yaw_method": "min_area_rect_rotating_calipers",
            "grip_yaw_deg": None, "grip_width_mm": None, "object_length_mm": None,
            "symmetry_period_deg": None, "shape_class": "unknown", "fill_ratio": None,
        }
    angle_rad, side_a_m, side_b_m = rotating_calipers_min_rect(hull)
    lo, hi = sorted((side_a_m, side_b_m))
    aspect_ratio = (hi / lo) if lo > 0 else None
    rect_area = side_a_m * side_b_m
    fill_ratio = (_polygon_area(hull) / rect_area) if rect_area > 0 else None

    # side_a_m spans the `angle_rad` direction, side_b_m the perpendicular
    # one (rotating_calipers_min_rect's docstring); whichever is shorter is
    # the axis the gripper's fingers must close along.
    if side_a_m <= side_b_m:
        grip_width_m, object_length_m, grip_axis_rad = side_a_m, side_b_m, angle_rad
    else:
        grip_width_m, object_length_m, grip_axis_rad = side_b_m, side_a_m, angle_rad + math.pi / 2.0
    grip_yaw_deg = math.degrees(grip_axis_rad) % 180.0

    # Order matters: aspect ratio alone tells "elongated" apart from "round
    # or square" (both have aspect ratio ~1), and only among the latter two
    # does fill_ratio distinguish them -- see ELONGATED_ASPECT_RATIO_MIN's
    # comment for why fill_ratio can't do the elongated/round split itself.
    if aspect_ratio is not None and aspect_ratio > ELONGATED_ASPECT_RATIO_MIN:
        shape_class, symmetry_period_deg = "elongated", 180.0
    elif fill_ratio is not None and fill_ratio < ROUND_FILL_RATIO_MAX:
        shape_class, symmetry_period_deg = "round", None
    else:
        shape_class, symmetry_period_deg = "square_like", 90.0

    return {
        "yaw_deg_estimated": math.degrees(angle_rad) % 90.0,
        "rect_side_a_mm": side_a_m * 1000.0,
        "rect_side_b_mm": side_b_m * 1000.0,
        "rect_aspect_ratio": aspect_ratio,
        "hull_point_count": int(len(hull)),
        "yaw_method": "min_area_rect_rotating_calipers",
        "grip_yaw_deg": grip_yaw_deg,
        "grip_width_mm": grip_width_m * 1000.0,
        "object_length_mm": object_length_m * 1000.0,
        "symmetry_period_deg": symmetry_period_deg,
        "shape_class": shape_class,
        "fill_ratio": fill_ratio,
    }
