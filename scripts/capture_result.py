"""result.json / diagnostics.json assembly for the D455 capture pipeline.

Builds the user-facing and debug JSON dicts, saves the per-run PNG/npy
products, and runs the green-pixel audit. One function here
(_read_camera_intrinsics) reads USD camera attributes off a live stage;
kept next to its only caller (_build_measurement).
"""

import math
import os

import numpy as np

from capture_constants import SELF_BODY_MAX_CAMERA_DISTANCE_M, CAMERA_PATHS, WIDTH, HEIGHT
from capture_geometry import _displacement, _intrinsics_from_fov, _pixel_depth_to_camera_xyz, _camera_to_world_xyz
from capture_segment import _select_target_surface_pixel
from capture_annotate import _save_png, _depth_preview
from capture_run import _round_floats

_TABLETOP_JSON_KEYS = (
    "id", "pixel_count", "bbox_px", "height_above_plane_m", "depth_median_m",
    "representative_pixel_px", "surface_center_world_xyz_m",
    "surface_centroid_world_xyz_m", "centroid_to_surface_offset_m",
    "surface_extent_world_m", "touches_border", "center_reliability",
    "camera_to_surface_center_distance_m", "vector_camera_to_object_m",
    "vector_armbase_to_object_m", "green_hue_fraction", "green_hue_only_fraction",
    "yaw_deg_estimated", "rect_side_a_mm", "rect_side_b_mm", "rect_aspect_ratio", "hull_point_count", "yaw_method",
    "grip_yaw_deg", "grip_width_mm", "object_length_mm", "symmetry_period_deg", "shape_class", "fill_ratio",
)


def _build_tabletop_objects_json(tabletop):
    """Build the tabletop_objects section of result.json.

    Args:
        tabletop: Tabletop detection result, with "plane" and "objects".

    Returns:
        dict with method, plane, params, object_count, objects, note.

    Note:
        Drops the internal-only "mask" array that annotation drawing needs
        but the JSON output must not carry.
    """
    plane = tabletop["plane"]
    plane_json = None if plane is None else {
        "normal": plane["normal"], "offset_m": plane["offset_m"], "inlier_fraction": plane["inlier_fraction"],
    }
    objects_json = [{key: obj[key] for key in _TABLETOP_JSON_KEYS} for obj in tabletop["objects"]]
    return _round_floats(
        {
            "method": "plane_fit_then_connected_regions_v1",
            "plane": plane_json,
            "params": {
                "self_body_max_m": SELF_BODY_MAX_CAMERA_DISTANCE_M,
                "above_plane_min_m": 0.010,
                "above_plane_max_m": 0.500,
                "min_px": 80,
                "depth_step_m": 0.02,
            },
            "object_count": len(objects_json),
            "objects": objects_json,
            "note": (
                "All tabletop objects above the fitted table plane, excluding the "
                "robot's own body and the table surface itself. No semantic "
                "identification performed. Coordinates are surface centres, not "
                "volumetric centres."
            ),
        }
    )


# Center region matches tools/green_audit.py's CENTER exactly -- result.json
# (online) and the offline audit tool must always agree on the same numbers.
_GREEN_CENTER_X0, _GREEN_CENTER_X1 = 160, 480
_GREEN_CENTER_Y0, _GREEN_CENTER_Y1 = 120, 360


def _green_audit_eye(rgb):
    """Count green pixels (IK-target-colored) in one eye's RGB image.

    Args:
        rgb: RGB array.

    Returns:
        dict with total_px, center_px (within the fixed center region),
        bbox_px, centroid_px. bbox_px/centroid_px are None when total_px
        is 0.
    """
    red = rgb[:, :, 0].astype(np.int32)
    green = rgb[:, :, 1].astype(np.int32)
    blue = rgb[:, :, 2].astype(np.int32)
    mask = (green > 100) & (green > red * 1.6) & (green > blue * 1.6)
    total_px = int(np.count_nonzero(mask))
    center_px = int(
        np.count_nonzero(
            mask[_GREEN_CENTER_Y0:_GREEN_CENTER_Y1, _GREEN_CENTER_X0:_GREEN_CENTER_X1]
        )
    )
    if total_px == 0:
        return {"total_px": 0, "center_px": 0, "bbox_px": None, "centroid_px": None}
    ys, xs = np.nonzero(mask)
    return {
        "total_px": total_px,
        "center_px": center_px,
        "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_px": [int(xs.mean()), int(ys.mean())],
    }


def _build_result_json(
    run_id,
    failures,
    camera_world_xyz_m,
    arm_base_world_xyz_m,
    measurement,
    images,
    source_integrity,
    target_outside_color_fov,
    tabletop,
):
    """Assemble the user-facing result.json dict.

    Args:
        run_id: Run identifier.
        failures: List of failure message strings.
        camera_world_xyz_m: Left camera world position.
        arm_base_world_xyz_m: Arm base world position.
        measurement: Measurement dict, see _build_measurement().
        images: dict of output image filenames.
        source_integrity: Source-file staleness dict.
        target_outside_color_fov: Whether the legacy target point falls
            outside the color camera's frame.
        tabletop: Tabletop detection result.

    Returns:
        dict: the full result.json contents.
    """
    target_world_xyz_m = measurement["target_world_xyz_m"]
    if target_world_xyz_m is None:
        vectors = {
            "vector_camera_to_target_m": None,
            "vector_camera_to_target_norm_m": None,
            "vector_armbase_to_target_m": None,
            "vector_armbase_to_target_norm_m": None,
        }
    else:
        cam_vec, cam_norm = _displacement(camera_world_xyz_m, target_world_xyz_m)
        arm_vec, arm_norm = _displacement(arm_base_world_xyz_m, target_world_xyz_m)
        vectors = {
            "vector_camera_to_target_m": cam_vec,
            "vector_camera_to_target_norm_m": cam_norm,
            "vector_armbase_to_target_m": arm_vec,
            "vector_armbase_to_target_norm_m": arm_norm,
        }
    return _round_floats(
        {
            "status": "pass" if not failures else "fail",
            "run_id": run_id,
            "failures": failures,
            "units": {
                "length": "m",
                "angle": "rad",
                "pixel": "px",
                "note": "fields ending in _m are metres, _cm are centimetres, _px are pixels",
            },
            "accuracy_note": (
                "Simulated RTX depth. No camera calibration performed. Numeric precision "
                "is unverified and must not be treated as a calibrated metrological result."
            ),
            "camera_world_xyz_m": camera_world_xyz_m,
            "arm_base_world_xyz_m": arm_base_world_xyz_m,
            "target_pixel_px": measurement["target_pixel_px"],
            "camera_to_target_distance_m": measurement["camera_to_target_distance_m"],
            "camera_to_target_distance_cm": measurement["camera_to_target_distance_cm"],
            "target_camera_xyz_m": measurement["target_camera_xyz_m"],
            "target_world_xyz_m": target_world_xyz_m,
            **vectors,
            "self_body_exclusion": measurement["self_body_exclusion"],
            "intrinsics": measurement["intrinsics"],
            "images": images,
            "target_outside_color_fov": target_outside_color_fov,
            "source_integrity": source_integrity,
            "target_identification": {
                "method": "nearest_surface_beyond_self_body",
                "note": (
                    "legacy rule: nearest surface beyond self-body; kept for "
                    "comparison"
                ),
            },
            "tabletop_objects": _build_tabletop_objects_json(tabletop),
        }
    )


def _build_diagnostics_json(
    green_audit,
    render,
    frames,
    stage_sha256_before,
    stage_sha256_after,
    timeline_playing_before,
    ik_target_hidden,
    stereo_baseline_m,
    settle,
):
    """Assemble the debug-only diagnostics.json dict.

    Args:
        green_audit: Per-eye green-pixel audit results.
        render: Render diagnostics.
        frames: Per-camera _camera_frame() results.
        stage_sha256_before: Stage root layer sha256 before capture.
        stage_sha256_after: Stage root layer sha256 after capture.
        timeline_playing_before: Whether the timeline was playing before capture.
        ik_target_hidden: Whether the IK target was hidden during capture.
        stereo_baseline_m: Measured left/right camera baseline, metres.
        settle: IK settle-wait result.

    Returns:
        dict: the full diagnostics.json contents.
    """
    return _round_floats(
        {
            "green_audit": green_audit,
            "render": render,
            "camera_frames": frames,
            "stage_sha256_before": stage_sha256_before,
            "stage_sha256_after": stage_sha256_after,
            "timeline_playing_before": timeline_playing_before,
            "ik_target_hidden": ik_target_hidden,
            "stereo_baseline_m": stereo_baseline_m,
            "settle": settle,
        }
    )


def _save_capture_products(run_dir, rgb, axial_left, radial_left):
    """Write the six saved files for one run.

    Args:
        run_dir: Output directory.
        rgb: dict with "color"/"left"/"right" RGB arrays.
        axial_left: Left-eye axial depth array.
        radial_left: Left-eye radial depth array.
    """
    _save_png(rgb["color"], os.path.join(run_dir, "rgb_color.png"))
    _save_png(rgb["left"], os.path.join(run_dir, "rgb_left.png"))
    _save_png(rgb["right"], os.path.join(run_dir, "rgb_right.png"))
    np.save(os.path.join(run_dir, "depth_axial_left.npy"), axial_left)
    np.save(os.path.join(run_dir, "depth_radial_left.npy"), radial_left)
    _save_png(_depth_preview(axial_left), os.path.join(run_dir, "depth_preview.png"))


def _read_camera_intrinsics(stage, camera_path):
    """Pinhole intrinsics from a USD camera's focal length and aperture.

    Args:
        stage: USD stage.
        camera_path: Camera prim path.

    Returns:
        dict with fx, fy, cx, cy, source, raw (focal_length,
        horizontal_aperture, horizontal_fov_rad).
    """
    from pxr import UsdGeom

    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    focal_length = camera.GetFocalLengthAttr().Get()
    horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
    horizontal_fov_rad = 2 * math.atan(horizontal_aperture / (2 * focal_length))
    intrinsics = _intrinsics_from_fov(WIDTH, HEIGHT, horizontal_fov_rad)
    # _intrinsics_from_fov() labels its own source "geometric" regardless of
    # where the FOV came from -- relabel it here with the true provenance.
    intrinsics["source"] = "usd_camera_focal_length_and_aperture"
    intrinsics["raw"] = {
        "focal_length": focal_length,
        "horizontal_aperture": horizontal_aperture,
        "horizontal_fov_rad": horizontal_fov_rad,
    }
    return intrinsics


def _build_measurement(stage, axial_left, left_frame):
    """Build the measurement dict for the nearest target beyond the self-body envelope.

    Args:
        stage: USD stage.
        axial_left: Left-eye axial depth array.
        left_frame: Left camera pose (a _camera_frame() return value).

    Returns:
        dict with target_pixel_px, camera_to_target_distance_m/cm,
        target_camera_xyz_m, target_world_xyz_m, intrinsics,
        self_body_exclusion. The xyz/distance fields are None when no
        target pixel was found.
    """
    intrinsics = _read_camera_intrinsics(stage, CAMERA_PATHS["left"])
    selection = _select_target_surface_pixel(axial_left)
    if selection["target_pixel_px"] is None:
        return {
            "target_pixel_px": None,
            "camera_to_target_distance_m": None,
            "camera_to_target_distance_cm": None,
            "target_camera_xyz_m": None,
            "target_world_xyz_m": None,
            "intrinsics": intrinsics,
            "self_body_exclusion": selection,
        }
    u, v = selection["target_pixel_px"]
    depth_m = float(axial_left[v, u])
    camera_xyz = _pixel_depth_to_camera_xyz(u, v, depth_m, intrinsics)
    distance_m = float(np.linalg.norm(camera_xyz))
    return {
        "target_pixel_px": [u, v],
        "camera_to_target_distance_m": distance_m,
        "camera_to_target_distance_cm": distance_m * 100.0,
        "target_camera_xyz_m": camera_xyz,
        "target_world_xyz_m": _camera_to_world_xyz(camera_xyz, left_frame),
        "intrinsics": intrinsics,
        "self_body_exclusion": selection,
    }
