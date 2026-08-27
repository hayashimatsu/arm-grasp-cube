"""Grasp execution: moves /World/IKTarget to a target pose, closes the
gripper, and lifts.

grasp_object() moves to a caller-supplied surface centre; grasp_from_capture()
reads the most recent demo_capture() result and picks the cube-sized
candidate. Requires demo_start.py to already be running the IK follow
controller -- this module never starts or touches it.

This environment cannot block-wait on physics inside one call
(event-loop reentrancy). grasp_object(..., dry_run=False) schedules an
async task and returns immediately; poll grasp_object_status() for the
result.
"""

from __future__ import annotations

import asyncio
import builtins
import glob
import json
import math
import os
import sys

GRASP_Z_OFFSET_M = 0.0021
TARGET_QUAT_WXYZ = [
    -4.4930672515874903e-07,
    0.7071068286895752,
    0.7071067094802856,
    4.984278234587691e-07,
]
OBJECT_PRIM_PATH = "/World/object"
IK_TARGET_PRIM_PATH = "/World/IKTarget"
# Convergence must be judged against the TCP prim ik_controller.py actually
# drives, not against IKTarget's own commanded pose -- that prim is set
# directly by _pose() and reads back as "arrived" on frame 0, before the arm
# (and hence the grasped object) has actually had time to move.
TCP_PRIM_PATH = "/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/base_link/tcp"
if "__file__" not in globals():
    raise RuntimeError("grasp_demo.py must be exec'd with __file__ set (see demo_api.py._grasp_demo_ns())")
GRIPPER_CONTROLLER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_controller.py")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_math  # noqa: E402

QUAT_CONVERGENCE_TOL_DEG = 0.5
MAX_FRAMES_WITH_YAW = 800
GRIPPER_OPEN_DEG = 0.0
GRIPPER_CLOSE_DEG = 47.0
CLOSE_STEPS = 9
CONTACT_THRESHOLD_RAD = 0.01
# World-axis-aligned surface_extent_world_m is NOT rotation-invariant -- a
# 50 mm cube yawed 20 deg has an AABB edge of 50*(cos+sin) = 64 mm.
# height_above_plane_m is the one quantity yaw never touches, so it is the
# primary rule; the horizontal footprint is only checked as a ratio to it.
HEIGHT_RANGE_M = (0.040, 0.060)
# sqrt(2) =~ 1.4142 is the theoretical AABB-diagonal/edge ratio for a square
# footprint at 45 deg yaw (the worst case); 1.50 adds measurement margin on
# top of that. Do not raise this past 1.50.
RATIO_RANGE = (0.85, 1.50)
HELD_FRACTION = 0.05  # |rise_error_mm| within this fraction of lift_mm -> held; above it (positive side) -> ejected
DROPPED_FRACTION = 0.15  # rise_error_mm below -this fraction -> dropped; between -DROPPED and -HELD -> partial_slip
_REGISTRY_ATTR = "_nhc12_grasp_demo_registry"


def _classify_grasp_outcome(lift_mm, object_rise_mm, contact_made):
    """Classify a lift attempt into one of five outcomes.

    Args:
        lift_mm: Commanded lift distance, mm.
        object_rise_mm: Measured object rise, mm.
        contact_made: Whether the gripper detected contact during closing.

    Returns:
        (outcome, rise_error_mm). outcome is one of: "no_contact" (no
        contact detected), "held" (|rise_error_mm| within HELD_FRACTION of
        lift_mm), "ejected" (rise_error_mm above that), "partial_slip"
        (rise_error_mm negative but within DROPPED_FRACTION), "dropped"
        (below that). rise_error_mm is object_rise_mm - lift_mm, or None
        when contact_made is False.
    """
    if not contact_made:
        return "no_contact", None
    rise_error_mm = object_rise_mm - lift_mm
    if rise_error_mm > HELD_FRACTION * lift_mm:
        outcome = "ejected"
    elif abs(rise_error_mm) <= HELD_FRACTION * lift_mm:
        outcome = "held"
    elif rise_error_mm >= -DROPPED_FRACTION * lift_mm:
        outcome = "partial_slip"
    else:
        outcome = "dropped"
    return outcome, rise_error_mm


def _registry():
    """This module's builtins-backed state.

    Returns:
        dict with "task" (the running grasp asyncio.Task, or None) and
        "last_result" (the most recent grasp result dict, or None).
    """
    if not hasattr(builtins, _REGISTRY_ATTR):
        setattr(builtins, _REGISTRY_ATTR, {"task": None, "last_result": None})
    return getattr(builtins, _REGISTRY_ATTR)


GRIPPER_FUNCS_REGISTRY_ATTR = "_nhc12_gripper_funcs_registry"


def _gripper_funcs():
    """The shared gripper_set_angle_deg/gripper_status function pair.

    Returns:
        (gripper_set_angle_deg, gripper_status), preferring the instance
        published in builtins by demo_start.py (so gripper state stays
        shared across namespaces); falls back to a private exec of
        gripper_controller.py if that was never published.
    """
    shared = getattr(builtins, GRIPPER_FUNCS_REGISTRY_ATTR, None)
    if shared is not None:
        return shared["gripper_set_angle_deg"], shared["gripper_status"]
    ns = {}
    exec(compile(open(GRIPPER_CONTROLLER_PATH, encoding="utf-8").read(), GRIPPER_CONTROLLER_PATH, "exec"), ns)
    return ns["gripper_set_angle_deg"], ns["gripper_status"]


def _target_pose(surface_center_world_xyz, height_above_plane_m):
    """The IKTarget xyz for grasping an object at this surface centre.

    Args:
        surface_center_world_xyz: World xyz of the object's surface centre.
        height_above_plane_m: Object height above the table plane, metres.

    Returns:
        [x, y, z] target position, offset down by half the object's height
        plus GRASP_Z_OFFSET_M.
    """
    x, y, z = surface_center_world_xyz
    z_offset = -(height_above_plane_m / 2.0) - GRASP_Z_OFFSET_M
    return [x, y, z + z_offset]


def _pose(path, set_xyz=None, set_quat=None, want_quat=False):
    """Read or write the world pose of a prim.

    Args:
        path: Prim path.
        set_xyz: If given, write this world position (and set_quat, or
            TARGET_QUAT_WXYZ if set_quat is None). If None, read instead.
        set_quat: Orientation to write with set_xyz; ignored when reading.
        want_quat: When reading, also return the orientation.

    Returns:
        When writing: set_xyz. When reading: position [x, y, z], or
        (position, quaternion_wxyz) if want_quat is True.
    """
    import numpy as np
    from isaacsim.core.experimental.prims import XformPrim
    import isaacsim.core.experimental.utils.backend as backend_utils

    prim = XformPrim(path)
    with backend_utils.use_backend("fabric", raise_on_fallback=True):
        if set_xyz is None:
            pos, quat = prim.get_world_poses()
            if want_quat:
                return pos.numpy()[0].tolist(), quat.numpy()[0].tolist()
            return pos.numpy()[0].tolist()
        quat_to_write = set_quat if set_quat is not None else TARGET_QUAT_WXYZ
        prim.set_world_poses(np.array([set_xyz]), np.array([quat_to_write]))
        return set_xyz


async def _wait_for_convergence(target_xyz, tol_mm=1.0, max_frames=300, target_quat=None, quat_tol_deg=None):
    """Wait for the TCP to reach a target position (and orientation).

    Args:
        target_xyz: Target world position.
        tol_mm: Position convergence tolerance, mm.
        max_frames: Give up after this many frames.
        target_quat: Target orientation (w, x, y, z); orientation is not
            checked when None.
        quat_tol_deg: Orientation convergence tolerance, degrees.

    Returns:
        (achieved_xyz, final_distance_mm) at convergence or timeout.
    """
    import omni.kit.app

    app = omni.kit.app.get_app()
    achieved, achieved_quat = _pose(TCP_PRIM_PATH, want_quat=True)
    for _ in range(max_frames):
        achieved, achieved_quat = _pose(TCP_PRIM_PATH, want_quat=True)
        pos_ok = math.dist(achieved, target_xyz) * 1000.0 <= tol_mm
        quat_ok = target_quat is None or pose_math.quat_angle_deg(achieved_quat, target_quat) <= quat_tol_deg
        if pos_ok and quat_ok:
            break
        await app.next_update_async()
    return achieved, math.dist(achieved, target_xyz) * 1000.0


def _close_lift_result(lift_mm, contact_angle_deg=None, contact_step_size_deg=None,
                        max_tracking_error_rad=0.0, object_rise_mm=None, rise_error_mm=None,
                        slip_mm=None, outcome="no_contact", contact_made=False,
                        lift_held=False, error=None):
    """Build a close/lift result dict.

    Args:
        lift_mm: Commanded lift distance, mm.
        contact_angle_deg: Gripper angle at which contact was detected.
        contact_step_size_deg: Gripper angle step size used while closing.
        max_tracking_error_rad: Largest gripper tracking error observed.
        object_rise_mm: Measured object rise, mm.
        rise_error_mm: object_rise_mm - lift_mm.
        slip_mm: abs(lift_mm - object_rise_mm); kept for backward
            compatibility, no diagnostic value -- read outcome instead.
        outcome: One of _classify_grasp_outcome's five outcome strings.
        contact_made: Whether the gripper detected contact.
        lift_held: Whether the lift is judged to have held.
        error: Error message, if any.

    Returns:
        dict with exactly these fields.

    Note:
        This is the single declaration of the close/lift result's field
        set. error defaults to None so it is always present in every path
        that builds a result through this function.
    """
    return dict(
        contact_angle_deg=contact_angle_deg, contact_step_size_deg=contact_step_size_deg,
        max_tracking_error_rad=max_tracking_error_rad, ik_target_rise_mm=lift_mm,
        object_rise_mm=object_rise_mm, rise_error_mm=rise_error_mm,
        slip_mm=slip_mm,  # kept for backward compatibility; no diagnostic value -- read outcome
        outcome=outcome, contact_made=contact_made, lift_held=lift_held, error=error,
    )


async def _close_and_lift(lift_mm):
    """Close the gripper in steps, detect contact, lift, and classify the outcome.

    Args:
        lift_mm: Distance to lift /World/IKTarget after contact, mm.

    Returns:
        dict, see _close_lift_result().

    Note:
        Assumes /World/IKTarget is already at the grasp position when
        called. Catches its own exceptions so a mid-close-loop failure
        still reports whatever partial contact/tracking state had been
        observed, via the returned "error" key.
    """
    import omni.kit.app

    gripper_set_angle_deg, gripper_status = _gripper_funcs()
    app = omni.kit.app.get_app()
    error = None
    contact_made = False
    contact_angle_deg = None
    max_tracking_error_rad = 0.0
    object_rise_mm = slip_mm = rise_error_mm = None
    outcome = "no_contact"
    lift_held = False
    try:
        object_before = _pose(OBJECT_PRIM_PATH)
        for step in range(1, CLOSE_STEPS + 1):
            deg = GRIPPER_CLOSE_DEG * step / CLOSE_STEPS
            gripper_set_angle_deg(deg)
            for _ in range(60):
                await app.next_update_async()
            tracking_err = abs(gripper_status()["tracking_error_rad"] or 0.0)
            max_tracking_error_rad = max(max_tracking_error_rad, tracking_err)
            if not contact_made and tracking_err >= CONTACT_THRESHOLD_RAD:
                contact_made, contact_angle_deg = True, deg
        contact_angle_deg = contact_angle_deg or gripper_status()["achieved_deg"]

        # A gripper that closed on nothing has no object to lift -- skip
        # straight to no_contact instead of lifting an empty hand.
        if contact_made:
            ik_before = _pose(IK_TARGET_PRIM_PATH)
            lift_target = [ik_before[0], ik_before[1], ik_before[2] + lift_mm / 1000.0]
            _pose(IK_TARGET_PRIM_PATH, lift_target)
            await _wait_for_convergence(lift_target)

            object_rise_mm = (_pose(OBJECT_PRIM_PATH)[2] - object_before[2]) * 1000.0
            slip_mm = abs(lift_mm - object_rise_mm)
        outcome, rise_error_mm = _classify_grasp_outcome(lift_mm, object_rise_mm, contact_made)
        lift_held = contact_made and object_rise_mm is not None and object_rise_mm >= (lift_mm * 0.85) and slip_mm <= 10.0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return _close_lift_result(
        lift_mm, contact_angle_deg=contact_angle_deg, contact_step_size_deg=GRIPPER_CLOSE_DEG / CLOSE_STEPS,
        max_tracking_error_rad=max_tracking_error_rad, object_rise_mm=object_rise_mm,
        rise_error_mm=rise_error_mm, slip_mm=slip_mm, outcome=outcome,
        contact_made=contact_made, lift_held=lift_held, error=error,
    )


async def _run_grasp_async(surface_center_world_xyz, height_above_plane_m, lift_mm):
    """Trace to a surface centre, then close and lift.

    Args:
        surface_center_world_xyz: World xyz of the object's surface centre.
        height_above_plane_m: Object height above the table plane, metres.
        lift_mm: Distance to lift after contact, mm.

    Returns:
        dict with target_xyz, ik_convergence_error_mm, error, and the
        fields from _close_lift_result().
    """
    target_xyz = _target_pose(surface_center_world_xyz, height_above_plane_m)
    error = None
    ik_convergence_error_mm = None
    try:
        gripper_set_angle_deg, _ = _gripper_funcs()
        gripper_set_angle_deg(GRIPPER_OPEN_DEG)
        _pose(IK_TARGET_PRIM_PATH, target_xyz)
        _, ik_convergence_error_mm = await _wait_for_convergence(target_xyz)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    close_lift = await _close_and_lift(lift_mm) if error is None else _close_lift_result(lift_mm)
    error = error or close_lift.pop("error")

    result = dict(
        target_xyz=target_xyz, ik_convergence_error_mm=ik_convergence_error_mm,
        error=error, **close_lift,
    )
    _registry()["last_result"] = result
    return result


def grasp_object(surface_center_world_xyz, height_above_plane_m, lift_mm=70.0, dry_run=False):
    """Grasp an object at a surface centre: trace, close, and lift.

    Args:
        surface_center_world_xyz: World xyz of the object's surface centre.
        height_above_plane_m: Object height above the table plane, metres.
        lift_mm: Distance to lift after contact, mm.
        dry_run: If True, compute and return the target pose without moving.

    Returns:
        When dry_run: dict with target_xyz, target_quat_wxyz, dry_run=True.
        Otherwise: {"status": "scheduled"} or {"status": "already_scheduled"};
        poll grasp_object_status() for the result.
    """
    target_xyz = _target_pose(surface_center_world_xyz, height_above_plane_m)
    if dry_run:
        return {"target_xyz": target_xyz, "target_quat_wxyz": TARGET_QUAT_WXYZ, "dry_run": True}

    registry = _registry()
    task = registry.get("task")
    if task is not None and not task.done():
        return {"status": "already_scheduled"}
    registry["last_result"] = None
    registry["task"] = asyncio.ensure_future(
        _run_grasp_async(surface_center_world_xyz, height_above_plane_m, lift_mm)
    )
    return {"status": "scheduled"}


def grasp_object_status():
    """Status of the most recent grasp_object() call.

    Returns:
        {"status": "running"} while in progress, otherwise the result
        dict from grasp_object(), or {"status": "not_run"}.
    """
    registry = _registry()
    task = registry.get("task")
    if task is not None and not task.done():
        return {"status": "running"}
    return registry.get("last_result") or {"status": "not_run"}


def _latest_result_json_path():
    """Path to the most recently written capture result.json.

    Returns:
        Path string. Raises RuntimeError if none is found.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage_path = stage.GetRootLayer().realPath or stage.GetRootLayer().identifier
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(stage_path)))
    output_root = os.environ.get(
        "D455_DEMO_OUTPUT_DIR", os.path.join(project_root, "outputs", "captures")
    )
    candidates = sorted(
        glob.glob(os.path.join(output_root, "*", "result.json")), key=os.path.getmtime
    )
    if not candidates:
        raise RuntimeError(f"no result.json found under {output_root}")
    return candidates[-1]


def _cube_fit_diagnosis(obj):
    """Rotation-invariant test for whether a detected object fits the target cube.

    Args:
        obj: One tabletop_objects entry from a capture result.json.

    Returns:
        (fits, detail). detail is a one-line string with the measured
        values and, on failure, which rule rejected it.
    """
    height = obj["height_above_plane_m"]
    extent = obj["surface_extent_world_m"]
    ratio = (max(extent["x"], extent["y"]) / height) if height else float("inf")
    stats = f"height={height:.4f} ratio={ratio:.2f}"
    if not (HEIGHT_RANGE_M[0] <= height <= HEIGHT_RANGE_M[1]):
        return False, f"{stats} -> rule 1 (height not in {HEIGHT_RANGE_M})"
    if not (RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]):
        return False, f"{stats} -> rule 2 (ratio not in {RATIO_RANGE})"
    if obj.get("touches_border", False):
        return False, f"{stats} -> rule 3 (touches_border)"
    if obj.get("center_reliability") != "ok":
        return False, f"{stats} -> rule 4 (center_reliability={obj.get('center_reliability')!r})"
    return True, f"{stats} -> ok"


def _fits_cube_extent(obj):
    """Whether a detected object fits the target cube.

    Args:
        obj: One tabletop_objects entry from a capture result.json.

    Returns:
        bool.
    """
    return _cube_fit_diagnosis(obj)[0]


def grasp_from_capture(result_json_path=None):
    """Grasp the single cube-sized candidate from the latest capture result.

    Args:
        result_json_path: Path to a capture result.json; defaults to the
            most recently written one.

    Returns:
        See grasp_object(). Raises RuntimeError if the candidate count is
        not exactly 1.
    """
    path = result_json_path or _latest_result_json_path()
    objects = json.load(open(path, encoding="utf-8"))["tabletop_objects"]["objects"]
    candidates = [obj for obj in objects if _fits_cube_extent(obj)]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly 1 red-cube-sized candidate, found {len(candidates)} of {len(objects)} in {path}"
        )
    obj = candidates[0]
    return grasp_object(obj["surface_center_world_xyz_m"], obj["height_above_plane_m"])
