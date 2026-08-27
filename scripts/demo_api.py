"""Release API: demo_capture() / demo_trace() / demo_grasp() / demo_run() /
demo_status().

This environment cannot block-wait on physics inside one call
(event-loop reentrancy). Every action function schedules an async task and
returns immediately; poll demo_status() for the result.

This module's demo_capture() overwrites capture_d455.py's same-named raw
capture function in the shared globals it is exec'd into.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import math
import os
import time

_project_root_fn = globals().get("_demo_project_root")
if _project_root_fn is None:
    raise RuntimeError("demo_start.py must be exec'd before demo_api.py (missing _demo_project_root)")
GRASP_DEMO_PATH = os.path.join(_project_root_fn(), "scripts", "grasp_demo.py")

_run_three_phase_capture = globals().get("run_three_phase_capture")
if _run_three_phase_capture is None:
    raise RuntimeError("capture_approach.py must be exec'd before demo_api.py")


def _grasp_demo_ns():
    """Exec grasp_demo.py into a fresh, isolated namespace.

    Returns:
        dict: the resulting namespace, including grasp_demo.py's module
        globals (functions, constants).
    """
    ns = {"__file__": GRASP_DEMO_PATH}
    src = open(GRASP_DEMO_PATH, encoding="utf-8").read()
    exec(compile(src, GRASP_DEMO_PATH, "exec"), ns)
    return ns


_GD = _grasp_demo_ns()
IK_TARGET_PRIM_PATH = _GD["IK_TARGET_PRIM_PATH"]
# The Phase 1 survey shot for capture_approach.py's three-phase capture.
# z is near the practical reachability ceiling for this arm -- do not raise
# further without re-verifying reachability live.
SURVEY_POSE = {"xyz": [-0.293832004070282, 0.03997138515114784, 1.1360009789466858], "quat_wxyz": _GD["TARGET_QUAT_WXYZ"]}
_REGISTRY_ATTR = "_nhc12_demo_api_registry"


def _api_registry():
    """This module's builtins-backed state (current stage, running task,
    last result).

    Returns:
        dict with stage, task, ok, result, error, last_capture,
        trace_target_xyz, trace_target_quat.
    """
    if not hasattr(builtins, _REGISTRY_ATTR):
        setattr(builtins, _REGISTRY_ATTR, dict(stage="idle", task=None, ok=None,
                result=None, error=None, last_capture=None, trace_target_xyz=None,
                trace_target_quat=None))
    return getattr(builtins, _REGISTRY_ATTR)


DRIFT_POS_TOL_MM = 5.0


def _drift_diagnosis(current_xyz, target_xyz, current_quat, target_quat,
                      pos_tol_mm=DRIFT_POS_TOL_MM, quat_tol_deg=None):
    """Check whether the IK target has drifted from a previously traced pose.

    Args:
        current_xyz: Current IK target world position.
        target_xyz: Previously traced target position.
        current_quat: Current IK target orientation (w, x, y, z).
        target_quat: Previously traced target orientation (w, x, y, z).
        pos_tol_mm: Position drift tolerance, mm.
        quat_tol_deg: Orientation drift tolerance, degrees; defaults to
            _GD["QUAT_CONVERGENCE_TOL_DEG"] (the same tolerance
            demo_trace()'s own convergence check uses).

    Returns:
        (drifted, message). message states which of position/orientation
        drifted and by how much; never a coordinate-range suggestion.

    Note:
        Position and orientation drift are checked independently so the
        message says which one drifted and by how much.
    """
    quat_tol_deg = _GD["QUAT_CONVERGENCE_TOL_DEG"] if quat_tol_deg is None else quat_tol_deg
    dist_mm = math.dist(current_xyz, target_xyz) * 1000.0
    quat_err_deg = _GD["pose_math"].quat_angle_deg(current_quat, target_quat)
    pos_bad, quat_bad = dist_mm > pos_tol_mm, quat_err_deg > quat_tol_deg
    if not pos_bad and not quat_bad:
        return False, None
    parts = []
    if pos_bad:
        parts.append(f"position drifted {dist_mm:.2f} mm (> {pos_tol_mm} mm)")
    if quat_bad:
        parts.append(f"orientation drifted {quat_err_deg:.3f} deg (> {quat_tol_deg} deg)")
    return True, "IK target drifted from the traced pose: " + "; ".join(parts)


def _finish(ok, result, error):
    """Record the outcome of the running action into the registry.

    Args:
        ok: Whether the action succeeded.
        result: Result dict, or None on failure.
        error: Error message, or None on success.
    """
    reg = _api_registry()
    reg["ok"], reg["result"], reg["error"] = ok, result, error


async def _run_capture_async():
    """Run the three-phase capture (survey, approach, measure) and record it.

    Returns:
        dict with surface_center_world_xyz_m, height_above_plane_m,
        surface_extent_world_m, approach_steps_taken, survey_pose_used,
        final_pose_used, px_offset_history. None on failure (see
        _api_registry()["error"]).
    """
    error = None
    result = None
    try:
        approach_result = await _run_three_phase_capture(
            SURVEY_POSE["xyz"], _GD["_pose"], _GD["_wait_for_convergence"],
            _GD["_cube_fit_diagnosis"], IK_TARGET_PRIM_PATH,
        )
        result = {
            "surface_center_world_xyz_m": approach_result["surface_center_world_xyz_m"],
            "height_above_plane_m": approach_result["height_above_plane_m"],
            "surface_extent_world_m": approach_result["surface_extent_world_m"],
            "approach_steps_taken": approach_result["approach_steps_taken"],
            "survey_pose_used": approach_result["survey_pose_used"],
            "final_pose_used": approach_result["final_pose_used"],
            "px_offset_history": approach_result["px_offset_history"],
        }
        _api_registry()["last_capture"] = result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _finish(error is None, result, error)
    return result


def demo_capture():
    """Schedule a tabletop capture. Raises RuntimeError if the gripper is
    not fully open.

    Returns:
        {"ok": True, "state": "scheduled", "stage": "capture"}. Poll
        demo_status() for the result.
    """
    _, gripper_status = _GD["_gripper_funcs"]()
    achieved_deg = gripper_status()["achieved_deg"]
    if achieved_deg > 2.0:
        raise RuntimeError(
            f"gripper not fully open (achieved_deg={achieved_deg:.2f} > 2.0 deg); "
            "parking now would drag whatever it's holding"
        )
    reg = _api_registry()
    reg.update(stage="capture", ok=None, result=None, error=None, last_capture=None,
               trace_target_xyz=None, trace_target_quat=None)
    reg["task"] = asyncio.ensure_future(_run_capture_async())
    return {"ok": True, "state": "scheduled", "stage": "capture"}


def _grasp_yaw_for_trace(surface_center_xyz):
    """Look up the grasp yaw for the traced object from the latest capture.

    Args:
        surface_center_xyz: World xyz of the object's surface centre, used
            to match it against the capture's tabletop_objects.

    Returns:
        (yaw_applied_deg, yaw_source).

    Note:
        yaw_source is one of:
          "capture"       Normal case: yaw_applied_deg is grip_yaw_deg
                           folded into the object's own symmetry period.
          "indeterminate" symmetry_period_deg is None (round/near-circular
                           object): yaw is meaningless, not an error.
          "none"          The field could not be read at all. Always
                           printed; yaw_applied_deg is 0.0, not a guess.
    """
    try:
        path = _GD["_latest_result_json_path"]()
        objects = json.load(open(path, encoding="utf-8"))["tabletop_objects"]["objects"]
    except Exception as exc:
        print(f"demo_trace: no result.json to read yaw from ({type(exc).__name__}: {exc}); yaw_source=none")
        return 0.0, "none"

    match = next(
        (o for o in objects if o.get("surface_center_world_xyz_m") is not None
         and math.dist(o["surface_center_world_xyz_m"], surface_center_xyz) < 1e-6),
        None,
    )
    if match is None or "grip_yaw_deg" not in match:
        print("demo_trace: traced object not found in result.json (or missing grip_yaw_deg); yaw_source=none")
        return 0.0, "none"

    symmetry_period_deg, grip_yaw_deg = match.get("symmetry_period_deg"), match.get("grip_yaw_deg")
    if symmetry_period_deg is None or grip_yaw_deg is None:
        print(f"demo_trace: object is round/indeterminate (shape_class={match.get('shape_class')}); "
              f"yaw_applied_deg=0.0, yaw_source=indeterminate")
        return 0.0, "indeterminate"

    return _GD["pose_math"].wrap_to_symmetry(grip_yaw_deg, symmetry_period_deg), "capture"


async def _run_trace_async(target_xyz, target_quat, yaw_applied_deg, yaw_source):
    """Move the IK target to a pose and wait for convergence.

    Args:
        target_xyz: Target world position.
        target_quat: Target orientation (w, x, y, z).
        yaw_applied_deg: Yaw applied, for the result dict.
        yaw_source: Yaw source, for the result dict.

    Returns:
        dict with target_xyz, achieved_xyz, ik_convergence_error_mm,
        pose_reached, yaw_applied_deg, yaw_source. None on failure.
    """
    error = None
    result = None
    try:
        gripper_set_angle_deg, _ = _GD["_gripper_funcs"]()
        gripper_set_angle_deg(0.0)
        _GD["_pose"](IK_TARGET_PRIM_PATH, target_xyz, set_quat=target_quat)
        achieved_xyz, err_mm = await _GD["_wait_for_convergence"](
            target_xyz, target_quat=target_quat, quat_tol_deg=_GD["QUAT_CONVERGENCE_TOL_DEG"],
            max_frames=_GD["MAX_FRAMES_WITH_YAW"],
        )
        result = {
            "target_xyz": target_xyz, "achieved_xyz": achieved_xyz,
            "ik_convergence_error_mm": err_mm, "pose_reached": err_mm <= 5.0,
            "yaw_applied_deg": yaw_applied_deg, "yaw_source": yaw_source,
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _finish(error is None, result, error)
    return result


def demo_trace(surface_center_xyz=None, thickness_m=None):
    """Schedule moving the IK target to grasp position above an object.

    Args:
        surface_center_xyz: World xyz of the object's surface centre;
            defaults to the most recent demo_capture() result.
        thickness_m: Object height above the table plane, metres; defaults
            to the most recent demo_capture() result.

    Returns:
        {"ok": True, "state": "scheduled", "stage": "trace"}. Poll
        demo_status() for the result. Raises RuntimeError if no arguments
        are given and there is no usable prior capture.
    """
    reg = _api_registry()
    if surface_center_xyz is None or thickness_m is None:
        cap = reg.get("last_capture")
        if cap is None:
            raise RuntimeError(
                "most recent capture failed or does not exist; call demo_capture() again "
                "(or pass surface_center_xyz and thickness_m explicitly)"
            )
        surface_center_xyz = cap["surface_center_world_xyz_m"] if surface_center_xyz is None else surface_center_xyz
        thickness_m = cap["height_above_plane_m"] if thickness_m is None else thickness_m
    target_xyz = _GD["_target_pose"](surface_center_xyz, thickness_m)
    yaw_applied_deg, yaw_source = _grasp_yaw_for_trace(surface_center_xyz)
    target_quat = _GD["pose_math"].quat_down_with_yaw(yaw_applied_deg)
    reg.update(stage="trace", ok=None, result=None, error=None,
               trace_target_xyz=target_xyz, trace_target_quat=target_quat)
    reg["task"] = asyncio.ensure_future(_run_trace_async(target_xyz, target_quat, yaw_applied_deg, yaw_source))
    return {"ok": True, "state": "scheduled", "stage": "trace"}


async def _run_grasp_stage_async(lift_mm):
    """Close the gripper, lift, and record the outcome.

    Args:
        lift_mm: Distance to lift after contact, mm.

    Returns:
        dict, see grasp_demo._close_lift_result().

    Note:
        result stays None on any failure; the error is surfaced separately
        via _finish.
    """
    error = None
    result = None
    try:
        close_lift = await _GD["_close_and_lift"](lift_mm)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = close_lift.pop("error")
        if error is None:
            result = close_lift
    _finish(error is None, result, error)
    return result


def demo_grasp(lift_mm=100.0):
    """Schedule closing the gripper and lifting.

    Args:
        lift_mm: Distance to lift after contact, mm.

    Returns:
        {"ok": True, "state": "scheduled", "stage": "grasp"}. Poll
        demo_status() for the result. Raises RuntimeError if there is no
        demo_trace() target, or if the IK target has drifted from it.
    """
    reg = _api_registry()
    target_xyz = reg.get("trace_target_xyz")
    target_quat = reg.get("trace_target_quat")
    if target_xyz is None or target_quat is None:
        raise RuntimeError("no demo_trace() target recorded; call demo_trace() first")
    current_xyz, current_quat = _GD["_pose"](IK_TARGET_PRIM_PATH, want_quat=True)
    drifted, message = _drift_diagnosis(current_xyz, target_xyz, current_quat, target_quat)
    if drifted:
        raise RuntimeError(message)
    reg.update(stage="grasp", ok=None, result=None, error=None)
    reg["task"] = asyncio.ensure_future(_run_grasp_stage_async(lift_mm))
    return {"ok": True, "state": "scheduled", "stage": "grasp"}


def demo_release():
    """Open the gripper to 0 deg; the held object falls under gravity.

    Synchronous -- only sets the drive target, does not wait for physics.

    Returns:
        {"ok": True, "angle_deg": 0.0}.

    Note:
        Clears the recorded demo_trace() target, so calling demo_grasp()
        without a fresh demo_trace() first raises.
    """
    gripper_set_angle_deg, _ = _GD["_gripper_funcs"]()
    result = gripper_set_angle_deg(0.0)
    _api_registry().update(trace_target_xyz=None, trace_target_quat=None)
    return {"ok": True, "angle_deg": result["target_deg"]}


def demo_gripper_angle(deg):
    """Directly command the gripper angle. Synchronous.

    Args:
        deg: Target angle in degrees, within [0, 47]. Raises ValueError
            outside that range.

    Returns:
        dict with target_deg and target_rad.
    """
    return _GD["_gripper_funcs"]()[0](deg)


def demo_gripper_status():
    """Current gripper joint angle, target tracking error, and fingertip poses.

    Returns:
        dict with target_deg, achieved_deg, achieved_rad,
        finger_origin_separation_mm, tracking_error_rad, fingertip_a_world,
        fingertip_b_world.
    """
    return _GD["_gripper_funcs"]()[1]()


# Fixed pause between phases for the object/arm to visibly settle before
# the next phase reads a position off it -- not a convergence tolerance
# (each phase already awaits its own _wait_for_convergence).
SETTLE_SECONDS = 1.0


async def _run_demo_async(lift_mm):
    """Run capture, trace, and grasp in sequence, printing progress.

    Args:
        lift_mm: Distance to lift after contact, mm.

    Returns:
        None. Result is recorded via _finish(); see demo_status().
    """
    reg = _api_registry()
    start_s = time.monotonic()
    stage = "capture"
    try:
        print("demo_run: capture starting")
        capture_result = await _run_capture_async()
        if capture_result is None:
            _finish(False, None, f"capture: {reg['error']}")
            return
        print(f"demo_run: capture done ({capture_result['approach_steps_taken']} approach step(s))")
        await asyncio.sleep(SETTLE_SECONDS)

        stage = "trace"
        print("demo_run: trace starting")
        target_xyz = _GD["_target_pose"](
            capture_result["surface_center_world_xyz_m"], capture_result["height_above_plane_m"]
        )
        yaw_applied_deg, yaw_source = _grasp_yaw_for_trace(capture_result["surface_center_world_xyz_m"])
        target_quat = _GD["pose_math"].quat_down_with_yaw(yaw_applied_deg)
        reg["trace_target_xyz"] = target_xyz
        reg["trace_target_quat"] = target_quat
        trace_result = await _run_trace_async(target_xyz, target_quat, yaw_applied_deg, yaw_source)
        if trace_result is None:
            _finish(False, None, f"trace: {reg['error']}")
            return
        print(f"demo_run: trace done (ik_convergence_error_mm={trace_result['ik_convergence_error_mm']:.3f}, "
              f"yaw_applied_deg={yaw_applied_deg:.3f}, yaw_source={yaw_source})")
        await asyncio.sleep(SETTLE_SECONDS)

        stage = "grasp"
        print("demo_run: grasp starting")
        grasp_result = await _run_grasp_stage_async(lift_mm)
        if grasp_result is None:
            _finish(False, None, f"grasp: {reg['error']}")
            return
        print(f"demo_run: grasp done (outcome={grasp_result['outcome']})")

        _finish(True, dict(capture=capture_result, trace=trace_result, grasp=grasp_result,
                            outcome=grasp_result["outcome"]), None)
    except Exception as exc:
        _finish(False, None, f"{stage}: {type(exc).__name__}: {exc}")
    finally:
        elapsed_s = time.monotonic() - start_s
        final = _api_registry()
        if final["ok"]:
            result = final["result"] or {}
            grasp = result.get("grasp") or {}
            print(
                f"demo_run: DONE  outcome={result.get('outcome')}  "
                f"object_rise_mm={grasp.get('object_rise_mm')}  elapsed_s={elapsed_s:.1f}"
            )
        else:
            error_first_line = (final["error"] or "").splitlines()[0] if final["error"] else ""
            print(f"demo_run: FAILED  stage={stage}  error={error_first_line}")


def demo_run(lift_mm=100.0):
    """Schedule capture -> trace -> grasp as one call.

    Args:
        lift_mm: Distance to lift after contact, mm.

    Returns:
        {"ok": True, "state": "scheduled", "stage": "run"}, or
        {"ok": False, "state": "busy", ...} if another action is running.
        Poll demo_status() for the merged result.

    Note:
        Schedules a single async task that awaits all three phases in
        sequence internally -- not three separate scheduled tasks.
    """
    reg = _api_registry()
    task = reg.get("task")
    if task is not None and not task.done():
        return {"ok": False, "state": "busy", "stage": reg["stage"]}
    reg.update(stage="run", ok=None, result=None, error=None)
    reg["task"] = asyncio.ensure_future(_run_demo_async(lift_mm))
    return {"ok": True, "state": "scheduled", "stage": "run"}


def demo_status():
    """Status of the most recent action (demo_capture/trace/grasp/run).

    Returns:
        dict with stage, running, ok, result, error.
    """
    reg = _api_registry()
    task = reg.get("task")
    running = task is not None and not task.done()
    return {"stage": reg["stage"], "running": running, "ok": reg["ok"],
            "result": reg["result"], "error": reg["error"]}
