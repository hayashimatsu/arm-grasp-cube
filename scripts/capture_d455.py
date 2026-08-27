"""D455 tabletop capture for the NHC12 demo scene.

Captures RGB + depth from the three cameras, segments tabletop objects,
measures the nearest target, draws annotated overlays, and writes
result.json/diagnostics.json.

Public entry points:
    demo_capture_setup()   Await; create and cache the camera render
                            products. Normally not called directly --
                            demo_capture_async() awaits it itself.
    demo_capture(label=None)   GUI-safe: schedules one capture, returns
                                {"status": "scheduled"} immediately.
    demo_capture_status()      Result of the most recent demo_capture();
                                {"status": "capturing"} while it runs.
    demo_capture_async(label=None)   Await; the actual capture.

Two constraints this module depends on:
    This module is loaded via exec(), not import, so it never has
    __file__. _source_file_path()/_source_sha256_or_none() depend on that
    fact and must not be moved to a normally-imported module (there,
    __file__ would always be set, to the wrong file's path).

    Waiting must use rep.orchestrator.step_async(): the synchronous
    step() always raises OrchestratorError inside a Kit GUI session, and
    falling back to a plain update-loop pump loses Replicator's
    render-sync guarantee.
"""

import asyncio
import builtins
import datetime as _datetime
import hashlib
import json
import math
import os
import re
import sys
import traceback

import numpy as np

# This module is loaded via exec(compile(source, path, "exec"), globals())
# (demo_start.py), not a normal `import capture_d455` -- __file__ is never
# set in that path, so sys.path can't be derived from it.
# sys._getframe().f_code.co_filename instead reads the real path straight
# out of the code object compile() was given, with no hardcoded path.
_scripts_dir = os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
import yaw_estimate
from capture_constants import (
    CAMERA_PATHS,
    IK_TARGET_PATH,
    WIDTH,
    HEIGHT,
    SELF_BODY_MAX_CAMERA_DISTANCE_M,
)
from capture_geometry import (
    _quaternion_wxyz_to_axes,
    # _intrinsics_from_fov / _camera_to_world_xyz: nothing in this file
    # calls them directly, but capture_approach.py reads them out of this
    # module's exec'd globals via globals().get(...) (see its own guard) --
    # removing these breaks that script even though grep on this file alone
    # says they're "unused".
    _intrinsics_from_fov,
    _camera_to_world_xyz,
    _camera_frame,
    _arm_base_world_xyz,
)
from capture_annotate import (
    _annotate_rgb_left,
    _annotate_rgb_color,
    _annotate_depth_left,
    _annotate_region_overlay,
)
from capture_run import (
    _registry,
    _create_unique_run_dir,
    _sha256,
    _active_stage_context,
    _loaded_source_sha256_at_bootstrap,
    _collect_failures,
)
from capture_segment import _build_tabletop_objects
from capture_result import (
    _save_capture_products,
    _green_audit_eye,
    _build_measurement,
    _read_camera_intrinsics,
    _build_result_json,
    _build_diagnostics_json,
)


def _source_file_path():
    """Best-effort path to this file on disk.

    Returns:
        __file__ if set; otherwise a path derived from the active stage's
        project root. None if neither works. Never raises.
    """
    path = globals().get("__file__")
    if path and os.path.isfile(path):
        return path
    try:
        _, _, stage_path, _ = _active_stage_context()
        candidate = os.path.join(os.path.dirname(os.path.dirname(stage_path)), "scripts", "capture_d455.py")
        return candidate if os.path.isfile(candidate) else None
    except Exception:
        return None


def _source_sha256_or_none():
    """The sha256 of this file's current contents on disk.

    Returns:
        Hex digest string, or None if the path could not be determined or
        read.
    """
    path = _source_file_path()
    try:
        return _sha256(path) if path else None
    except Exception:
        return None


async def _force_render_step(rt_subframes=4):
    """Advance rendering by one synchronized Replicator step.

    Args:
        rt_subframes: Real-time subframes to render.

    Returns:
        dict with method ("orchestrator_step_async" or
        "app_update_fallback") and exception (None on the fast path).
    """
    import omni.kit.app

    try:
        import omni.replicator.core as rep

        await rep.orchestrator.step_async(
            rt_subframes=rt_subframes,
            pause_timeline=False,
        )
        return {"method": "orchestrator_step_async", "exception": None}
    except Exception as exc:
        # step_async() should not raise here, but the fallback is kept
        # intentionally so a capture never hard-fails if Replicator's async
        # path changes underneath us. The exception type and message are
        # recorded rather than swallowed, so a silent regression to blind
        # pumping stays observable.
        app = omni.kit.app.get_app()
        for _ in range(max(1, rt_subframes)):
            await app.next_update_async()
        return {
            "method": "app_update_fallback",
            "exception": f"{type(exc).__name__}: {exc}",
        }


async def _read_rgb_with_retry(annotator, max_retries=4, minimum_mean=1.0):
    """Read an RGB annotator, retrying (with a render step) if it's too dark.

    Args:
        annotator: RGB Replicator annotator.
        max_retries: Maximum retry attempts.
        minimum_mean: Minimum acceptable mean pixel value.

    Returns:
        (image, diagnostics). diagnostics has mean_at_each_attempt,
        retries, valid.
    """
    attempts = []
    image = None
    for attempt in range(max_retries + 1):
        image = np.asarray(annotator.get_data())[:, :, :3].astype(np.uint8)
        mean = float(image.mean())
        attempts.append(mean)
        if mean >= minimum_mean:
            return image, {
                "mean_at_each_attempt": attempts,
                "retries": attempt,
                "valid": True,
            }
        await _force_render_step()
    return image, {
        "mean_at_each_attempt": attempts,
        "retries": max_retries,
        "valid": False,
    }


async def _wait_for_ik_settle(max_frames=180, stable_frames=5):
    """Wait for the running IK follow controller to reach its target.

    Args:
        max_frames: Give up after this many frames.
        stable_frames: Consecutive converged frames required to settle.

    Returns:
        dict with settled, frames, controller_status, and reason (on
        failure to converge). If no IK controller is active, only a short
        render warm-up is performed and settled is True.
    """
    import omni.kit.app

    app = omni.kit.app.get_app()
    controller_registry = getattr(
        builtins, "_arm310d_ik_follow_controller_registry", {"instance": None}
    )
    controller = controller_registry.get("instance")
    if controller is None:
        for _ in range(10):
            await app.next_update_async()
        return {"settled": True, "reason": "IK controller is not running"}

    stable = 0
    last_status = None
    for frame in range(1, max_frames + 1):
        await app.next_update_async()
        last_status = controller.status()
        error = last_status.get("last_error")
        if isinstance(error, dict):
            position_ok = error.get("pos_error_norm_m", 999.0) <= 0.003
            orientation_ok = error.get("rot_error_norm_rad", 999.0) <= 0.05
            stable = stable + 1 if position_ok and orientation_ok else 0
            if stable >= stable_frames:
                return {
                    "settled": True,
                    "frames": frame,
                    "controller_status": last_status,
                }
    return {
        "settled": False,
        "frames": max_frames,
        "controller_status": last_status,
        "reason": "IK target did not converge before capture timeout",
    }


async def demo_capture_setup(force=False):
    """Create and cache the three camera render products for the active stage.

    Args:
        force: Recreate the render products even if already cached for
            this stage.

    Returns:
        dict with status ("already_setup" or "setup_complete") and, on
        setup, stage_id, timeline_was_playing, render_method.
    """
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline

    context, stage, _, _ = _active_stage_context()
    registry = _registry()
    stage_id = context.get_stage_id()
    if (
        not force
        and registry["stage_id"] == stage_id
        and registry["render_products"] is not None
    ):
        return {"status": "already_setup", "stage_id": stage_id}

    render_products = {}
    annotators = {}
    for eye, path in CAMERA_PATHS.items():
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Camera prim not found: {path}")
        product = rep.create.render_product(path, (WIDTH, HEIGHT))
        render_products[eye] = product
        eye_annotators = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb")}
        eye_annotators["rgb"].attach(product)
        if eye in ("left", "right"):
            for name in ("distance_to_image_plane", "distance_to_camera"):
                eye_annotators[name] = rep.AnnotatorRegistry.get_annotator(name)
                eye_annotators[name].attach(product)
        annotators[eye] = eye_annotators

    timeline = omni.timeline.get_timeline_interface()
    was_playing = timeline.is_playing()
    if not was_playing:
        timeline.play()
    app = omni.kit.app.get_app()
    for _ in range(30):
        await app.next_update_async()
    render_method = await _force_render_step(rt_subframes=8)

    registry.update(
        {
            "stage_id": stage_id,
            "render_products": render_products,
            "annotators": annotators,
        }
    )
    return {
        "status": "setup_complete",
        "stage_id": stage_id,
        "timeline_was_playing": was_playing,
        "render_method": render_method,
    }


async def demo_capture_async(label=None) -> dict:
    """Capture, segment, measure, and annotate one run.

    Args:
        label: Optional label suffix for the run directory.

    Returns:
        The result.json dict. Also writes result.json, diagnostics.json,
        the raw and annotated images, and the depth arrays to a new run
        directory.
    """
    import omni.kit.app
    import omni.timeline
    from pxr import Usd, UsdGeom

    timeline = omni.timeline.get_timeline_interface()
    timeline_playing_before = timeline.is_playing()

    setup_result = await demo_capture_setup()
    _, stage, stage_path, output_root = _active_stage_context()
    stage_sha256_before = _sha256(stage_path)
    run_id, run_dir, _timestamp, _label, _sequence = _create_unique_run_dir(
        output_root, label
    )
    annotators = _registry()["annotators"]
    app = omni.kit.app.get_app()

    # Hide before _wait_for_ik_settle() so Hydra has the whole settle window
    # to propagate the visibility change.
    target = stage.GetPrimAtPath(IK_TARGET_PATH)
    target_imageable = None
    previous_visibility = None
    ik_target_hidden = False
    if target.IsValid() and target.IsA(UsdGeom.Imageable):
        target_imageable = UsdGeom.Imageable(target)
        previous_visibility = target_imageable.GetVisibilityAttr().Get()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            target_imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        ik_target_hidden = True

    try:
        settle = await _wait_for_ik_settle()
        for _ in range(5):
            await app.next_update_async()
        render_step_result = await _force_render_step()
        rgb = {}
        rgb_diagnostics = {}
        for eye in ("left", "right", "color"):
            rgb[eye], rgb_diagnostics[eye] = await _read_rgb_with_retry(
                annotators[eye]["rgb"]
            )
        axial_left = np.asarray(
            annotators["left"]["distance_to_image_plane"].get_data(),
            dtype=np.float32,
        )
        radial_left = np.asarray(
            annotators["left"]["distance_to_camera"].get_data(),
            dtype=np.float32,
        )
    finally:
        # Only restore visibility here -- no cleanup call. Tearing down the
        # render graph after every capture previously made the IK target
        # undraggable afterward.
        if target_imageable is not None:
            restored = previous_visibility or UsdGeom.Tokens.inherited
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                target_imageable.GetVisibilityAttr().Set(restored)

    _save_capture_products(run_dir, rgb, axial_left, radial_left)

    frames = {name: _camera_frame(stage, path) for name, path in CAMERA_PATHS.items()}
    arm_base_world_xyz_m = _arm_base_world_xyz(stage)
    baseline = float(
        np.linalg.norm(
            np.asarray(frames["left"]["position_m"])
            - np.asarray(frames["right"]["position_m"])
        )
    )
    green_audit = {eye: _green_audit_eye(rgb[eye]) for eye in ("left", "right", "color")}
    # green_audit is computed on the as-saved rgb, never on an annotated
    # copy (annotation happens after this, on a separate in-memory copy).
    measurement = _build_measurement(stage, axial_left, frames["left"])
    tabletop = _build_tabletop_objects(axial_left, rgb["left"], measurement["intrinsics"], frames["left"], arm_base_world_xyz_m)
    failures = _collect_failures(rgb_diagnostics, settle, green_audit, tabletop["objects"])
    failures.extend(tabletop["failures"])
    if measurement["target_pixel_px"] is None:
        failures.append("no target beyond self-body envelope")
    stage_sha256_after = _sha256(stage_path)

    # Catches the file on disk having changed since demo_start.py's
    # bootstrap exec -- unknown (None) is reported as unknown, never
    # coerced to false.
    loaded_sha256 = _loaded_source_sha256_at_bootstrap()
    disk_sha256 = _source_sha256_or_none()
    stale = None if loaded_sha256 is None or disk_sha256 is None else loaded_sha256 != disk_sha256
    source_integrity = {
        "loaded_sha256": loaded_sha256,
        "disk_sha256": disk_sha256,
        "stale": stale,
    }
    if stale is True:
        failures.append("loaded capture module is stale -- re-run demo_start.py")

    color_intrinsics = _read_camera_intrinsics(stage, CAMERA_PATHS["color"])
    _annotate_rgb_left(rgb["left"], tabletop, run_dir)
    outside_color_fov = _annotate_rgb_color(rgb["color"], measurement, tabletop, frames["color"], color_intrinsics, run_dir)
    _annotate_depth_left(axial_left, tabletop, SELF_BODY_MAX_CAMERA_DISTANCE_M, run_dir)
    _annotate_region_overlay(rgb["left"], axial_left, SELF_BODY_MAX_CAMERA_DISTANCE_M, tabletop, run_dir)
    images = {
        "rgb_left_annotated": "rgb_left_annotated.png",
        "rgb_color_annotated": "rgb_color_annotated.png",
        "depth_left_annotated": "depth_left_annotated.png",
        "region_overlay": "region_overlay.png",
    }
    result = _build_result_json(
        run_id,
        failures,
        frames["left"]["position_m"],
        arm_base_world_xyz_m,
        measurement,
        images,
        source_integrity,
        outside_color_fov,
        tabletop,
    )
    diagnostics = _build_diagnostics_json(
        green_audit,
        {"setup": setup_result, "step": render_step_result},
        frames,
        stage_sha256_before,
        stage_sha256_after,
        timeline_playing_before,
        ik_target_hidden,
        baseline,
        settle,
    )
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "diagnostics.json"), "w", encoding="utf-8") as output:
        json.dump(diagnostics, output, indent=2, ensure_ascii=False)
    print(result)
    return result


async def _scheduled_capture_wrapper(label) -> dict:
    """Run demo_capture_async() and record its result (or error) in the registry.

    Args:
        label: Optional label suffix for the run directory.

    Returns:
        The result dict, or an {"status": "error", ...} dict on exception.
    """
    registry = _registry()
    try:
        result = await demo_capture_async(label)
    except Exception as exc:
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    registry["last_capture_result"] = result
    registry["capture_task"] = None
    return result


def demo_capture(label=None) -> dict:
    """Schedule one capture. GUI-safe: returns immediately.

    Args:
        label: Optional label suffix for the run directory.

    Returns:
        {"status": "scheduled"} or {"status": "already_scheduled"}. Use
        demo_capture_status() to read the result once it's done.
    """
    registry = _registry()
    task = registry.get("capture_task")
    if task is not None and not task.done():
        return {"status": "already_scheduled"}
    task = asyncio.ensure_future(_scheduled_capture_wrapper(label))
    registry["capture_task"] = task
    return {"status": "scheduled"}


def demo_capture_status() -> dict:
    """Status of the most recent demo_capture() call.

    Returns:
        {"status": "capturing"} while running, {"status": "not_captured"}
        if none has run yet, otherwise the result dict.
    """
    registry = _registry()
    task = registry.get("capture_task")
    if task is not None and not task.done():
        return {"status": "capturing"}
    last = registry.get("last_capture_result")
    if last is None:
        return {"status": "not_captured"}
    return last
