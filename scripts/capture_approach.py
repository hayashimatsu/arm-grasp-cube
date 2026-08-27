"""Capture the tabletop in three phases: survey, approach, measure -- so no
single fixed camera pose is ever asked to see the whole workspace by itself.

Must be exec'd into demo_start.py's shared globals, after capture_d455.py
and before demo_api.py: it binds capture_d455.py's raw demo_capture/
demo_capture_status and camera-geometry helpers to private names here,
before demo_api.py shadows demo_capture with its own release API. The
IK-target pose/convergence functions and the cube-fit diagnosis rule live
in grasp_demo.py's isolated exec namespace, so run_three_phase_capture()
takes those as call-time arguments instead.
"""

from __future__ import annotations

import math

_survey_capture = globals().get("demo_capture")
_survey_capture_status = globals().get("demo_capture_status")
_geo_camera_frame = globals().get("_camera_frame")
_geo_intrinsics_from_fov = globals().get("_intrinsics_from_fov")
_geo_camera_to_world_xyz = globals().get("_camera_to_world_xyz")
_CAMERA_PATHS = globals().get("CAMERA_PATHS")
if None in (_survey_capture, _survey_capture_status, _geo_camera_frame,
            _geo_intrinsics_from_fov, _geo_camera_to_world_xyz, _CAMERA_PATHS):
    raise RuntimeError("capture_d455.py must be exec'd before capture_approach.py")

# Phase 2's own ceiling on half-step moves before giving up and rolling back
# to last_good_pose.
MAX_APPROACH_STEPS = 3
# Pixel offset from image centre below which a capture is trusted without
# a Phase 2 approach step.
PX_OFFSET_MAX = 70.0
# Each Phase 2 step closes only half the measured offset, never all of it:
# this converges geometrically and never overshoots past the object on a
# single bad depth reading. Do not raise past 1.0 (removes the safety
# margin) or set to 0 (no motion).
APPROACH_STEP_FRACTION = 0.5


def px_offset_from_centre(pixel_xy, width, height):
    """Euclidean pixel distance from pixel_xy to the image's own centre.

    Args:
        pixel_xy: (u, v) pixel coordinates.
        width: Image width, px.
        height: Image height, px.

    Returns:
        Distance in pixels.
    """
    return math.hypot(pixel_xy[0] - width / 2.0, pixel_xy[1] - height / 2.0)


def view_centre_on_plane(camera_frame, intrinsics, width, height, plane_z):
    """World [x, y] where the camera's optical axis crosses a horizontal plane.

    Args:
        camera_frame: Camera pose (a _camera_frame() return value).
        intrinsics: Unused; kept for signature compatibility.
        width: Unused; kept for signature compatibility.
        height: Unused; kept for signature compatibility.
        plane_z: World z of the horizontal plane.

    Returns:
        [x, y] where camera_frame["forward"] crosses z = plane_z.
    """
    del intrinsics, width, height
    position = camera_frame["position_m"]
    forward = camera_frame["forward"]
    if forward[2] == 0:
        raise RuntimeError("camera forward is parallel to the table plane; no intersection")
    t = (plane_z - position[2]) / forward[2]
    return [position[0] + t * forward[0], position[1] + t * forward[1]]


def half_step_target(camera_xy, object_xy, view_centre_xy, fraction):
    """A new camera position, shifted partway from view_centre_xy toward object_xy.

    Args:
        camera_xy: Current camera [x, y].
        object_xy: Target object [x, y].
        view_centre_xy: World [x, y] the camera is currently looking at.
        fraction: Fraction of the view_centre_xy-to-object_xy offset to
            close in this step. 1.0 would fully close the gap in one move.

    Returns:
        New camera [x, y].
    """
    delta_x = object_xy[0] - view_centre_xy[0]
    delta_y = object_xy[1] - view_centre_xy[1]
    return [camera_xy[0] + fraction * delta_x, camera_xy[1] + fraction * delta_y]


def _height_only_candidates(objects, height_range_m):
    """Phase 1's reduced candidate filter: height alone, which is
    rotation-invariant and unaffected by border clipping.

    Args:
        objects: List of detected tabletop objects.
        height_range_m: (min, max) acceptable height above the table plane.

    Returns:
        Filtered list of objects.
    """
    return [o for o in objects if height_range_m[0] <= o["height_above_plane_m"] <= height_range_m[1]]


# Known props that are not grasp targets but sit inside the survey pose's
# field of view and can coincidentally match the cube's height band.
# Excluded by known fixture identity, not a loosened height/ratio/border
# threshold -- every remaining object still has to pass the same rules.
_KNOWN_FIXTURE_PATHS = ("/World/container_h20",)
_fixture_aabb_cache = {}


def _known_fixture_world_aabbs(stage):
    """World AABBs of _KNOWN_FIXTURE_PATHS, computed once and cached.

    Args:
        stage: USD stage.

    Returns:
        List of (min_xyz, max_xyz) AABBs, one per fixture path.
    """
    if "aabbs" not in _fixture_aabb_cache:
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
        aabbs = []
        for path in _KNOWN_FIXTURE_PATHS:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"known fixture prim not found: {path}")
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            aabbs.append((list(box.GetMin()), list(box.GetMax())))
        _fixture_aabb_cache["aabbs"] = aabbs
    return _fixture_aabb_cache["aabbs"]


def _is_known_fixture(surface_center_xyz, aabbs):
    """Whether a world point falls inside any known fixture's AABB.

    Args:
        surface_center_xyz: World xyz to test.
        aabbs: List of (min_xyz, max_xyz) AABBs.

    Returns:
        bool.
    """
    return any(all(lo[i] <= surface_center_xyz[i] <= hi[i] for i in range(3)) for lo, hi in aabbs)


async def _capture_tabletop():
    """One raw capture, with known fixtures removed from the candidate list.

    Returns:
        The tabletop_objects dict, with "objects" filtered. Raises if the
        capture produced no usable result.
    """
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    _survey_capture()
    status = _survey_capture_status()
    while status.get("status") == "capturing":
        await app.next_update_async()
        status = _survey_capture_status()
    if "tabletop_objects" not in status:
        raise RuntimeError(f"capture produced no usable result: status={status.get('status')}, error={status.get('error')}")
    tabletop = status["tabletop_objects"]
    aabbs = _known_fixture_world_aabbs(omni.usd.get_context().get_stage())
    objects = [o for o in tabletop["objects"] if not _is_known_fixture(o["surface_center_world_xyz_m"], aabbs)]
    return {**tabletop, "objects": objects}


def _camera_world_xy(camera_path):
    """World pose and [x, y] position of a camera.

    Args:
        camera_path: Camera prim path.

    Returns:
        (camera_frame, [x, y]).
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    frame = _geo_camera_frame(stage, camera_path)
    return frame, [frame["position_m"][0], frame["position_m"][1]]


async def run_three_phase_capture(survey_pose_xyz, pose_fn, wait_for_convergence_fn,
                                   cube_fit_diagnosis_fn, ik_target_path,
                                   height_range_m=(0.040, 0.060), width=640, height=480):
    """Capture the target object via three phases: survey (height-only
    filter), approach (half-step moves under a monotonic pixel-offset
    guard), and measure (full-criteria).

    Args:
        survey_pose_xyz: World xyz for the Phase 1 survey camera pose.
        pose_fn: Callable(prim_path, xyz) to move a prim.
        wait_for_convergence_fn: Async callable(xyz) to wait for the move
            to converge.
        cube_fit_diagnosis_fn: Callable(obj) -> (fits, detail).
        ik_target_path: IK target prim path.
        height_range_m: (min, max) acceptable object height above the
            table plane.
        width: Image width, px.
        height: Image height, px.

    Returns:
        dict with surface_center_world_xyz_m, height_above_plane_m,
        surface_extent_world_m, approach_steps_taken, survey_pose_used,
        final_pose_used, px_offset_history. Raises RuntimeError if Phase 1
        or Phase 3 does not find exactly one candidate.
    """
    camera_path = _CAMERA_PATHS["left"]

    async def _move_and_capture(xyz):
        pose_fn(ik_target_path, xyz)
        await wait_for_convergence_fn(xyz)
        return await _capture_tabletop()

    def _height_hit(tabletop):
        hits = _height_only_candidates(tabletop["objects"], height_range_m)
        return hits[0] if len(hits) == 1 else None

    tabletop = await _move_and_capture(survey_pose_xyz)
    hit = _height_hit(tabletop)
    if hit is None:
        lines = [f"  [{o['id']}] height={o['height_above_plane_m']:.4f}" for o in tabletop["objects"]]
        raise RuntimeError(
            f"Phase 1 survey expected exactly 1 candidate by height alone, "
            f"found {len(tabletop['objects'])} objects total:\n" + "\n".join(lines)
        )

    pose = list(survey_pose_xyz)
    last_good_pose = list(survey_pose_xyz)
    prev_offset = px_offset_from_centre(hit["representative_pixel_px"], width, height)
    px_offset_history = [prev_offset]
    approach_steps_taken = 0

    for _step in range(MAX_APPROACH_STEPS):
        diagnoses = [cube_fit_diagnosis_fn(o) for o in tabletop["objects"]]
        full_candidates = [o for o, (fits, _d) in zip(tabletop["objects"], diagnoses) if fits]
        if len(full_candidates) == 1 and prev_offset <= PX_OFFSET_MAX:
            break

        camera_frame, camera_xy = _camera_world_xy(camera_path)
        view_xy = view_centre_on_plane(camera_frame, None, width, height, tabletop["plane"]["offset_m"])
        new_xy = half_step_target(camera_xy, hit["surface_center_world_xyz_m"][:2], view_xy, APPROACH_STEP_FRACTION)
        new_pose = [new_xy[0], new_xy[1], pose[2]]
        new_tabletop = await _move_and_capture(new_pose)
        new_hit = _height_hit(new_tabletop)
        new_offset = px_offset_from_centre(new_hit["representative_pixel_px"], width, height) if new_hit else None
        px_offset_history.append(new_offset)

        if new_hit is None or new_offset >= prev_offset:
            tabletop = await _move_and_capture(last_good_pose)
            pose = list(last_good_pose)
            break

        last_good_pose, pose, tabletop, hit, prev_offset = new_pose, new_pose, new_tabletop, new_hit, new_offset
        approach_steps_taken += 1

    diagnoses = [cube_fit_diagnosis_fn(o) for o in tabletop["objects"]]
    candidates = [o for o, (fits, _d) in zip(tabletop["objects"], diagnoses) if fits]
    if len(candidates) != 1:
        lines = [f"  [{o['id']}] {d}" for o, (_f, d) in zip(tabletop["objects"], diagnoses)]
        raise RuntimeError(
            f"Phase 3 measurement expected exactly 1 candidate, found {len(candidates)} of {len(tabletop['objects'])}:\n"
            + "\n".join(lines)
            + f"\napproach_steps_taken={approach_steps_taken}, px_offset_history={px_offset_history}"
        )
    obj = candidates[0]
    return {
        "surface_center_world_xyz_m": obj["surface_center_world_xyz_m"],
        "height_above_plane_m": obj["height_above_plane_m"],
        "surface_extent_world_m": obj["surface_extent_world_m"],
        "approach_steps_taken": approach_steps_taken,
        "survey_pose_used": list(survey_pose_xyz),
        "final_pose_used": pose,
        "px_offset_history": px_offset_history,
    }
