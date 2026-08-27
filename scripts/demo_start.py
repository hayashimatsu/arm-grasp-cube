"""One-step GUI bootstrap for the NHC12 + Robotiq + D455 demo.

Open the demo USD, press Play, open this file in Isaac Sim's Script Editor,
and press Ctrl+Enter. The script loads the local IK and capture modules into
THIS file's own globals (the same globals the Script Editor console uses),
starts the IK follow controller, and exposes ``demo_capture()`` in the
Script Editor globals. Re-running this file safely replaces the previous IK
callback.
"""

# Do not synchronously pump the app-update loop after scheduling an async
# task (e.g. ik_follow_start()'s asyncio.ensure_future). Isaac Sim's
# omni.kit.async_engine drives its asyncio loop from the same Kit update
# event (on_event=lambda _: self._loop.run_once()), so a synchronous pump
# call while a task is pending re-enters _run_once(): the inner call drains
# _ready, and the outer call's popleft() then raises
# IndexError: pop from an empty deque, aborting that round of scheduling.

import builtins
import hashlib
import os

# Must match capture_d455.py's REGISTRY_NAME -- that module can't be
# imported yet at the point this is used (it's exec'd, not imported, and
# only after this runs), so the string is duplicated here rather than
# shared.
_CAPTURE_REGISTRY_NAME = "_nhc12_d455_capture_registry"


def _record_capture_source_sha256_at_bootstrap(path):
    """Write capture_d455.py's current sha256 into the shared builtins
    registry, before exec'ing it, so its own staleness check can later
    read this back as the "loaded" baseline.

    Args:
        path: Path to capture_d455.py.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if not hasattr(builtins, _CAPTURE_REGISTRY_NAME):
        setattr(builtins, _CAPTURE_REGISTRY_NAME, {})
    getattr(builtins, _CAPTURE_REGISTRY_NAME)["capture_source_sha256_at_bootstrap"] = digest.hexdigest()


def _demo_project_root():
    """The project root directory, derived from the active stage's path.

    Returns:
        Path string. Raises RuntimeError if no stage is open or it is not
        a saved local file.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    layer = stage.GetRootLayer()
    stage_path = layer.realPath or layer.identifier
    if not stage_path or not os.path.isfile(stage_path):
        raise RuntimeError("Open the saved local demo USD before running this script.")
    return os.path.dirname(os.path.dirname(os.path.abspath(stage_path)))


def _execute_local_script(path):
    """Exec a local script into this module's own globals.

    Args:
        path: Script path.
    """
    with open(path, "r", encoding="utf-8") as source:
        code = compile(source.read(), path, "exec")
    exec(code, globals())


def demo_start():
    """Load the IK and capture modules, and start the IK follow controller.

    Returns:
        dict with status, project_root, capture_setup_result,
        approach_setup_result, gripper_setup_result, api_setup_result,
        ik_start_result, next_action. Raises RuntimeError if the active
        stage is not the demo scene.
    """
    import omni.timeline
    import omni.usd

    root = _demo_project_root()
    stage = omni.usd.get_context().get_stage()
    required_prims = [
        "/World/ArmWithHandOnly",
        "/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/base_link",
        "/World/IKTarget",
        "/World/object",
    ]
    missing = [path for path in required_prims if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"The active stage is not the demo scene; missing: {missing}")

    timeline = omni.timeline.get_timeline_interface()

    if not timeline.is_playing():
        timeline.play()

    _execute_local_script(os.path.join(root, "scripts", "ik_controller.py"))

    capture_script_path = os.path.join(root, "scripts", "capture_d455.py")
    if os.path.isfile(capture_script_path):
        _record_capture_source_sha256_at_bootstrap(capture_script_path)
        _execute_local_script(capture_script_path)
        # capture_d455.py's setup function is a coroutine. Calling it
        # synchronously here would only produce a never-awaited coroutine
        # object and a RuntimeWarning. demo_capture_async() awaits it
        # itself as its first step (cheap to call repeatedly -- cached by
        # stage_id), so this script does not call or schedule it at all.
        capture_setup_result = {
            "status": "deferred",
            "reason": "setup runs inside the first demo_capture()",
        }
    else:
        capture_setup_result = {
            "status": "skipped",
            "reason": "scripts/capture_d455.py not present",
        }

    # capture_approach.py must load after capture_d455.py (it binds that
    # module's raw demo_capture/demo_capture_status and camera-geometry
    # helpers at its own exec time) and before demo_api.py (which calls
    # run_three_phase_capture() and needs that name to already exist in
    # these shared globals).
    approach_script_path = os.path.join(root, "scripts", "capture_approach.py")
    if os.path.isfile(approach_script_path):
        _execute_local_script(approach_script_path)
        approach_setup_result = {"status": "loaded"}
    else:
        approach_setup_result = {
            "status": "skipped",
            "reason": "scripts/capture_approach.py not present",
        }

    # gripper_controller.py is exec'd into these same shared globals, and
    # its two functions published to builtins so grasp_demo._gripper_funcs()
    # (which runs in its own isolated exec namespace) can reuse this one
    # instance instead of exec'ing a second, state-invisible copy.
    gripper_script_path = os.path.join(root, "scripts", "gripper_controller.py")
    if os.path.isfile(gripper_script_path):
        _execute_local_script(gripper_script_path)
        setattr(builtins, "_nhc12_gripper_funcs_registry", {
            "gripper_set_angle_deg": gripper_set_angle_deg,
            "gripper_status": gripper_status,
        })
        gripper_setup_result = {"status": "loaded"}
    else:
        gripper_setup_result = {
            "status": "skipped",
            "reason": "scripts/gripper_controller.py not present",
        }

    # demo_api.py must be exec'd after capture_d455.py -- it raises if
    # capture_d455.py hasn't run yet. Wrong order here is caught there, on
    # purpose -- do not reorder to work around a failure.
    api_script_path = os.path.join(root, "scripts", "demo_api.py")
    if os.path.isfile(api_script_path):
        _execute_local_script(api_script_path)
        api_setup_result = {"status": "loaded"}
    else:
        api_setup_result = {
            "status": "skipped",
            "reason": "scripts/demo_api.py not present",
        }

    # ik_follow_start() schedules an async task and returns immediately
    # ("status": "scheduled"). Do not pump the update loop here to force it
    # forward -- Kit drives its own update loop; let it run and check
    # ik_follow_status() from the console afterward.
    ik_start_result = ik_follow_start()

    result = {
        "status": "ready",
        "project_root": root,
        "capture_setup_result": capture_setup_result,
        "approach_setup_result": approach_setup_result,
        "gripper_setup_result": gripper_setup_result,
        "api_setup_result": api_setup_result,
        "ik_start_result": ik_start_result,
        "next_action": (
            "Wait a second or two, then call ik_follow_status() from the "
            "console. Once running is True, run this three-step loop from "
            "demo_api.py: demo_capture() -> wait -> demo_status(); "
            "demo_trace() -> wait -> demo_status(); "
            "demo_grasp() -> wait -> demo_status(). Each call schedules an "
            "async task and returns immediately -- read the actual result "
            "back from demo_status()."
        ),
    }
    print(result)
    return result


def demo_stop():
    """Stop the IK callback and restore the validated static arm pose."""
    result = ik_follow_stop()
    print(result)
    return result


DEMO_START_RESULT = demo_start()
