"""Envelope-based "open a USD stage in Isaac Sim via MCP" procedure.

execute_script's return value is broken in this environment; the executed
code writes a JSON completion envelope to validation/tmp/ instead, and the
caller reads it back. Never call save_stage() from this pattern."""

import os


def open_scene_code(usd_path: str, envelope_dir: str) -> str:
    """Build the Python source to paste into execute_script's code argument.

    Args:
        usd_path: USD stage path to open.
        envelope_dir: This project's validation/tmp/ absolute host path,
            where the executed code will write its completion envelope.

    Returns:
        Python source string.
    """

    return f'''
import omni.usd, json, datetime, os, traceback

USD_PATH = {usd_path!r}
ENVELOPE_DIR = {envelope_dir!r}
os.makedirs(ENVELOPE_DIR, exist_ok=True)

ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
envelope_path = os.path.join(ENVELOPE_DIR, "open_scene_" + ts + ".json")

result = {{"timestamp_utc": ts, "requested_usd_path": USD_PATH}}

try:
    ctx = omni.usd.get_context()
    open_ok = ctx.open_stage(USD_PATH)
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("get_stage() returned None after open_stage()")
    root_layer = stage.GetRootLayer()
    default_prim = stage.GetDefaultPrim()
    world_children = []
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        world_children = [c.GetPath().pathString for c in world_prim.GetChildren()]
    result.update({{
        "status": "success",
        "open_stage_returned": bool(open_ok),
        "stage_url": ctx.get_stage_url(),
        "root_layer_identifier": root_layer.identifier,
        "root_layer_dirty": bool(root_layer.dirty),
        "default_prim_path": str(default_prim.GetPath()) if default_prim else None,
        "world_children": world_children,
    }})
except Exception as e:
    result.update({{
        "status": "error",
        "error": str(e),
        "traceback": traceback.format_exc(),
    }})

with open(envelope_path, "w") as f:
    json.dump(result, f, indent=2)

print("ENVELOPE_WRITTEN:" + envelope_path)
'''


# Both referenced projects are siblings of this one under the same
# workspace root -- derive that root from this file's own location instead
# of a hardcoded absolute path.
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This project's envelope scratch directory (host path, not the sandbox
# path -- execute_script runs on the user's real machine).
PROJECT_ENVELOPE_DIR = os.path.join(_WORKSPACE_ROOT, "arm310d-d455-ik-demo", "validation", "tmp")

# Known reference scenes this project cares about.
RIDGEBACK_YOLO_M1_R3_USD = os.path.join(
    _WORKSPACE_ROOT, "ridgeback-franka-d455-yolo-demo",
    "scenes", "ridgeback_franka_d455_yolo_demo_m1_r3.usd",
)
ARM_310D_SOURCE_USD = (
    "/home/rci05/User/Lin/ref_ik_target/IK/Sim_Env/env/"
    "310D_FR_LH_UPR_original2_flattend_EditBlender3.usd"
)
