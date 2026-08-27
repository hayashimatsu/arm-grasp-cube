"""Run bookkeeping for the D455 capture pipeline.

Registry access, output-directory allocation, file hashing, and the
quality-gate/rounding helpers used to assemble a capture result. Most of
this is offline-unit-testable; _active_stage_context() is the exception (it
reads the live USD context). See capture_d455.py's module docstring for
why _source_file_path()/_source_sha256_or_none() stay there instead of here.
"""

import builtins
import datetime as _datetime
import hashlib
import os
import re

from capture_constants import REGISTRY_NAME


def _registry():
    """This module's builtins-backed capture-setup state.

    Returns:
        dict with stage_id, render_products, annotators, capture_task,
        last_capture_result, capture_source_sha256_at_bootstrap.
    """
    if not hasattr(builtins, REGISTRY_NAME):
        setattr(builtins, REGISTRY_NAME, {})
    registry = getattr(builtins, REGISTRY_NAME)
    registry.setdefault("stage_id", None)
    registry.setdefault("render_products", None)
    registry.setdefault("annotators", None)
    registry.setdefault("capture_task", None)
    registry.setdefault("last_capture_result", None)
    return registry


def _sanitize_label(label):
    """Sanitize a user-supplied run label for use in a directory name.

    Args:
        label: Raw label, or None.

    Returns:
        Sanitized label (alphanumeric, dot, underscore, hyphen only,
        truncated to 48 chars), or None if label is None or empties out.
    """
    if label is None:
        return None
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label).strip()).strip("-._")
    return value[:48] or None


def _create_unique_run_dir(output_root, label=None):
    """Allocate a new, uniquely-numbered run directory for today.

    Args:
        output_root: Parent directory for all runs.
        label: Optional label appended to the run id.

    Returns:
        (run_id, run_dir, timestamp, suffix, sequence).
    """
    os.makedirs(output_root, exist_ok=True)
    now_utc = _datetime.datetime.now(_datetime.timezone.utc)
    timestamp = now_utc.strftime("%Y%m%dT%H%M%S%fZ")
    date_prefix = now_utc.astimezone().strftime("%Y-%m%d")
    suffix = _sanitize_label(label)
    existing_pattern = re.compile(
        rf"^{re.escape(date_prefix)}-(\d+)(?:-[A-Za-z0-9._-]+)?$"
    )
    existing_numbers = []
    for name in os.listdir(output_root):
        match = existing_pattern.match(name)
        if match and os.path.isdir(os.path.join(output_root, name)):
            existing_numbers.append(int(match.group(1)))
    first_sequence = max(existing_numbers, default=0) + 1
    for sequence in range(first_sequence, first_sequence + 1000):
        base = f"{date_prefix}-{sequence}"
        run_id = base if suffix is None else f"{base}-{suffix}"
        run_dir = os.path.join(output_root, run_id)
        try:
            os.makedirs(run_dir)
            return run_id, run_dir, timestamp, suffix, sequence
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique capture directory.")


def _sha256(path):
    """SHA-256 hash of a file's contents.

    Args:
        path: File path.

    Returns:
        Hex digest string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_stage_context():
    """The active USD stage context and this run's output root.

    Returns:
        (context, stage, stage_path, output_root). Raises RuntimeError if
        no stage is open or the stage is not a saved local file.
    """
    import omni.usd

    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    layer = stage.GetRootLayer()
    stage_path = layer.realPath or layer.identifier
    if not stage_path or not os.path.isfile(stage_path):
        raise RuntimeError(
            "The active stage must be a saved local USD before demo capture."
        )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(stage_path)))
    output_root = os.environ.get(
        "D455_DEMO_OUTPUT_DIR", os.path.join(project_root, "outputs", "captures")
    )
    return context, stage, os.path.abspath(stage_path), os.path.abspath(output_root)


def _loaded_source_sha256_at_bootstrap():
    """The sha256 of capture_d455.py as it was when demo_start.py exec'd it.

    Returns:
        Hex digest string, or None if unknown.

    Note:
        The baseline value comes from a key demo_start.py writes into this
        same builtins registry right before it execs capture_d455.py.
        None if that key isn't there -- reported as unknown, never coerced
        to a false pass.
    """
    return _registry().get("capture_source_sha256_at_bootstrap")


def _collect_failures(rgb_diagnostics, settle, green_audit, tabletop_objects):
    """Collect human-readable failure messages for one capture.

    Args:
        rgb_diagnostics: Per-eye RGB validity dict.
        settle: IK settle-wait result.
        green_audit: Per-eye green-pixel audit results.
        tabletop_objects: List of detected tabletop objects.

    Returns:
        List of failure message strings; empty when the capture is clean.

    Note:
        Do not loosen green_audit's two conditions: total_px catches a
        signal center_px alone would miss. Invariant: capture-level clean
        implies no object may be flagged green.
    """
    failures = [
        f"{eye} RGB is black or invalid"
        for eye, diagnostic in rgb_diagnostics.items()
        if not diagnostic["valid"]
    ]
    if not settle.get("settled", False):
        failures.append("IK target did not converge before capture timeout")
    for eye, audit in green_audit.items():
        if audit["center_px"] > 100:
            failures.append(f"IK target visible near image centre in {eye} RGB")
        if audit["total_px"] > 100:
            failures.append(f"unexplained green pixels in {eye} RGB")
    capture_level_clean = all(audit["total_px"] == 0 for audit in green_audit.values())
    for obj in tabletop_objects:
        if obj["green_hue_fraction"] <= 0.5:
            continue
        if capture_level_clean:
            failures.append(f"object {obj['id']}: green flag inconsistent with capture-level green_audit")
        else:
            failures.append(f"detected object {obj['id']} is dominated by green hue -- IK target may not be hidden")
    return failures


def _round_floats(value, ndigits=6):
    """Recursively round floats in a JSON-shaped value.

    Args:
        value: A float, dict, list, or other JSON-shaped value.
        ndigits: Decimal digits to round to.

    Returns:
        The same shape, with floats rounded.

    Note:
        For readability only -- not a precision claim.
    """
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: _round_floats(item, ndigits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item, ndigits) for item in value]
    return value
