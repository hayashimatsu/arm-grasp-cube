"""Annotated-image rendering for the D455 capture pipeline.

Draws labelled crosshairs, mask overlays, and saves PNGs -- all PIL/numpy
image work, no Isaac Sim / USD access. Testable offline against saved
.npy/.png fixtures.
"""

import os

import numpy as np

from capture_constants import WIDTH, HEIGHT
from capture_geometry import _camera_xyz_to_pixel, _world_xyz_to_camera_xyz


def _save_png(array, path):
    """Save an array as a PNG.

    Args:
        array: Image array.
        path: Output path.
    """
    from PIL import Image

    Image.fromarray(array.astype(np.uint8)).save(path)


def _depth_preview(depth):
    """Grayscale RGB preview of a depth array, contrast-stretched to its
    2nd-98th percentile range.

    Args:
        depth: Depth array; non-finite or non-positive values are invalid.

    Returns:
        uint8 RGB array, same height/width as depth.
    """
    valid_mask = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid_mask):
        values = depth[valid_mask]
        low, high = np.percentile(values, [2.0, 98.0])
        scaled = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        preview[valid_mask] = (scaled[valid_mask] * 255.0).astype(np.uint8)
    return np.stack([preview, preview, preview], axis=-1)


# Fixed palette -- never green, that's green_audit's detection colour.
_TARGET_COLOR = (255, 0, 255)
_NEAREST_COLOR = (0, 200, 255)  # reserved; currently unused
_SELF_BODY_COLOR = (255, 140, 0)
_REGION_PALETTE = [
    (255, 0, 255), (0, 200, 255), (255, 140, 0), (255, 64, 64),
    (170, 120, 255), (255, 210, 0), (0, 160, 200), (200, 80, 160),
]


def _annotation_font(size=15):
    """Load a truetype font, falling back to PIL's default if unavailable.

    Args:
        size: Font size, px.

    Returns:
        A PIL ImageFont.
    """
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_label_box(draw, anchor_xy, text, accent_rgb, image_size, font=None, pad=4, offset=(14, -10)):
    """Draw a white-fill/black-text label box with an accent border.

    Args:
        draw: PIL ImageDraw instance.
        anchor_xy: (x, y) anchor point.
        text: Label text.
        accent_rgb: Border color.
        image_size: (width, height) of the target image, for edge clamping.
        font: PIL font; defaults to _annotation_font().
        pad: Padding around the text, px.
        offset: (x, y) offset from anchor_xy.

    Returns:
        (x0, y0, x1, y1) box bounds. Auto-flips off the image edge.
    """
    font = font or _annotation_font()
    width, height = image_size
    ax, ay = anchor_xy
    bbox = draw.textbbox((0, 0), text, font=font)
    box_w = (bbox[2] - bbox[0]) + 2 * pad
    box_h = (bbox[3] - bbox[1]) + 2 * pad
    x, y = ax + offset[0], ay + offset[1]
    if x + box_w > width:
        x = ax - box_w - 14
    if y < 0:
        y = ay + 10
    x = max(0, min(x, width - box_w))
    y = max(0, min(y, height - box_h))
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(255, 255, 255), outline=accent_rgb, width=2)
    draw.text((x + pad, y + pad), text, font=font, fill=(0, 0, 0))
    return x, y, x + box_w, y + box_h


def _draw_crosshair(draw, xy, accent_rgb, arm=9, gap=3, radius=7):
    """Draw a double-stroked crosshair and ring: white outer, accent inner.

    Args:
        draw: PIL ImageDraw instance.
        xy: (x, y) center point.
        accent_rgb: Inner stroke color.
        arm: Crosshair arm length, px.
        gap: Gap between ring and arms, px.
        radius: Ring radius, px.
    """
    x, y = xy
    for width, color, r in ((4, (255, 255, 255), radius + 1), (2, accent_rgb, radius)):
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=width)
        for x0, y0, x1, y1 in (
            (-(gap + arm), 0, -gap, 0), (gap, 0, gap + arm, 0),
            (0, -(gap + arm), 0, -gap), (0, gap, 0, gap + arm),
        ):
            draw.line([x + x0, y + y0, x + x1, y + y1], fill=color, width=width)


def _blend_mask(image, mask, color_rgb, alpha_255):
    """Alpha-blend a color into an image where mask is True.

    Args:
        image: PIL RGB image.
        mask: Boolean mask, same height/width as image.
        color_rgb: Color to blend in.
        alpha_255: Blend strength, 0-255.

    Returns:
        A new PIL RGB image.
    """
    from PIL import Image

    array = np.asarray(image.convert("RGB")).astype(np.float32)
    alpha = alpha_255 / 255.0
    array[mask] = array[mask] * (1 - alpha) + np.array(color_rgb, dtype=np.float32) * alpha
    return Image.fromarray(array.astype(np.uint8), "RGB")


def _mask_centroid(mask):
    """Pixel centroid of a boolean mask.

    Args:
        mask: Boolean mask.

    Returns:
        (x, y) centroid, or None if the mask is empty.
    """
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.mean()), int(ys.mean())


def _object_color(index):
    """The palette color for the object at this index.

    Args:
        index: Object index.

    Returns:
        (r, g, b) tuple.
    """
    return _REGION_PALETTE[index % len(_REGION_PALETTE)]


def _annotate_rgb_left(rgb_left, tabletop, run_dir):
    """Draw one crosshair and label per tabletop object on the left RGB image.

    Args:
        rgb_left: Left-eye RGB array.
        tabletop: Tabletop detection result, with an "objects" list.
        run_dir: Output directory; writes rgb_left_annotated.png.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb_left.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, obj in enumerate(tabletop["objects"]):
        color = _object_color(index)
        u, v = obj["representative_pixel_px"]
        _draw_crosshair(draw, (u, v), color)
        _draw_label_box(draw, (u, v), f"OBJ{obj['id']} ({u},{v})", color, image.size)
    image.save(os.path.join(run_dir, "rgb_left_annotated.png"))


def _annotate_rgb_color(rgb_color, measurement, tabletop, color_frame, color_intrinsics, run_dir):
    """Reproject each object's surface centre into the color camera and annotate.

    Args:
        rgb_color: Color-eye RGB array.
        measurement: Measurement dict with target_world_xyz_m.
        tabletop: Tabletop detection result, with an "objects" list.
        color_frame: Color camera pose (a _camera_frame() return value).
        color_intrinsics: Color camera intrinsics.
        run_dir: Output directory; writes rgb_color_annotated.png.

    Returns:
        bool: whether the legacy single-target point (from measurement)
        falls outside the color camera's frame. Objects outside the frame
        are simply not drawn.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb_color.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, obj in enumerate(tabletop["objects"]):
        color = _object_color(index)
        projected = _camera_xyz_to_pixel(
            _world_xyz_to_camera_xyz(obj["surface_center_world_xyz_m"], color_frame), color_intrinsics
        )
        if projected is None or not (0 <= projected[0] < WIDTH and 0 <= projected[1] < HEIGHT):
            continue
        u, v = int(round(projected[0])), int(round(projected[1]))
        _draw_crosshair(draw, (u, v), color)
        _draw_label_box(draw, (u, v), f"OBJ{obj['id']} ({u},{v})", color, image.size)
    image.save(os.path.join(run_dir, "rgb_color_annotated.png"))
    legacy_world = measurement["target_world_xyz_m"]
    if legacy_world is None:
        return False
    legacy_projected = _camera_xyz_to_pixel(_world_xyz_to_camera_xyz(legacy_world, color_frame), color_intrinsics)
    return legacy_projected is None or not (0 <= legacy_projected[0] < WIDTH and 0 <= legacy_projected[1] < HEIGHT)


def _annotate_depth_left(axial_left, tabletop, self_body_max_m, run_dir):
    """Draw a depth preview with a self-body overlay and one labelled
    crosshair per tabletop object.

    Args:
        axial_left: Left-eye axial depth array.
        tabletop: Tabletop detection result, with an "objects" list.
        self_body_max_m: Depth below this counts as the robot's own body.
        run_dir: Output directory; writes depth_left_annotated.png.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(_depth_preview(axial_left), "RGB")
    self_body_mask = np.isfinite(axial_left) & (axial_left > 0) & (axial_left < self_body_max_m)
    if np.any(self_body_mask):
        image = _blend_mask(image, self_body_mask, _SELF_BODY_COLOR, 90)
    draw = ImageDraw.Draw(image)
    centroid = _mask_centroid(self_body_mask)
    if centroid is not None:
        _draw_label_box(draw, centroid, "SELF-BODY", _SELF_BODY_COLOR, image.size)
    for index, obj in enumerate(tabletop["objects"]):
        color = _object_color(index)
        u, v = obj["representative_pixel_px"]
        dist_m = obj["camera_to_surface_center_distance_m"]
        _draw_crosshair(draw, (u, v), color)
        _draw_label_box(draw, (u, v), f"OBJ{obj['id']} {dist_m:.3f} m", color, image.size)
    image.save(os.path.join(run_dir, "depth_left_annotated.png"))


_TABLE_COLOR = (160, 160, 160)


def _annotate_region_overlay(rgb_left, axial_left, self_body_max_m, tabletop, run_dir):
    """Draw the table plane, each detected object, and the self-body mask
    as colored overlays on the left RGB image.

    Args:
        rgb_left: Left-eye RGB array.
        axial_left: Left-eye axial depth array.
        self_body_max_m: Depth below this counts as the robot's own body.
        tabletop: Tabletop detection result, with table_mask and "objects".
        run_dir: Output directory; writes region_overlay.png.
    """
    from PIL import Image, ImageDraw

    valid = np.isfinite(axial_left) & (axial_left > 0)
    self_body_mask = valid & (axial_left < self_body_max_m)
    image = Image.fromarray(rgb_left.astype(np.uint8)).convert("RGB")
    image = _blend_mask(image, tabletop["table_mask"], _TABLE_COLOR, 60)
    for index, obj in enumerate(tabletop["objects"]):
        image = _blend_mask(image, obj["mask"], _object_color(index), 100)
    image = _blend_mask(image, self_body_mask, _SELF_BODY_COLOR, 90)
    draw = ImageDraw.Draw(image)
    table_centroid = _mask_centroid(tabletop["table_mask"])
    if table_centroid is not None:
        _draw_label_box(draw, table_centroid, "TABLE", _TABLE_COLOR, image.size)
    self_body_centroid = _mask_centroid(self_body_mask)
    if self_body_centroid is not None:
        _draw_label_box(draw, self_body_centroid, "SELF-BODY", _SELF_BODY_COLOR, image.size)
    for index, obj in enumerate(tabletop["objects"]):
        color = _object_color(index)
        u, v = obj["representative_pixel_px"]
        _draw_crosshair(draw, (u, v), color)
        _draw_label_box(draw, (u, v), f"OBJ{obj['id']} ({u},{v})", color, image.size)
    image.save(os.path.join(run_dir, "region_overlay.png"))
