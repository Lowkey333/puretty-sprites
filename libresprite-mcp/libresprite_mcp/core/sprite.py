"""Document helpers shared by the tool modules.

Tools load a sprite, change it and save it back in one call:

    with edit(filename) as spr:
        cel = drawing_cel(spr, layer_name, frame_index, create_if_missing)
        raster.rect(cel.image, x, y, w, h, color_value(spr, color))

A ToolError raised inside the block aborts without saving; tools turn it
into their "Failed to ...: <reason>" reply. Frame indices are 1-based at
the tool boundary, as in the upstream server, and 0-based in the model.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np

from . import ase
from .colors import parse_hex_color, to_pixel, transparent_pixel, image_to_rgba, rgba_to_image
from .palettes import PRESETS


class ToolError(Exception):
    """A tool precondition failed; the message is shown to the caller."""


def reject_traversal(path: str) -> str | None:
    """Reject parent-directory traversal in a user-supplied path.

    Returns an error message when the path contains a `..` component,
    or None when it looks safe. Absolute paths are allowed: this guards
    against traversal, not access scoping.
    """
    parts = os.path.normpath(path).replace("\\", "/").split("/")
    if ".." in parts:
        return "Invalid filename: parent directory traversal not allowed"
    return None


def require_file(filename: str) -> None:
    if not filename or not os.path.isfile(filename):
        raise ToolError(f"File {filename} not found")


def load(filename: str) -> ase.Sprite:
    require_file(filename)
    try:
        return ase.read(filename)
    except ase.AseError as e:
        raise ToolError(f"Cannot read {filename}: {e}") from e


def save(spr: ase.Sprite, filename: str) -> None:
    try:
        ase.write(spr, filename)
    except ValueError as e:
        raise ToolError(str(e)) from e


@contextmanager
def edit(filename: str):
    """Load a sprite, yield it, and save it if the block finishes."""
    spr = load(filename)
    yield spr
    save(spr, filename)


def new_sprite(width: int, height: int, color_mode: str = ase.RGB) -> ase.Sprite:
    """A one-frame sprite with an empty "Layer 1" cel, like Aseprite's Sprite(w, h)."""
    if color_mode == ase.GRAY:
        palette = [(v, v, v, 255) for v in range(0, 256, 17)]
    else:
        palette = list(PRESETS["dawnbringer32"])
    spr = ase.Sprite(width, height, color_mode, palette=palette)
    layer = ase.Layer("Layer 1")
    layer.cels[0] = ase.Cel(blank_image(spr, width, height))
    spr.layers.append(layer)
    return spr


def blank_image(spr: ase.Sprite, width: int, height: int) -> np.ndarray:
    img = np.zeros((height, width, spr.channels), np.uint8)
    img[...] = transparent_pixel(spr)
    return img


def frame0(spr: ase.Sprite, frame_index: int) -> int:
    """Validate a 1-based frame index and return it 0-based."""
    if not isinstance(frame_index, int) or not 1 <= frame_index <= spr.frame_count:
        raise ToolError(f"Frame index out of range (1-{spr.frame_count}): {frame_index}")
    return frame_index - 1


def frame_range(spr: ase.Sprite, start_frame: int, end_frame: int | None) -> range:
    """Validate a 1-based inclusive range (end None/0 = last frame); 0-based range."""
    end = spr.frame_count if not end_frame else end_frame
    if not (1 <= start_frame <= end <= spr.frame_count):
        raise ToolError(f"Frame range out of bounds: {start_frame}-{end} (sprite has {spr.frame_count} frames)")
    return range(start_frame - 1, end)


def find_layer(spr: ase.Sprite, name: str) -> ase.Layer | None:
    for layer in spr.layers:
        if layer.name == name:
            return layer
    return None


def get_layer(spr: ase.Sprite, name: str) -> ase.Layer:
    layer = find_layer(spr, name)
    if layer is None:
        names = ", ".join(repr(l.name) for l in spr.layers) or "none"
        raise ToolError(f"Layer not found: {name!r} (layers: {names})")
    return layer


def layer_index(spr: ase.Sprite, layer: ase.Layer) -> int:
    return next(i for i, l in enumerate(spr.layers) if l is layer)


def color_value(spr: ase.Sprite, color: str) -> tuple[int, ...]:
    rgba = parse_hex_color(color)
    if rgba is None:
        raise ToolError(f"Invalid color value: {color}")
    return to_pixel(spr, rgba)


def parse_color(color: str) -> tuple[int, int, int, int]:
    rgba = parse_hex_color(color)
    if rgba is None:
        raise ToolError(f"Invalid color value: {color}")
    return rgba


def get_cel(spr: ase.Sprite, layer: ase.Layer, frame: int) -> ase.Cel:
    cel = layer.cels.get(frame)
    if cel is None:
        raise ToolError(f"No cel on layer '{layer.name}' at frame {frame + 1}")
    return cel


def normalize_cel(spr: ase.Sprite, cel: ase.Cel) -> ase.Cel:
    """Make the cel canvas-sized at (0, 0), so sprite coordinates index it.

    Like the upstream tools' normalize_cel: the cel's offset is baked into
    its pixels and anything outside the canvas is clipped.
    """
    if (cel.x, cel.y, cel.width, cel.height) == (0, 0, spr.width, spr.height):
        return cel
    img = blank_image(spr, spr.width, spr.height)
    x0, y0 = max(cel.x, 0), max(cel.y, 0)
    x1, y1 = min(cel.x + cel.width, spr.width), min(cel.y + cel.height, spr.height)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = cel.image[y0 - cel.y:y1 - cel.y, x0 - cel.x:x1 - cel.x]
    cel.image, cel.x, cel.y = img, 0, 0
    return cel


def drawing_cel(spr: ase.Sprite, layer_name: str, frame_index: int, create: bool = True) -> ase.Cel:
    """The cel to draw on, created if missing (when allowed) and normalized."""
    layer = get_layer(spr, layer_name)
    frame = frame0(spr, frame_index)
    cel = layer.cels.get(frame)
    if cel is None:
        if not create:
            raise ToolError(f"No cel on layer '{layer_name}' at frame {frame_index} (create_if_missing is off)")
        cel = layer.cels[frame] = ase.Cel(blank_image(spr, spr.width, spr.height))
    return normalize_cel(spr, cel)


def cel_rgba(spr: ase.Sprite, cel: ase.Cel) -> np.ndarray:
    return image_to_rgba(spr, cel.image)


def set_cel_rgba(spr: ase.Sprite, cel: ase.Cel, rgba: np.ndarray) -> None:
    cel.image = rgba_to_image(spr, rgba)


def paste_rgba(spr: ase.Sprite, img: np.ndarray, rgba: np.ndarray, x: int, y: int,
               mask: np.ndarray | None = None) -> None:
    """Write an RGBA block into a cel image at (x, y), clipped, in the sprite's mode.

    With mask, only pixels where it is True are written.
    """
    h, w = img.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + rgba.shape[1], w), min(y + rgba.shape[0], h)
    if x0 >= x1 or y0 >= y1:
        return
    block = rgba_to_image(spr, np.ascontiguousarray(rgba[y0 - y:y1 - y, x0 - x:x1 - x]))
    if mask is None:
        img[y0:y1, x0:x1] = block
    else:
        m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        img[y0:y1, x0:x1][m] = block[m]


def update_cel_rgba(spr: ase.Sprite, cel: ase.Cel, rgba: np.ndarray, mask: np.ndarray) -> int:
    """Write the RGBA pixels selected by mask back into the cel; returns the count.

    Only masked pixels are converted, so an indexed cel's untouched pixels
    keep their exact palette index.
    """
    if mask.any():
        converted = rgba_to_image(spr, rgba)
        cel.image[mask] = converted[mask]
    return int(mask.sum())


def alpha_channel(spr: ase.Sprite) -> int | None:
    return {ase.RGB: 3, ase.GRAY: 1}.get(spr.color_mode)


def opaque(spr: ase.Sprite, image: np.ndarray) -> np.ndarray:
    channel = alpha_channel(spr)
    if channel is not None:
        return image[..., channel] > 0
    return image[..., 0] != spr.transparent_index


def content_bounds(spr: ase.Sprite, cel: ase.Cel) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) of the cel's non-transparent pixels in sprite coordinates."""
    mask = opaque(spr, cel.image)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (int(xs.min()) + cel.x, int(ys.min()) + cel.y,
            int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)


# The "active" layer and frame. LibreSprite keeps no active state in the
# file, so set_layer/set_frame remember it for this server session and the
# tools without a layer argument (draw_pixels, draw_line, ...) draw there.
_active: dict[str, dict] = {}


def _key(filename: str) -> str:
    return os.path.normcase(os.path.abspath(filename))


def set_active(filename: str, layer: str | None = None, frame: int | None = None) -> None:
    state = _active.setdefault(_key(filename), {})
    if layer is not None:
        state["layer"] = layer
    if frame is not None:
        state["frame"] = frame


def active_target(spr: ase.Sprite, filename: str) -> tuple[str, int]:
    """(layer name, 1-based frame) for tools that draw on the active cel.

    Defaults to the topmost layer and frame 1 when nothing was set.
    """
    if not spr.layers:
        raise ToolError("The sprite has no layers; add one with add_layer")
    state = _active.get(_key(filename), {})
    layer = state.get("layer")
    if layer is None or find_layer(spr, layer) is None:
        layer = spr.layers[-1].name
    frame = state.get("frame", 1)
    if not 1 <= frame <= spr.frame_count:
        frame = 1
    return layer, frame
