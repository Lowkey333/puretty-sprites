import json

import numpy as np

from ..core import ase, render
from ..core.colors import image_to_rgba
from ..core.sprite import ToolError, active_target, frame0, get_layer, load
from .. import mcp

# Rectangle reads return one JSON object per pixel, so cap them to keep
# replies a sane size; look at bigger areas with export_frame instead.
MAX_RECT_PIXELS = 4096


def _describe(rgba, index=None) -> str:
    r, g, b, a = (int(v) for v in rgba)
    extra = f", index={index}" if index is not None else ""
    return f"#{r:02x}{g:02x}{b:02x} (r={r}, g={g}, b={b}, a={a}{extra})"


def _entry(x: int, y: int, rgba, index=None) -> dict:
    r, g, b, a = (int(v) for v in rgba)
    entry = {"x": x, "y": y, "hex": f"#{r:02x}{g:02x}{b:02x}", "r": r, "g": g, "b": b, "a": a}
    if index is not None:
        entry["index"] = index
    return entry


def _source_cel(spr: ase.Sprite, filename: str, layer_name: str, frame_index: int) -> ase.Cel:
    """The cel a single-layer read looks at: the named layer, else the active one."""
    frame = frame0(spr, frame_index)
    if layer_name:
        cel = get_layer(spr, layer_name).cels.get(frame)
        if cel is None:
            raise ToolError(f"No cel at layer '{layer_name}' frame {frame_index}")
        return cel
    active, _ = active_target(spr, filename)
    cel = next(l for l in spr.layers if l.name == active).cels.get(frame)
    if cel is None:
        raise ToolError(f"No active cel (layer '{active}' has no cel at frame {frame_index})")
    return cel


def _cel_block(spr: ase.Sprite, cel: ase.Cel, x: int, y: int, width: int, height: int):
    """RGBA (and indices for indexed sprites) of a sprite-coordinate rectangle of one cel.

    Pixels outside the cel read as transparent black, with no index.
    """
    rgba = np.zeros((height, width, 4), np.uint8)
    index = np.full((height, width), -1, np.int32)
    x0, y0 = max(x, cel.x), max(y, cel.y)
    x1, y1 = min(x + width, cel.x + cel.width), min(y + height, cel.y + cel.height)
    if x0 < x1 and y0 < y1:
        part = cel.image[y0 - cel.y:y1 - cel.y, x0 - cel.x:x1 - cel.x]
        rgba[y0 - y:y1 - y, x0 - x:x1 - x] = image_to_rgba(spr, part)
        if spr.color_mode == ase.INDEXED:
            index[y0 - y:y1 - y, x0 - x:x1 - x] = part[..., 0]
    return rgba, index


def _composite_block(spr: ase.Sprite, frame: int, x: int, y: int, width: int, height: int) -> np.ndarray:
    flat = render.composite(spr, frame)
    rgba = np.zeros((height, width, 4), np.uint8)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, spr.width), min(y + height, spr.height)
    if x0 < x1 and y0 < y1:
        rgba[y0 - y:y1 - y, x0 - x:x1 - x] = flat[y0:y1, x0:x1]
    return rgba


@mcp.tool()
async def get_pixel_color(
    filename: str,
    x: int,
    y: int,
    layer_name: str = "",
    frame_index: int = 1,
) -> str:
    """Read the RGBA color of a single pixel of one cel.

    On indexed sprites the palette color is reported, followed by the
    palette index.

    Args:
        filename: Sprite file to read
        x: X coordinate (sprite-global)
        y: Y coordinate (sprite-global)
        layer_name: Layer to read from (uses the active layer when empty)
        frame_index: Frame index starting at 1
    """
    try:
        spr = load(filename)
        cel = _source_cel(spr, filename, layer_name, frame_index)
        rgba, index = _cel_block(spr, cel, x, y, 1, 1)
    except ToolError as e:
        return f"Failed to read pixel: {e}"
    return _describe(rgba[0, 0], int(index[0, 0]) if index[0, 0] >= 0 else None)


@mcp.tool()
async def get_pixels_rect(
    filename: str,
    x: int,
    y: int,
    width: int,
    height: int,
    layer_name: str = "",
    frame_index: int = 1,
) -> str:
    """Read all pixel colors in a rectangular region of one cel.

    Args:
        filename: Sprite file to read
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Width of the region
        height: Height of the region (width x height at most 4096 pixels)
        layer_name: Layer to read from (uses the active layer when empty)
        frame_index: Frame index starting at 1

    Returns:
        JSON array of {x, y, hex, r, g, b, a} objects, row by row
        (plus "index" on indexed sprites)
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    if width * height > MAX_RECT_PIXELS:
        return (f"Region must be at most {MAX_RECT_PIXELS} pixels (got {width * height}); "
                f"read it in smaller pieces or look at it with export_frame")
    try:
        spr = load(filename)
        cel = _source_cel(spr, filename, layer_name, frame_index)
        rgba, index = _cel_block(spr, cel, x, y, width, height)
    except ToolError as e:
        return f"Failed to read pixels: {e}"
    pixels = [_entry(x + dx, y + dy, rgba[dy, dx], int(index[dy, dx]) if index[dy, dx] >= 0 else None)
              for dy in range(height) for dx in range(width)]
    return json.dumps(pixels)


@mcp.tool()
async def get_composite_pixel(filename: str, x: int, y: int, frame_index: int = 1) -> str:
    """Read the RGBA colour VISIBLE at a pixel (flattened composite of all layers).

    Unlike get_pixel_color (which reads a single cel), this composites every
    visible layer with its opacity and blend mode, exactly as LibreSprite
    renders it: "what the player actually sees".

    Args:
        filename: Sprite file to read
        x: X coordinate (sprite-global)
        y: Y coordinate (sprite-global)
        frame_index: Frame index starting at 1
    """
    try:
        spr = load(filename)
        rgba = _composite_block(spr, frame0(spr, frame_index), x, y, 1, 1)
    except ToolError as e:
        return f"Failed to read composite pixel: {e}"
    return _describe(rgba[0, 0])


@mcp.tool()
async def get_composite_rect(
    filename: str,
    x: int,
    y: int,
    width: int,
    height: int,
    frame_index: int = 1,
) -> str:
    """Read VISIBLE RGBA over a rectangle (flattened composite of all layers).

    The rectangular counterpart of get_composite_pixel: reads the composited
    pixels every visible layer produces, not a single cel.

    Args:
        filename: Sprite file to read
        x: Top-left x (sprite-global)
        y: Top-left y (sprite-global)
        width: Region width
        height: Region height (width x height at most 4096 pixels)
        frame_index: Frame index starting at 1

    Returns:
        JSON array of {x, y, hex, r, g, b, a} objects, row by row
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    if width * height > MAX_RECT_PIXELS:
        return (f"Region must be at most {MAX_RECT_PIXELS} pixels (got {width * height}); "
                f"read it in smaller pieces or look at it with export_frame")
    try:
        spr = load(filename)
        rgba = _composite_block(spr, frame0(spr, frame_index), x, y, width, height)
    except ToolError as e:
        return f"Failed to read composite pixels: {e}"
    return json.dumps([_entry(x + dx, y + dy, rgba[dy, dx]) for dy in range(height) for dx in range(width)])
