import numpy as np

from ..core.colors import image_to_rgba, parse_hex_color, transparent_pixel
from ..core.sprite import ToolError, drawing_cel, edit, frame0, get_cel, get_layer, normalize_cel, opaque
from .. import mcp


def _clip_rect(spr, x, y, w, h):
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, spr.width), min(y + h, spr.height)
    return x0, y0, x1, y1


def _paste_block(spr, img, block, dest_x, dest_y) -> None:
    """Paste a pixel block, skipping fully transparent source pixels."""
    bh, bw = block.shape[:2]
    x0, y0 = max(dest_x, 0), max(dest_y, 0)
    x1, y1 = min(dest_x + bw, spr.width), min(dest_y + bh, spr.height)
    if x0 >= x1 or y0 >= y1:
        return
    part = block[y0 - dest_y:y1 - dest_y, x0 - dest_x:x1 - dest_x]
    mask = opaque(spr, part)
    img[y0:y1, x0:x1][mask] = part[mask]


@mcp.tool()
async def move_region(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
    dest_x: int,
    dest_y: int,
) -> str:
    """Cut a rectangular region of pixels and paste it at a new position.

    Coordinates are sprite-global. The source area is left transparent.
    Pixels moved outside the canvas are discarded. Fully transparent
    source pixels do not overwrite destination pixels.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to operate on
        frame_index: Frame index starting at 1
        x: Left edge of the source region
        y: Top edge of the source region
        width: Region width
        height: Region height
        dest_x: New left edge for the region
        dest_y: New top edge for the region
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        with edit(filename) as spr:
            cel = drawing_cel(spr, layer_name, frame_index, create=False)
            x0, y0, x1, y1 = _clip_rect(spr, x, y, width, height)
            if x0 >= x1 or y0 >= y1:
                raise ToolError("The source region is outside the canvas")
            block = cel.image[y0:y1, x0:x1].copy()
            cel.image[y0:y1, x0:x1] = transparent_pixel(spr)
            _paste_block(spr, cel.image, block, dest_x + (x0 - x), dest_y + (y0 - y))
    except ToolError as e:
        return f"Failed to move region: {e}"
    return (f"Moved {width}x{height} region from ({x},{y}) to ({dest_x},{dest_y}) "
            f"on '{layer_name}' frame {frame_index} in {filename}")


@mcp.tool()
async def copy_region(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
    dest_x: int,
    dest_y: int,
    target_layer_name: str = "",
    target_frame_index: int = 0,
) -> str:
    """Copy a rectangular region of pixels to another position, layer, or frame.

    Coordinates are sprite-global. Fully transparent source pixels do not
    overwrite destination pixels. The destination cel is created when missing.

    Args:
        filename: Sprite file to modify
        layer_name: Source layer
        frame_index: Source frame index starting at 1
        x: Left edge of the source region
        y: Top edge of the source region
        width: Region width
        height: Region height
        dest_x: Left edge of the destination
        dest_y: Top edge of the destination
        target_layer_name: Destination layer (default: same as source)
        target_frame_index: Destination frame (default: same as source)
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        with edit(filename) as spr:
            source = normalize_cel(spr, get_cel(spr, get_layer(spr, layer_name), frame0(spr, frame_index)))
            x0, y0, x1, y1 = _clip_rect(spr, x, y, width, height)
            if x0 >= x1 or y0 >= y1:
                raise ToolError("The source region is outside the canvas")
            block = source.image[y0:y1, x0:x1].copy()
            target = drawing_cel(spr, target_layer_name or layer_name, target_frame_index or frame_index)
            _paste_block(spr, target.image, block, dest_x + (x0 - x), dest_y + (y0 - y))
    except ToolError as e:
        return f"Failed to copy region: {e}"
    return f"Copied {width}x{height} region from ({x},{y}) to ({dest_x},{dest_y}) in {filename}"


@mcp.tool()
async def erase_region(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    """Erase (make transparent) a rectangular region of pixels.

    Coordinates are sprite-global.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to operate on
        frame_index: Frame index starting at 1
        x: Left edge of the region
        y: Top edge of the region
        width: Region width
        height: Region height
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        with edit(filename) as spr:
            cel = drawing_cel(spr, layer_name, frame_index, create=False)
            x0, y0, x1, y1 = _clip_rect(spr, x, y, width, height)
            cel.image[y0:y1, x0:x1] = transparent_pixel(spr)
    except ToolError as e:
        return f"Failed to erase region: {e}"
    return f"Erased {width}x{height} region at ({x},{y}) on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def erase_color(
    filename: str,
    layer_name: str,
    frame_index: int,
    color: str,
    tolerance: int = 0,
) -> str:
    """Make all pixels of a given color transparent (like a magic eraser).

    Args:
        filename: Sprite file to modify
        layer_name: Layer to operate on
        frame_index: Frame index starting at 1
        color: Hex color to erase, e.g. "#FF00FF"
        tolerance: Per-channel tolerance 0-255 (default 0 = exact match)
    """
    rgba = parse_hex_color(color)
    if rgba is None:
        return f"Invalid color value: {color}"
    if not 0 <= tolerance <= 255:
        return "Tolerance must be between 0 and 255"
    try:
        with edit(filename) as spr:
            cel = get_cel(spr, get_layer(spr, layer_name), frame0(spr, frame_index))
            pixels = image_to_rgba(spr, cel.image).astype(np.int32)
            match = np.all(np.abs(pixels[..., :3] - np.array(rgba[:3])) <= tolerance, axis=-1)
            match &= opaque(spr, cel.image)
            count = int(match.sum())
            cel.image[match] = transparent_pixel(spr)
    except ToolError as e:
        return f"Failed to erase color: {e}"
    return f"Erased {count} pixels of {color} on '{layer_name}' frame {frame_index} in {filename}"
