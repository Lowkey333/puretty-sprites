import numpy as np

from ..core import ase
from ..core.sprite import ToolError, blank_image, edit, frame0, get_cel, get_layer
from .. import mcp


def _unique_cels(spr: ase.Sprite):
    """Each cel once, even when linked across frames."""
    seen: set[int] = set()
    for layer in spr.layers:
        for cel in layer.cels.values():
            if id(cel) not in seen:
                seen.add(id(cel))
                yield layer, cel


def _scale_nearest(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    rows = np.minimum((np.arange(new_h) + 0.5) * h / new_h, h - 1).astype(int)
    cols = np.minimum((np.arange(new_w) + 0.5) * w / new_w, w - 1).astype(int)
    return image[rows][:, cols]


@mcp.tool()
async def flip_layer(filename: str, layer_name: str, frame_index: int, direction: str = "horizontal") -> str:
    """Flip a layer's cel image horizontally or vertically (in place).

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to flip
        frame_index: Frame index starting at 1
        direction: "horizontal" or "vertical"
    """
    if direction not in ("horizontal", "vertical"):
        return "direction must be 'horizontal' or 'vertical'"
    try:
        with edit(filename) as spr:
            cel = get_cel(spr, get_layer(spr, layer_name), frame0(spr, frame_index))
            cel.image = np.ascontiguousarray(cel.image[:, ::-1] if direction == "horizontal" else cel.image[::-1])
    except ToolError as e:
        return f"Failed to flip layer: {e}"
    return f"Layer '{layer_name}' flipped {direction}ly in {filename}"


@mcp.tool()
async def rotate_layer(filename: str, layer_name: str, frame_index: int, angle: int = 90) -> str:
    """Rotate a layer's cel image 90, 180, or 270 degrees clockwise.

    The cel keeps its top-left position; 90/270 swap its width and height.

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to rotate
        frame_index: Frame index starting at 1
        angle: Rotation angle: 90, 180, or 270 (clockwise)
    """
    if angle not in (90, 180, 270):
        return "angle must be 90, 180, or 270"
    try:
        with edit(filename) as spr:
            cel = get_cel(spr, get_layer(spr, layer_name), frame0(spr, frame_index))
            cel.image = np.ascontiguousarray(np.rot90(cel.image, k=-(angle // 90), axes=(0, 1)))
    except ToolError as e:
        return f"Failed to rotate layer: {e}"
    return f"Layer '{layer_name}' rotated {angle} degrees clockwise in {filename}"


@mcp.tool()
async def resize_canvas(filename: str, width: int, height: int) -> str:
    """Scale the sprite and all its content to new dimensions (nearest neighbor).

    Integer factors (2x, 3x, 0.5x) keep pixel art crisp. To change the
    canvas without scaling the art, use canvas_size.

    Args:
        filename: Sprite file to modify
        width: New canvas width in pixels
        height: New canvas height in pixels
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        with edit(filename) as spr:
            sx, sy = width / spr.width, height / spr.height
            for _layer, cel in _unique_cels(spr):
                new_w = max(1, round(cel.width * sx))
                new_h = max(1, round(cel.height * sy))
                cel.image = _scale_nearest(cel.image, new_w, new_h)
                cel.x, cel.y = round(cel.x * sx), round(cel.y * sy)
            spr.width, spr.height = width, height
    except ToolError as e:
        return f"Failed to resize canvas: {e}"
    return f"Canvas resized to {width}x{height} in {filename}"


def _crop(spr: ase.Sprite, x: int, y: int, width: int, height: int) -> None:
    """Move the canvas origin to (x, y), keep a width x height area and clip every cel to it."""
    for layer in spr.layers:
        done: dict[int, ase.Cel | None] = {}
        for frame, cel in list(layer.cels.items()):
            if id(cel) not in done:
                nx, ny = cel.x - x, cel.y - y
                x0, y0 = max(nx, 0), max(ny, 0)
                x1, y1 = min(nx + cel.width, width), min(ny + cel.height, height)
                if layer.is_background:
                    # New area on a background takes its (opaque) top-left color.
                    fill = cel.image[0, 0].copy()
                    if spr.color_mode != ase.INDEXED:
                        fill[-1] = 255
                    img = blank_image(spr, width, height)
                    img[...] = fill
                    if x0 < x1 and y0 < y1:
                        img[y0:y1, x0:x1] = cel.image[y0 - ny:y1 - ny, x0 - nx:x1 - nx]
                    cel.image, cel.x, cel.y = img, 0, 0
                    done[id(cel)] = cel
                elif x0 < x1 and y0 < y1:
                    cel.image = np.ascontiguousarray(cel.image[y0 - ny:y1 - ny, x0 - nx:x1 - nx])
                    cel.x, cel.y = x0, y0
                    done[id(cel)] = cel
                else:
                    done[id(cel)] = None
            if done[id(cel)] is None:
                del layer.cels[frame]
    spr.width, spr.height = width, height


@mcp.tool()
async def crop_canvas(filename: str, x: int, y: int, width: int, height: int) -> str:
    """Crop the canvas to the given rectangle, discarding content outside it.

    Args:
        filename: Sprite file to modify
        x: Left edge of the crop area
        y: Top edge of the crop area
        width: Width of the crop area
        height: Height of the crop area
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        with edit(filename) as spr:
            _crop(spr, x, y, width, height)
    except ToolError as e:
        return f"Failed to crop canvas: {e}"
    return f"Canvas cropped to ({x},{y}) {width}x{height} in {filename}"


_ANCHORS = {
    "topleft": (0, 0), "top": (0.5, 0), "topright": (1, 0),
    "left": (0, 0.5), "center": (0.5, 0.5), "right": (1, 0.5),
    "bottomleft": (0, 1), "bottom": (0.5, 1), "bottomright": (1, 1),
}


@mcp.tool()
async def canvas_size(filename: str, width: int, height: int, anchor: str = "center") -> str:
    """Grow or shrink the canvas without scaling the art (LibreSprite's Canvas Size).

    Content keeps its size; anchor decides which edge or corner stays put,
    e.g. anchor="bottom" adds room above a character for a raised weapon.
    Shrinking clips content outside the new canvas.

    Args:
        filename: Sprite file to modify
        width: New canvas width in pixels
        height: New canvas height in pixels
        anchor: topleft, top, topright, left, center (default), right,
            bottomleft, bottom or bottomright
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    if anchor not in _ANCHORS:
        return f"Unknown anchor '{anchor}': use {', '.join(_ANCHORS)}"
    ax, ay = _ANCHORS[anchor]
    try:
        with edit(filename) as spr:
            old_w, old_h = spr.width, spr.height
            dx = round((width - old_w) * ax)
            dy = round((height - old_h) * ay)
            _crop(spr, -dx, -dy, width, height)
    except ToolError as e:
        return f"Failed to change canvas size: {e}"
    return f"Canvas changed from {old_w}x{old_h} to {width}x{height} (anchor {anchor}) in {filename}"
