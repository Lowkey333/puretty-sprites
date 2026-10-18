import numpy as np

from ..core import raster
from ..core.colors import hsl_to_rgb_array, parse_hex_color, rgb_to_hsl_array
from ..core.sprite import (ToolError, cel_rgba, color_value, drawing_cel, edit, frame0, get_cel,
                           get_layer, opaque, paste_rgba, update_cel_rgba)
from .. import mcp


@mcp.tool()
async def outline_cel(
    filename: str,
    layer_name: str,
    frame_index: int,
    color: str = "#000000",
    include_diagonals: bool = False,
    place: str = "outside",
) -> str:
    """Add a 1px outline around all opaque pixels of a cel.

    Transparent pixels adjacent to opaque pixels are filled with the
    outline color. Great for making sprites read clearly against any
    background.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to outline
        frame_index: Frame index starting at 1
        color: Outline hex color (default black)
        include_diagonals: Also outline diagonal neighbors for a thicker,
            rounded outline (default False)
        place: "outside" (default) paints the transparent pixels around the
            shape; "inside" repaints the shape's own edge pixels instead
    """
    if place not in ("outside", "inside"):
        return "place must be 'outside' or 'inside'"
    try:
        with edit(filename) as spr:
            value = color_value(spr, color)
            cel = drawing_cel(spr, layer_name, frame_index, create=False)
            mask = raster.outline_mask(opaque(spr, cel.image), include_diagonals, inside=place == "inside")
            cel.image[mask] = value
    except ToolError as e:
        return f"Failed to outline cel: {e}"
    return f"Outline added to '{layer_name}' frame {frame_index} in {filename} ({int(mask.sum())} pixels)"


@mcp.tool()
async def replace_color(
    filename: str,
    layer_name: str,
    frame_index: int,
    from_color: str,
    to_color: str,
    tolerance: int = 0,
) -> str:
    """Replace one color with another in a cel, preserving alpha.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to operate on
        frame_index: Frame index starting at 1
        from_color: Hex color to replace, e.g. "#FF0000"
        to_color: Replacement hex color
        tolerance: Per-channel tolerance 0-255 (default 0 = exact match)
    """
    src = parse_hex_color(from_color)
    dst = parse_hex_color(to_color)
    if src is None or dst is None:
        return "Colors must use #RRGGBB values"
    if tolerance < 0 or tolerance > 255:
        return "Tolerance must be between 0 and 255"
    try:
        with edit(filename) as spr:
            cel = get_cel(spr, get_layer(spr, layer_name), frame0(spr, frame_index))
            rgba = cel_rgba(spr, cel)
            match = np.all(np.abs(rgba[..., :3].astype(np.int32) - np.array(src[:3])) <= tolerance, axis=-1)
            match &= opaque(spr, cel.image)
            rgba[match, :3] = dst[:3]
            count = update_cel_rgba(spr, cel, rgba, match)
    except ToolError as e:
        return f"Failed to replace color: {e}"
    return (f"Replaced {count} pixels {from_color} -> {to_color} on '{layer_name}' "
            f"frame {frame_index} in {filename}")


@mcp.tool()
async def adjust_hsl(
    filename: str,
    layer_name: str,
    frame_index: int,
    hue_shift: float = 0,
    saturation_shift: float = 0,
    lightness_shift: float = 0,
    x: int = 0,
    y: int = 0,
    width: int = 0,
    height: int = 0,
) -> str:
    """Shift hue, saturation, and lightness of all opaque pixels in a cel.

    Useful for creating palette-swapped variants and shading: e.g. darken
    a duplicated layer for shadows or hue-shift toward blue for night
    scenes.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to adjust
        frame_index: Frame index starting at 1
        hue_shift: Degrees to rotate hue, -360 to 360
        saturation_shift: Saturation delta, -100 to 100
        lightness_shift: Lightness delta, -100 to 100
        x: Optional region left edge (sprite coordinates)
        y: Optional region top edge
        width: Optional region width (0 = whole cel)
        height: Optional region height (0 = whole cel)
    """
    if not (-360 <= hue_shift <= 360):
        return "hue_shift must be between -360 and 360"
    if not (-100 <= saturation_shift <= 100):
        return "saturation_shift must be between -100 and 100"
    if not (-100 <= lightness_shift <= 100):
        return "lightness_shift must be between -100 and 100"
    try:
        with edit(filename) as spr:
            cel = get_cel(spr, get_layer(spr, layer_name), frame0(spr, frame_index))
            rgba = cel_rgba(spr, cel)
            mask = opaque(spr, cel.image) & _region_mask(cel, x, y, width, height)
            h, s, l = rgb_to_hsl_array(rgba[..., :3])
            h = h + hue_shift
            s = np.clip(s + saturation_shift / 100, 0, 1)
            l = np.clip(l + lightness_shift / 100, 0, 1)
            rgba[..., :3] = hsl_to_rgb_array(h, s, l)
            update_cel_rgba(spr, cel, rgba, mask)
    except ToolError as e:
        return f"Failed to adjust HSL: {e}"
    return (f"Adjusted HSL (h{hue_shift:+g}, s{saturation_shift:+g}, l{lightness_shift:+g}) "
            f"on '{layer_name}' frame {frame_index} in {filename}")


def _region_mask(cel, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Pixels of the cel inside a sprite-coordinate rectangle (all when width/height are 0)."""
    if width <= 0 or height <= 0:
        return np.ones(cel.image.shape[:2], bool)
    ys, xs = np.mgrid[0:cel.height, 0:cel.width]
    xs, ys = xs + cel.x, ys + cel.y
    return (xs >= x) & (xs < x + width) & (ys >= y) & (ys < y + height)


def _dither_fill(spr, img, x, y, width, height, color_a, color_b, choose_b) -> None:
    ys, xs = np.mgrid[y:y + height, x:x + width]
    pick = choose_b(xs, ys, raster.bayer_threshold(xs, ys))
    block = np.where(pick[..., None], np.array(color_b, np.uint8), np.array(color_a, np.uint8))
    paste_rgba(spr, img, block, x, y)


@mcp.tool()
async def apply_dither_gradient(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
    color_start: str,
    color_end: str,
    horizontal: bool = False,
    create_if_missing: bool = True,
) -> str:
    """Fill a rectangle with a two-color gradient using Bayer 4x4 ordered dithering.

    This is the classic pixel-art way to blend two colors without
    introducing new intermediate colors. The gradient runs from
    color_start (top/left) to color_end (bottom/right).

    Args:
        filename: Sprite file to modify
        layer_name: Layer to draw on
        frame_index: Frame index starting at 1
        x: Left edge of the rectangle
        y: Top edge of the rectangle
        width: Rectangle width
        height: Rectangle height
        color_start: Hex color at the start of the gradient
        color_end: Hex color at the end of the gradient
        horizontal: Run the gradient left-to-right instead of top-to-bottom
        create_if_missing: Create the cel if it does not exist
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    start = parse_hex_color(color_start)
    end = parse_hex_color(color_end)
    if start is None or end is None:
        return "Colors must use #RRGGBB values"
    span = max(1, (width if horizontal else height) - 1)

    def choose_end(xs, ys, threshold):
        return ((xs - x) if horizontal else (ys - y)) / span >= threshold

    try:
        with edit(filename) as spr:
            img = drawing_cel(spr, layer_name, frame_index, create_if_missing).image
            _dither_fill(spr, img, x, y, width, height, start, end, choose_end)
    except ToolError as e:
        return f"Failed to apply dither gradient: {e}"
    direction = "horizontal" if horizontal else "vertical"
    return (f"Dithered {direction} gradient {color_start} -> {color_end} applied at "
            f"({x},{y}) {width}x{height} on '{layer_name}' frame {frame_index} in {filename}")


@mcp.tool()
async def apply_dither_pattern(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
    color_a: str,
    color_b: str,
    density: float = 0.5,
    create_if_missing: bool = True,
) -> str:
    """Fill a rectangle with a uniform Bayer-dithered mix of two colors.

    density controls the ratio: 0.0 = all color_a, 0.5 = checkerboard,
    1.0 = all color_b. Useful for textures (stone, grass) and flat
    mid-tones between two palette colors.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to draw on
        frame_index: Frame index starting at 1
        x: Left edge of the rectangle
        y: Top edge of the rectangle
        width: Rectangle width
        height: Rectangle height
        color_a: Base hex color
        color_b: Mixed-in hex color
        density: Fraction of color_b, 0.0-1.0 (default 0.5)
        create_if_missing: Create the cel if it does not exist
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    if not (0 <= density <= 1):
        return "density must be between 0.0 and 1.0"
    a = parse_hex_color(color_a)
    b = parse_hex_color(color_b)
    if a is None or b is None:
        return "Colors must use #RRGGBB values"
    try:
        with edit(filename) as spr:
            img = drawing_cel(spr, layer_name, frame_index, create_if_missing).image
            _dither_fill(spr, img, x, y, width, height, a, b, lambda xs, ys, t: density > t)
    except ToolError as e:
        return f"Failed to apply dither pattern: {e}"
    return (f"Dither pattern ({color_a}/{color_b}, density {density}) applied at "
            f"({x},{y}) {width}x{height} on '{layer_name}' frame {frame_index} in {filename}")
