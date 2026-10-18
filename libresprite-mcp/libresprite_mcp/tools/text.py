"""Text drawing.

Glyphs are rasterised in the server (core/fonts.py) and written into the
sprite as pixels; LibreSprite's text tool cannot be driven by scripts.
"""

from typing import Optional

from ..core import ase, raster
from ..core import fonts as fontlib
from ..core.colors import parse_hex_color, to_pixel
from ..core.sprite import ToolError, active_target, drawing_cel, edit, find_layer, frame0, require_file
from .. import mcp

_ANCHORS = (
    "topleft", "top", "topright",
    "left", "center", "right",
    "bottomleft", "bottom", "bottomright",
    "baselineleft", "baseline", "baselineright",
)


def _text_origin(anchor: str, x: int, y: int, metrics: dict) -> tuple[int, int]:
    """Map an anchor point to the pen origin the glyph run should start from.

    Ink coordinates run from the pen origin on x and from the baseline on y, so
    the baseline* anchors need no vertical correction at all while the box
    anchors are offset by the ink extents.
    """
    width, height = metrics["width"], metrics["height"]

    if anchor.endswith("right"):
        x -= width
    elif anchor in ("top", "center", "bottom", "baseline"):
        x -= width // 2
    origin_x = x - metrics["left"]

    if anchor.startswith("baseline"):
        return origin_x, y
    if anchor.startswith("bottom"):
        y -= height
    elif anchor in ("left", "center", "right"):
        y -= height // 2
    return origin_x, y - metrics["top"]


@mcp.tool()
async def list_text_fonts() -> str:
    """List fonts that draw_text can use.

    Covers bitmap sprite-sheet fonts and .ttf/.otf files installed under
    ~/.libresprite-mcp/fonts (and ~/.aseprite-mcp/fonts), plus TrueType
    fonts found in the system font directories. Bitmap fonts are the right
    choice for pixel art: their glyphs are already pixels and `size`
    scales them by whole numbers.
    """
    try:
        found = fontlib.available_fonts()
    except OSError as exc:  # pragma: no cover
        return f"Failed to list fonts: {exc}"
    if not found:
        return f"No fonts found. Drop a .ttf into {fontlib.FONT_DIR}."

    user = [f for f in found if f["source"] == "user"]
    system = [f for f in found if f["source"] == "system"]
    lines = []
    if user:
        lines.append("User fonts (~/.libresprite-mcp/fonts):")
        lines += [f"  {f['name']}  [{f['kind']}]" for f in user]
    if system:
        lines.append(f"System fonts ({len(system)}, all truetype):")
        lines += [f"  {f['name']}" for f in system]
    return "\n".join(lines)


@mcp.tool()
async def measure_text(
    text: str,
    font: str,
    size: int = 1,
    letter_spacing: int = 0,
    bold: int = 0,
    antialias: bool = False,
) -> str:
    """Measure text without drawing it.

    Use this to size a panel or centre a label in one shot instead of
    guessing and re-rendering.

    Args:
        text: The string to measure
        font: Font name from list_text_fonts, or a path to a .ttf/.otf or
            bitmap font directory
        size: Pixel size for TrueType fonts; integer scale factor for bitmap
            fonts (default 1)
        letter_spacing: Extra pixels between glyphs (default 0)
        bold: Faux-bold passes; each grows strokes by 1px (default 0)
        antialias: Count faint edge pixels as ink, for heavier glyphs
            (default False: only pixels at least half covered)

    Returns:
        width, height, advance_width, and the ink extents above/below the
        baseline.
    """
    try:
        f = fontlib.load_font(font)
        _, m = fontlib.shape(text, f, size, letter_spacing, bold, antialias)
    except (fontlib.FontError, OSError, ValueError) as exc:
        return f"Failed to measure text: {exc}"
    return (
        f"width={m['width']} height={m['height']} advance_width={m['advance_width']} "
        f"above_baseline={-m['top']} below_baseline={m['bottom'] + 1} "
        f"left_bearing={m['left']}"
    )


@mcp.tool()
async def draw_text(
    filename: str,
    text: str,
    x: int,
    y: int,
    font: str,
    size: int = 1,
    color: str = "#FFFFFF",
    layer_name: Optional[str] = None,
    frame_index: int = 1,
    anchor: str = "topleft",
    letter_spacing: int = 0,
    bold: int = 0,
    outline_color: Optional[str] = None,
    outline_width: int = 1,
    outline_diagonal: bool = True,
    shadow_color: Optional[str] = None,
    shadow_dx: int = 1,
    shadow_dy: int = 1,
    antialias: bool = False,
    create_if_missing: bool = True,
) -> str:
    """Draw text onto a sprite.

    Glyphs are rasterised by the server and written into the cel as
    pixels; text pixels replace what is under them, the rest of the cel
    is untouched.

    Args:
        filename: Sprite file to modify
        text: String to draw
        x: X coordinate of the anchor point
        y: Y coordinate of the anchor point
        font: Font name from list_text_fonts, or a path to a .ttf/.otf or a
            bitmap font directory
        size: Pixel size for TrueType fonts; integer scale factor for bitmap
            fonts (default 1)
        color: Text colour, #RRGGBB or #RRGGBBAA (default white)
        layer_name: Layer to draw on; the active layer if omitted
        frame_index: Frame index starting at 1 (default 1)
        anchor: Which point of the text (x, y) refers to - one of topleft,
            top, topright, left, center, right, bottomleft, bottom,
            bottomright, baselineleft, baseline, baselineright
            (default topleft)
        letter_spacing: Extra pixels between glyphs (default 0)
        bold: Faux-bold passes; each grows strokes by 1px (default 0)
        outline_color: If set, draw an outline in this colour
        outline_width: Outline thickness in pixels (default 1)
        outline_diagonal: Include diagonal neighbours in the outline, giving a
            rounded corner; False gives a boxier 4-way outline (default True)
        shadow_color: If set, draw a drop shadow in this colour
        shadow_dx: Shadow X offset (default 1)
        shadow_dy: Shadow Y offset (default 1)
        antialias: Count faint edge pixels as ink, for heavier glyphs
            (default False)
        create_if_missing: Create the layer/cel if absent (default True)
    """
    try:
        require_file(filename)
    except ToolError as e:
        return str(e)
    if anchor not in _ANCHORS:
        return f"Invalid anchor '{anchor}'. Expected one of: {', '.join(sorted(_ANCHORS))}"

    fill = parse_hex_color(color)
    if fill is None:
        return f"Invalid color value: {color}"
    outline_rgba = None
    if outline_color:
        outline_rgba = parse_hex_color(outline_color)
        if outline_rgba is None:
            return f"Invalid outline_color value: {outline_color}"
    shadow_rgba = None
    if shadow_color:
        shadow_rgba = parse_hex_color(shadow_color)
        if shadow_rgba is None:
            return f"Invalid shadow_color value: {shadow_color}"

    try:
        f = fontlib.load_font(font)
        ink, metrics = fontlib.shape(text, f, size, letter_spacing, bold, antialias)
    except (fontlib.FontError, OSError, ValueError) as exc:
        return f"Failed to draw text: {exc}"
    if not ink:
        return "OK: nothing to draw (text has no visible glyphs)"

    # Build the layers of the stamp, back to front.
    outline = set()
    if outline_rgba:
        outline = set(ink)
        for _ in range(max(1, outline_width)):
            grown = set(outline)
            for px, py in outline:
                grown.update({(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)})
                if outline_diagonal:
                    grown.update({(px - 1, py - 1), (px + 1, py - 1),
                                  (px - 1, py + 1), (px + 1, py + 1)})
            outline = grown
        outline -= ink
    shadow = {(px + shadow_dx, py + shadow_dy) for px, py in ink} if shadow_rgba else set()
    shadow -= ink | outline

    everything = ink | outline | shadow
    min_x = min(p[0] for p in everything)
    min_y = min(p[1] for p in everything)
    w = max(p[0] for p in everything) - min_x + 1
    h = max(p[1] for p in everything) - min_y + 1

    # The anchor is resolved against the glyph box alone, so adding an outline
    # or a shadow never shifts where the letters themselves land.
    origin_x, origin_y = _text_origin(anchor, x, y, metrics)

    try:
        with edit(filename) as spr:
            frame0(spr, frame_index)
            if layer_name:
                if find_layer(spr, layer_name) is None:
                    if not create_if_missing:
                        raise ToolError(f"Layer not found: {layer_name!r}")
                    spr.layers.append(ase.Layer(layer_name))
                target = layer_name
            else:
                target, _ = active_target(spr, filename)
            img = drawing_cel(spr, target, frame_index, create_if_missing).image
            for group, rgba in ((shadow, shadow_rgba), (outline, outline_rgba), (ink, fill)):
                if rgba:
                    raster.put_points(img, [(origin_x + gx, origin_y + gy) for gx, gy in group],
                                      to_pixel(spr, rgba))
    except ToolError as e:
        return f"Failed to draw text: {e}"

    return (f"Drew '{text}' at ({origin_x + min_x}, {origin_y + min_y}), "
            f"text box {metrics['width']}x{metrics['height']}, stamp {w}x{h}")
