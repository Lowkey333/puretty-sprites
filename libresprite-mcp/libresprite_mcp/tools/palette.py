import colorsys
import json
from typing import List

import numpy as np

from ..core import ase
from ..core.colors import image_to_rgba, parse_hex_color, quantize_rgba, rgba_to_image, to_hex
from ..core.palettes import PRESET_HEX, PRESETS
from ..core.sprite import (ToolError, cel_rgba, edit, frame0, frame_range, get_layer, load, opaque,
                           update_cel_rgba)
from .. import mcp


def _hex_list(palette) -> list[str]:
    return [to_hex(c, with_alpha=c[3] != 255) for c in palette]


@mcp.tool()
async def get_palette(filename: str) -> str:
    """Get the sprite palette as a JSON array of hex colors.

    Colors are #RRGGBB, or #RRGGBBAA for entries that aren't fully opaque.
    For indexed sprites the entry at the transparent index is see-through
    on transparent layers; the reply notes which index that is.
    """
    try:
        spr = load(filename)
    except ToolError as e:
        return f"Failed to get palette: {e}"
    colors = json.dumps(_hex_list(spr.palette))
    if spr.color_mode == ase.INDEXED:
        return f"{colors}\n(indexed sprite: transparent index {spr.transparent_index})"
    return colors


@mcp.tool()
async def set_palette(filename: str, colors: List[str]) -> str:
    """Set the sprite palette from a list of hex colors (1-256 entries).

    On RGB sprites the palette is just the swatch set; run
    quantize_to_palette to make existing art use it. On indexed sprites the
    pixels keep their indices, so the art is recolored.
    """
    if not colors:
        return "Colors list cannot be empty"
    if len(colors) > 256:
        return "A palette can have at most 256 colors"
    parsed = [parse_hex_color(c) for c in colors]
    if any(p is None for p in parsed):
        return "Colors must use #RRGGBB or #RRGGBBAA values"
    try:
        with edit(filename) as spr:
            spr.palette = parsed  # type: ignore[assignment]
    except ToolError as e:
        return f"Failed to set palette: {e}"
    return f"Palette set with {len(colors)} colors in {filename}"


@mcp.tool()
async def remap_colors_in_cel_range(
    filename: str,
    layer_name: str,
    start_frame: int,
    end_frame: int,
    mappings: List[dict],
    create_missing_cels: bool = False,
    source_frame_index: int | None = None
) -> str:
    """Remap colors in a layer across a frame range using explicit mappings.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to remap colors in
        start_frame: First frame (1-based)
        end_frame: Last frame (1-based, inclusive)
        mappings: List of {"from": "#RRGGBB", "to": "#RRGGBB"}; alpha is kept
        create_missing_cels: Create missing cels by copying the source frame's cel
        source_frame_index: Frame to copy missing cels from (default: start_frame)
    """
    if not mappings:
        return "Mappings cannot be empty"
    pairs = []
    for m in mappings:
        src = parse_hex_color(m.get("from") or "")
        dst = parse_hex_color(m.get("to") or "")
        if src is None or dst is None:
            return "Each mapping needs valid 'from' and 'to' hex colors"
        pairs.append((src[:3], dst[:3]))
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            frames = frame_range(spr, start_frame, end_frame)
            source = frame0(spr, source_frame_index or start_frame)
            changed = 0
            for frame in frames:
                cel = layer.cels.get(frame)
                if cel is None and create_missing_cels and source in layer.cels:
                    cel = layer.cels[frame] = layer.cels[source].copy()
                if cel is None:
                    continue
                rgba = cel_rgba(spr, cel)
                visible = opaque(spr, cel.image)
                mask = np.zeros(visible.shape, bool)
                original = rgba[..., :3].copy()
                for src, dst in pairs:
                    hit = visible & np.all(original == src, axis=-1)
                    rgba[hit, :3] = dst
                    mask |= hit
                changed += update_cel_rgba(spr, cel, rgba, mask)
    except ToolError as e:
        return f"Failed to remap colors: {e}"
    return (f"Remapped {changed} pixels on '{layer_name}' frames {start_frame}-{end_frame} "
            f"in {filename}")


@mcp.tool()
async def list_palette_presets() -> str:
    """List the built-in retro palette presets with their colors."""
    return json.dumps(PRESET_HEX, indent=1)


@mcp.tool()
async def apply_palette_preset(filename: str, preset: str) -> str:
    """Set the sprite palette to a built-in retro preset.

    Presets: gameboy, monochrome, grayscale_4, cga, pico8, c64,
    dawnbringer16, dawnbringer32, sweetie16, endesga32. Follow with
    quantize_to_palette to snap existing RGB art to the new palette.

    Args:
        filename: Sprite file to modify
        preset: Preset name (see list_palette_presets)
    """
    key = preset.strip().lower()
    if key not in PRESETS:
        return f"Unknown preset '{preset}'. Available: {', '.join(PRESETS)}"
    try:
        with edit(filename) as spr:
            spr.palette = list(PRESETS[key])
    except ToolError as e:
        return f"Failed to apply palette preset: {e}"
    return f"Applied palette preset '{key}' ({len(PRESETS[key])} colors) to {filename}"


@mcp.tool()
async def generate_color_ramp(
    base_color: str,
    steps: int = 5,
    hue_shift_degrees: float = 20,
    lightness_range: float = 0.5,
) -> str:
    """Generate a shading ramp (dark to light) from a base color.

    Produces the standard pixel-art shading technique of hue-shifting:
    shadows lean cooler (hue shifted one way), highlights lean warmer.
    Use the returned colors for shading instead of plain darker/lighter
    versions of the same hue.

    Args:
        base_color: Hex color the ramp is built around, e.g. "#D04648"
        steps: Number of colors in the ramp, 2-16 (default 5)
        hue_shift_degrees: Total hue rotation across the ramp (default 20)
        lightness_range: Total lightness span across the ramp, 0-1 (default 0.5)

    Returns:
        JSON array of hex colors ordered darkest to lightest.
    """
    rgba = parse_hex_color(base_color)
    if rgba is None:
        return f"Invalid color value: {base_color}"
    if not (2 <= steps <= 16):
        return "steps must be between 2 and 16"
    if not (0 <= lightness_range <= 1):
        return "lightness_range must be between 0 and 1"

    h, l, s = colorsys.rgb_to_hls(*(c / 255 for c in rgba[:3]))
    ramp = []
    mid = (steps - 1) / 2
    for i in range(steps):
        # t in [-0.5, 0.5]: negative = shadow side, positive = highlight side
        t = (i - mid) / (steps - 1)
        nh = (h - t * (hue_shift_degrees / 360)) % 1.0
        nl = min(1.0, max(0.0, l + t * lightness_range))
        # Shadows slightly more saturated, highlights slightly less
        ns = min(1.0, max(0.0, s - t * 0.15))
        nr, ng, nb = colorsys.hls_to_rgb(nh, nl, ns)
        ramp.append("#{:02X}{:02X}{:02X}".format(round(nr * 255), round(ng * 255), round(nb * 255)))
    return json.dumps(ramp)


@mcp.tool()
async def quantize_to_palette(
    filename: str,
    layer_name: str = "",
    start_frame: int = 1,
    end_frame: int = 0,
) -> str:
    """Snap every pixel to the nearest color in the sprite's palette.

    Walks the chosen cels and replaces each opaque pixel with the
    closest palette color (RGB distance), keeping its alpha. Run after
    apply_palette_preset or set_palette to make existing art conform.
    Indexed sprites always use palette colors, so there is nothing to do.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to quantize (empty = all layers)
        start_frame: First frame to process (default 1)
        end_frame: Last frame to process (default 0 = last frame)
    """
    try:
        with edit(filename) as spr:
            if spr.color_mode == ase.INDEXED:
                return f"{filename} is indexed: its pixels already use palette colors"
            layers = [get_layer(spr, layer_name)] if layer_name else spr.layers
            frames = frame_range(spr, start_frame, end_frame)
            pal = np.array([c[:3] for c in spr.palette], np.int32)
            count = 0
            seen: set[int] = set()
            for layer in layers:
                for frame in frames:
                    cel = layer.cels.get(frame)
                    if cel is None or id(cel) in seen:
                        continue
                    seen.add(id(cel))
                    rgba = cel_rgba(spr, cel)
                    flat = rgba[..., :3].reshape(-1, 3).astype(np.int32)
                    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
                    dist = ((uniq[:, None, :] - pal[None, :, :]) ** 2).sum(axis=-1)
                    snapped = pal[dist.argmin(axis=1)][inverse.reshape(-1)].reshape(rgba.shape[:2] + (3,))
                    mask = opaque(spr, cel.image) & np.any(snapped != rgba[..., :3], axis=-1)
                    rgba[..., :3] = snapped
                    count += update_cel_rgba(spr, cel, rgba, mask)
    except ToolError as e:
        return f"Failed to quantize: {e}"
    return f"Quantized {count} pixels to the {len(pal)}-color palette in {filename}"


@mcp.tool()
async def set_color_mode(filename: str, mode: str) -> str:
    """Convert the sprite's color mode.

    rgb -> indexed maps every pixel to its nearest palette color. In
    indexed mode the palette entry at the transparent index (0) is
    see-through, so when that entry is a real color in use the conversion
    first inserts a transparent entry at index 0 (if the palette has room).

    Args:
        filename: Sprite file to modify
        mode: "rgb", "grayscale" (or "gray"), or "indexed"
    """
    target = {"grayscale": ase.GRAY, "greyscale": ase.GRAY}.get(mode.lower(), mode.lower())
    if target not in ase.COLOR_MODES:
        return "mode must be 'rgb', 'grayscale', or 'indexed'"
    note = ""
    try:
        with edit(filename) as spr:
            if spr.color_mode == target:
                return f"{filename} is already in {mode} mode"
            source_mode = spr.color_mode
            # Every cel to RGBA first (shared/linked cels converted once).
            converted: dict[int, np.ndarray] = {}
            for layer in spr.layers:
                for cel in layer.cels.values():
                    if id(cel) not in converted:
                        rgba = image_to_rgba(spr, cel.image)
                        if layer.is_background and source_mode == ase.INDEXED:
                            pal = np.zeros((256, 4), np.uint8)
                            pal[:len(spr.palette)] = spr.palette
                            rgba = pal[cel.image[..., 0]]
                            rgba[..., 3] = 255
                        converted[id(cel)] = rgba
            if target == ase.INDEXED:
                if spr.palette[0][3] != 0:
                    if len(spr.palette) < 256:
                        spr.palette = [(0, 0, 0, 0)] + list(spr.palette)
                        note = " (inserted a transparent palette entry at index 0)"
                    else:
                        note = (" (palette is full: entry 0 became the transparent color, so pixels "
                                "of that color were mapped to their nearest other entry)")
                spr.transparent_index = 0
            elif target == ase.GRAY:
                spr.palette = [(v, v, v, 255) for v in range(0, 256, 17)]
            spr.color_mode = target
            for layer in spr.layers:
                for cel in layer.cels.values():
                    if id(cel) in converted:
                        rgba = converted.pop(id(cel))
                        if target == ase.INDEXED:
                            cel.image = quantize_rgba(rgba, spr.palette, spr.transparent_index)[..., None]
                        else:
                            cel.image = rgba_to_image(spr, rgba)
    except ToolError as e:
        return f"Failed to set color mode: {e}"
    return f"Color mode set to {mode} in {filename}{note}"
