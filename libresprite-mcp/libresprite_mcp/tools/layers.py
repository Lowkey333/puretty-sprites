import numpy as np

from ..core import ase, render
from ..core.colors import rgba_to_image
from ..core.sprite import ToolError, edit, find_layer, get_layer, layer_index
from .. import mcp

_BLEND_MODES = {name: i for i, name in enumerate(ase.BLEND_MODES)}
_BLEND_ALIASES = {"hsl_hue": "hue", "hsl_saturation": "saturation", "hsl_color": "color",
                  "hsl_luminosity": "luminosity", "colordodge": "color_dodge",
                  "colorburn": "color_burn", "hardlight": "hard_light", "softlight": "soft_light"}


@mcp.tool()
async def delete_layer(filename: str, layer_name: str) -> str:
    """Delete a layer by name.

    Args:
        filename: Sprite file to modify
        layer_name: Name of the layer to delete
    """
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            if len(spr.layers) == 1:
                raise ToolError("Cannot delete the only layer")
            spr.layers.remove(layer)
    except ToolError as e:
        return f"Failed to delete layer: {e}"
    return f"Layer '{layer_name}' deleted from {filename}"


@mcp.tool()
async def rename_layer(filename: str, layer_name: str, new_name: str) -> str:
    """Rename a layer.

    Args:
        filename: Sprite file to modify
        layer_name: Current layer name
        new_name: New layer name
    """
    if not new_name:
        return "New name cannot be empty"
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            if new_name != layer_name and find_layer(spr, new_name):
                raise ToolError(f"A layer named '{new_name}' already exists")
            layer.name = new_name
    except ToolError as e:
        return f"Failed to rename layer: {e}"
    return f"Layer '{layer_name}' renamed to '{new_name}' in {filename}"


@mcp.tool()
async def duplicate_layer(filename: str, layer_name: str, new_name: str = "") -> str:
    """Duplicate a layer with all its cels across every frame.

    The copy inherits the source's visibility, opacity and blend mode and is
    placed directly above the source. Linked cels stay linked in the copy.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to duplicate
        new_name: Name for the copy (default: "<layer_name> copy")
    """
    final_name = new_name or f"{layer_name} copy"
    try:
        with edit(filename) as spr:
            source = get_layer(spr, layer_name)
            if find_layer(spr, final_name):
                raise ToolError(f"A layer named '{final_name}' already exists")
            copies: dict[int, ase.Cel] = {}
            cels = {}
            for frame, cel in source.cels.items():
                if id(cel) not in copies:
                    copies[id(cel)] = cel.copy()
                cels[frame] = copies[id(cel)]
            copy = ase.Layer(final_name, source.flags & ~(ase.FLAG_BACKGROUND | ase.FLAG_LOCK_MOVEMENT),
                             source.blend_mode, source.opacity, cels)
            spr.layers.insert(layer_index(spr, source) + 1, copy)
    except ToolError as e:
        return f"Failed to duplicate layer: {e}"
    return f"Layer '{layer_name}' duplicated as '{final_name}' in {filename}"


@mcp.tool()
async def reorder_layer(filename: str, layer_name: str, position: int) -> str:
    """Move a layer to a new position in the layer stack.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to move
        position: Target stack position, 1-based where 1 is the bottom layer
    """
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            if not 1 <= position <= len(spr.layers):
                raise ToolError(f"Position must be between 1 and {len(spr.layers)}")
            if spr.layers[0].is_background and position == 1 and layer is not spr.layers[0]:
                raise ToolError("Position 1 holds the background layer, which must stay at the bottom")
            if layer.is_background and position != 1:
                raise ToolError("The background layer must stay at the bottom")
            spr.layers.remove(layer)
            spr.layers.insert(position - 1, layer)
    except ToolError as e:
        return f"Failed to reorder layer: {e}"
    return f"Layer '{layer_name}' moved to position {position} in {filename}"


@mcp.tool()
async def set_layer_blend_mode(filename: str, layer_name: str, mode: str) -> str:
    """Set a layer's blend mode.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to modify
        mode: One of LibreSprite's blend modes: normal, multiply, screen,
            overlay, darken, lighten, color_dodge, color_burn, hard_light,
            soft_light, difference, exclusion, hue, saturation, color,
            luminosity (Aseprite's addition/subtract/divide don't exist in
            LibreSprite)
    """
    key = mode.strip().lower().replace(" ", "_").replace("-", "_")
    key = _BLEND_ALIASES.get(key, key)
    if key not in _BLEND_MODES:
        return f"Unknown blend mode '{mode}'. Valid modes: {', '.join(ase.BLEND_MODES)}"
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            if layer.is_background:
                raise ToolError("Background layers always use normal blending")
            layer.blend_mode = _BLEND_MODES[key]
    except ToolError as e:
        return f"Failed to set blend mode: {e}"
    return f"Layer '{layer_name}' blend mode set to {key} in {filename}"


def _layer_rgba(spr: ase.Sprite, layer: ase.Layer, frame: int) -> np.ndarray | None:
    """A layer's cel on the canvas as int32 RGBA, or None without a cel."""
    cel = layer.cels.get(frame)
    if cel is None:
        return None
    out = np.zeros((spr.height, spr.width, 4), np.int32)
    x0, y0 = max(cel.x, 0), max(cel.y, 0)
    x1, y1 = min(cel.x + cel.width, spr.width), min(cel.y + cel.height, spr.height)
    if x0 < x1 and y0 < y1:
        out[y0:y1, x0:x1] = render.cel_rgba(spr, cel.image[y0 - cel.y:y1 - cel.y, x0 - cel.x:x1 - cel.x])
    return out


@mcp.tool()
async def merge_layer_down(filename: str, layer_name: str) -> str:
    """Merge a layer into the layer directly below it.

    The upper layer is composited with its opacity, cel opacity and blend
    mode into the lower layer (which keeps its name and settings) on every
    frame, then removed.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to merge down (must not be the bottom layer)
    """
    try:
        with edit(filename) as spr:
            upper = get_layer(spr, layer_name)
            index = layer_index(spr, upper)
            if index == 0:
                raise ToolError("Layer is the bottom layer; nothing to merge into")
            lower = spr.layers[index - 1]
            for frame in sorted(set(upper.cels) | set(lower.cels)):
                top = _layer_rgba(spr, upper, frame)
                base = _layer_rgba(spr, lower, frame)
                if top is None:
                    continue
                if base is None:
                    base = np.zeros_like(top)
                cel = upper.cels[frame]
                merged = render.blend(base, top, render.mul_un8(cel.opacity, upper.opacity), upper.blend_mode)
                lower.cels[frame] = ase.Cel(rgba_to_image(spr, merged))
            spr.layers.remove(upper)
    except ToolError as e:
        return f"Failed to merge layer down: {e}"
    return f"Layer '{layer_name}' merged down in {filename}"


@mcp.tool()
async def flatten_sprite(filename: str) -> str:
    """Flatten all visible layers into a single layer named "Flattened".

    Hidden layers are discarded, as in LibreSprite's Flatten command. The
    result is a background layer when the sprite had one.

    Args:
        filename: Sprite file to modify
    """
    try:
        with edit(filename) as spr:
            hidden = [l.name for l in spr.layers if not l.visible]
            had_background = bool(spr.layers) and spr.layers[0].is_background and spr.layers[0].visible
            flat = ase.Layer("Flattened")
            flat.is_background = had_background
            for frame in range(spr.frame_count):
                rgba = render.composite(spr, frame)
                if had_background or rgba[..., 3].any():
                    flat.cels[frame] = ase.Cel(rgba_to_image(spr, rgba))
            spr.layers = [flat]
    except ToolError as e:
        return f"Failed to flatten sprite: {e}"
    dropped = f" (discarded hidden layers: {', '.join(hidden)})" if hidden else ""
    return f"Sprite flattened in {filename}{dropped}"
