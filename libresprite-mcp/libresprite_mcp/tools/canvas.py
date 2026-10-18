import os

from ..core import ase
from ..core.sprite import (ToolError, edit, get_layer, find_layer, frame0, new_sprite,
                           reject_traversal, save, set_active, load)
from .. import mcp


@mcp.tool()
async def create_canvas(width: int, height: int, filename: str = "canvas.ase", color_mode: str = "rgb") -> str:
    """Create a new LibreSprite sprite with the specified dimensions.

    The sprite starts with one frame, one transparent layer named "Layer 1"
    and the DawnBringer 32 palette. Files are saved as .ase (".aseprite"
    also works; LibreSprite opens both).

    Args:
        width: Width of the canvas in pixels
        height: Height of the canvas in pixels
        filename: Name of the output file (default: canvas.ase)
        color_mode: "rgb" (default), "gray" or "indexed"
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    if width > 65535 or height > 65535:
        return "Width and height must be <= 65535"
    err = reject_traversal(filename)
    if err:
        return err
    mode = {"grayscale": ase.GRAY}.get(color_mode.lower(), color_mode.lower())
    if mode not in ase.COLOR_MODES:
        return f"Unknown color_mode '{color_mode}': use rgb, gray or indexed"

    try:
        save(new_sprite(width, height, mode), filename)
    except (ToolError, OSError) as e:
        return f"Failed to create canvas: {e}"
    return f"Canvas created successfully: {filename}"


@mcp.tool()
async def add_layer(filename: str, layer_name: str) -> str:
    """Add a new layer on top of the layer stack.

    LibreSprite has no layer groups, so all layers live in one flat stack.

    Args:
        filename: Name of the sprite file to modify
        layer_name: Name of the new layer
    """
    if not layer_name:
        return "Layer name cannot be empty"
    try:
        with edit(filename) as spr:
            if find_layer(spr, layer_name):
                raise ToolError(f"A layer named '{layer_name}' already exists")
            spr.layers.append(ase.Layer(layer_name))
    except ToolError as e:
        return f"Failed to add layer: {e}"
    return f"Layer '{layer_name}' added to {filename}"


@mcp.tool()
async def add_frame(filename: str) -> str:
    """Add a new empty frame at the end, with the last frame's duration.

    Args:
        filename: Name of the sprite file to modify
    """
    try:
        with edit(filename) as spr:
            if spr.frame_count >= 65535:
                raise ToolError("The sprite already has the maximum number of frames")
            spr.durations.append(spr.durations[-1])
    except ToolError as e:
        return f"Failed to add frame: {e}"
    return f"New frame added successfully to {filename}"


@mcp.tool()
async def set_frame(filename: str, frame_index: int) -> str:
    """Set the active frame by index (1-based).

    LibreSprite does not store an active frame in the file; the choice is
    remembered by this server for draw_pixels, draw_line and the other
    tools that draw on the active cel.

    Args:
        filename: Name of the sprite file
        frame_index: Frame index starting at 1
    """
    try:
        frame0(load(filename), frame_index)
    except ToolError as e:
        return f"Failed to set frame: {e}"
    set_active(filename, frame=frame_index)
    return f"Active frame set to {frame_index} in {filename}"


@mcp.tool()
async def set_frame_duration(filename: str, frame_index: int, duration_ms: int) -> str:
    """Set the duration of a frame in milliseconds.

    Args:
        filename: Name of the sprite file to modify
        frame_index: Frame index starting at 1
        duration_ms: Duration in milliseconds (1-65535)
    """
    if not 1 <= duration_ms <= 65535:
        return "Duration must be between 1 and 65535 ms"
    try:
        with edit(filename) as spr:
            spr.durations[frame0(spr, frame_index)] = duration_ms
    except ToolError as e:
        return f"Failed to set frame duration: {e}"
    return f"Frame {frame_index} duration set to {duration_ms}ms in {filename}"


@mcp.tool()
async def set_layer(filename: str, layer_name: str, create_if_missing: bool = False) -> str:
    """Set the active layer by name.

    LibreSprite does not store an active layer in the file; the choice is
    remembered by this server for the tools that draw on the active cel.

    Args:
        filename: Name of the sprite file
        layer_name: Layer name to activate
        create_if_missing: Create the layer (on top) if it does not exist
    """
    try:
        if create_if_missing:
            with edit(filename) as spr:
                if not find_layer(spr, layer_name):
                    spr.layers.append(ase.Layer(layer_name))
        else:
            get_layer(load(filename), layer_name)
    except ToolError as e:
        return f"Failed to set layer: {e}"
    set_active(filename, layer=layer_name)
    return f"Active layer set to '{layer_name}' in {filename}"
