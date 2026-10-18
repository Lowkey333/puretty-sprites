import json
import math
import os
from typing import List

from ..core import ase
from ..core.sprite import (ToolError, blank_image, edit, frame0, frame_range, get_cel, get_layer, load)
from .. import mcp
from .transform import _scale_nearest

_EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out", "smoothstep")
_DIRECTIONS = {"forward": 0, "reverse": 1, "pingpong": 2, "ping_pong": 2, "ping-pong": 2}


def _ease(mode: str, t: float) -> float:
    if mode == "linear":
        return t
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1 - (1 - t) * (1 - t)
    if mode == "ease_in_out":
        return 2 * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 2) / 2
    return t * t * (3 - 2 * t)


def _round(v: float) -> int:
    # Round half up; the epsilon stops float noise from sending cos(240°) =
    # -0.5000000000000004 to -1 while cos(120°) = -0.4999999999999998 gives 0.
    return math.floor(v + 0.5 + 1e-9)


def _cel_for(spr: ase.Sprite, layer: ase.Layer, frame: int, create: bool,
             source_frame: int | None) -> ase.Cel | None:
    """The cel at frame; when missing and create, a copy of the source frame's
    cel (default: the range start) or an empty canvas-sized cel."""
    cel = layer.cels.get(frame)
    if cel is None and create:
        source = layer.cels.get(source_frame) if source_frame is not None else None
        cel = source.copy() if source else ase.Cel(blank_image(spr, spr.width, spr.height))
        layer.cels[frame] = cel
    return cel


def _source0(spr: ase.Sprite, source_frame_index: int | None, default0: int) -> int:
    return default0 if source_frame_index is None else frame0(spr, source_frame_index)


def _unlink(layer: ase.Layer, frame: int) -> ase.Cel:
    """Give a linked cel its own copy before changing position or opacity."""
    cel = layer.cels[frame]
    if any(f != frame and c is cel for f, c in layer.cels.items()):
        cel = layer.cels[frame] = cel.copy()
    return cel


@mcp.tool()
async def add_frames(filename: str, count: int, duration_ms: int | None = None, copy_previous: bool = False) -> str:
    """Add multiple frames to the end of a sprite.

    New frames are empty unless copy_previous is set, which copies the
    last frame's cels into each new frame (LibreSprite's "New Frame").

    Args:
        filename: Sprite file to modify
        count: Number of frames to add
        duration_ms: Optional duration for each new frame (default: the last frame's)
        copy_previous: Copy the last frame's cels into the new frames
    """
    if count < 1:
        return "Count must be >= 1"
    if duration_ms is not None and not 1 <= duration_ms <= 65535:
        return "Duration must be between 1 and 65535 ms"
    try:
        with edit(filename) as spr:
            if spr.frame_count + count > 65535:
                raise ToolError("A sprite can have at most 65535 frames")
            last = spr.frame_count - 1
            for _ in range(count):
                spr.durations.append(duration_ms or spr.durations[last])
                if copy_previous:
                    for layer in spr.layers:
                        if last in layer.cels:
                            layer.cels[spr.frame_count - 1] = layer.cels[last].copy()
    except ToolError as e:
        return f"Failed to add frames: {e}"
    return f"Added {count} frames to {filename}"


@mcp.tool()
async def set_frame_duration_all(filename: str, duration_ms: int) -> str:
    """Set the duration of all frames in milliseconds.

    Args:
        filename: Sprite file to modify
        duration_ms: Duration in milliseconds (1-65535)
    """
    if not 1 <= duration_ms <= 65535:
        return "Duration must be between 1 and 65535 ms"
    try:
        with edit(filename) as spr:
            spr.durations = [duration_ms] * spr.frame_count
    except ToolError as e:
        return f"Failed to set frame durations: {e}"
    return f"Set duration of all frames to {duration_ms}ms in {filename}"


@mcp.tool()
async def set_layer_visibility(filename: str, layer_name: str, visible: bool = True) -> str:
    """Set layer visibility by name.

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        visible: Whether the layer should be visible
    """
    try:
        with edit(filename) as spr:
            get_layer(spr, layer_name).visible = visible
    except ToolError as e:
        return f"Failed to set layer visibility: {e}"
    return f"Layer '{layer_name}' visibility set to {visible} in {filename}"


@mcp.tool()
async def set_layer_opacity(filename: str, layer_name: str, opacity: int) -> str:
    """Set layer opacity by name (0-255).

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        opacity: Opacity value 0-255
    """
    if opacity < 0 or opacity > 255:
        return "Opacity must be between 0 and 255"
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            if layer.is_background:
                raise ToolError("Background layers are always fully opaque")
            layer.opacity = opacity
    except ToolError as e:
        return f"Failed to set layer opacity: {e}"
    return f"Layer '{layer_name}' opacity set to {opacity} in {filename}"


@mcp.tool()
async def get_sprite_info(filename: str) -> str:
    """Return sprite info as JSON (size, color mode, frame durations, layers, tags).

    Frame numbers in the reply are 1-based, like the tools' frame_index.
    "warnings" lists anything LibreSprite would not keep (e.g. layer groups
    from an Aseprite file).

    Args:
        filename: Sprite file to inspect
    """
    try:
        spr = load(filename)
    except ToolError as e:
        return f"Failed to get sprite info: {e}"
    info = {
        "width": spr.width,
        "height": spr.height,
        "color_mode": spr.color_mode,
        "frames": spr.frame_count,
        "durations_ms": spr.durations,
        "layers": [
            {
                "name": layer.name,
                "visible": layer.visible,
                "opacity": layer.opacity,
                "blend_mode": ase.BLEND_MODES[layer.blend_mode],
                "is_background": layer.is_background,
                "cel_frames": sorted(f + 1 for f in layer.cels),
            }
            for layer in spr.layers
        ],
        "tags": [
            {"name": t.name, "from": t.from_frame + 1, "to": t.to_frame + 1,
             "direction": ase.DIRECTIONS[t.direction]}
            for t in spr.tags
        ],
        "palette_size": len(spr.palette),
    }
    if spr.color_mode == ase.INDEXED:
        info["transparent_index"] = spr.transparent_index
    if spr.warnings:
        info["warnings"] = spr.warnings
    return json.dumps(info)


@mcp.tool()
async def duplicate_frame_range(filename: str, start_frame: int, end_frame: int, times: int = 1) -> str:
    """Duplicate a frame range and append copies to the end.

    Args:
        filename: Sprite file to modify
        start_frame: Starting frame index (1-based)
        end_frame: Ending frame index (1-based, inclusive)
        times: Number of times to append the range (default: 1)
    """
    if times < 1:
        return "Times must be >= 1"
    try:
        with edit(filename) as spr:
            frames = frame_range(spr, start_frame, end_frame)
            for _ in range(times):
                for source in frames:
                    spr.durations.append(spr.durations[source])
                    new = spr.frame_count - 1
                    for layer in spr.layers:
                        if source in layer.cels:
                            layer.cels[new] = layer.cels[source].copy()
            if spr.frame_count > 65535:
                raise ToolError("A sprite can have at most 65535 frames")
    except ToolError as e:
        return f"Failed to duplicate frame range: {e}"
    return f"Duplicated frames {start_frame}-{end_frame} (x{times}) in {filename}"


@mcp.tool()
async def set_cel_position(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    create_if_missing: bool = False,
    source_frame_index: int | None = None
) -> str:
    """Set a cel's position in a specific layer and frame.

    Positions are absolute: a cel drawn with the drawing tools sits at
    (0, 0), so (2, -1) shifts that art 2px right and 1px up.

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        frame_index: Frame index starting at 1
        x: X position in pixels
        y: Y position in pixels
        create_if_missing: Create the cel if it does not exist
        source_frame_index: Optional frame to copy the new cel's image from
    """
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            frame = frame0(spr, frame_index)
            cel = _cel_for(spr, layer, frame, create_if_missing, _source0(spr, source_frame_index, frame))
            if cel is None:
                raise ToolError(f"No cel on '{layer_name}' at frame {frame_index} (create_if_missing is off)")
            cel = _unlink(layer, frame)
            cel.x, cel.y = x, y
    except ToolError as e:
        return f"Failed to set cel position: {e}"
    return f"Cel position set to ({x}, {y}) on '{layer_name}' frame {frame_index} in {filename}"


def _tween_positions(filename, layer_name, start_frame, end_frame, start_x, start_y, end_x, end_y,
                     easing, create_missing_cels, source_frame_index) -> int:
    with edit(filename) as spr:
        layer = get_layer(spr, layer_name)
        frames = frame_range(spr, start_frame, end_frame)
        source = _source0(spr, source_frame_index, frames.start)
        span = len(frames) - 1
        moved = 0
        for i, frame in enumerate(frames):
            e = _ease(easing, i / span if span > 0 else 0)
            if _cel_for(spr, layer, frame, create_missing_cels, source) is None:
                continue
            cel = _unlink(layer, frame)
            cel.x = _round(start_x + (end_x - start_x) * e)
            cel.y = _round(start_y + (end_y - start_y) * e)
            moved += 1
        if moved == 0:
            raise ToolError(f"'{layer_name}' has no cels in frames {start_frame}-{end_frame} "
                            "(pass create_missing_cels=True)")
    return moved


@mcp.tool()
async def tween_cel_positions(
    filename: str,
    layer_name: str,
    start_frame: int,
    end_frame: int,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    create_missing_cels: bool = False,
    source_frame_index: int | None = None
) -> str:
    """Tween cel positions linearly across a frame range.

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        start_frame: First frame (1-based)
        end_frame: Last frame (1-based, inclusive)
        start_x: Cel x at start_frame
        start_y: Cel y at start_frame
        end_x: Cel x at end_frame
        end_y: Cel y at end_frame
        create_missing_cels: Create missing cels (copied from the source frame)
        source_frame_index: Frame to copy missing cels from (default: start_frame)
    """
    try:
        _tween_positions(filename, layer_name, start_frame, end_frame, start_x, start_y, end_x, end_y,
                         "linear", create_missing_cels, source_frame_index)
    except ToolError as e:
        return f"Failed to tween cel positions: {e}"
    return f"Tweened cel positions on '{layer_name}' frames {start_frame}-{end_frame} in {filename}"


@mcp.tool()
async def offset_cel_positions(
    filename: str,
    layer_name: str,
    start_frame: int,
    end_frame: int,
    dx: int,
    dy: int
) -> str:
    """Offset cel positions by a delta across a frame range.

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        start_frame: First frame (1-based)
        end_frame: Last frame (1-based, inclusive)
        dx: Pixels to move right (negative = left)
        dy: Pixels to move down (negative = up)
    """
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            moved = 0
            done: set[int] = set()
            for frame in frame_range(spr, start_frame, end_frame):
                cel = layer.cels.get(frame)
                if cel is None or id(cel) in done:
                    continue
                done.add(id(cel))
                cel.x += dx
                cel.y += dy
                moved += 1
            if moved == 0:
                raise ToolError(f"'{layer_name}' has no cels in frames {start_frame}-{end_frame}")
    except ToolError as e:
        return f"Failed to offset cel positions: {e}"
    return f"Offset cel positions by ({dx}, {dy}) on '{layer_name}' frames {start_frame}-{end_frame} in {filename}"


@mcp.tool()
async def create_cel(filename: str, layer_name: str, frame_index: int, x: int = 0, y: int = 0) -> str:
    """Create an empty canvas-sized cel on a layer/frame.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to create the cel on
        frame_index: Frame index starting at 1
        x: Cel x position (default 0)
        y: Cel y position (default 0)
    """
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            frame = frame0(spr, frame_index)
            if frame in layer.cels:
                raise ToolError(f"'{layer_name}' already has a cel at frame {frame_index}")
            layer.cels[frame] = ase.Cel(blank_image(spr, spr.width, spr.height), x, y)
    except ToolError as e:
        return f"Failed to create cel: {e}"
    return f"Cel created on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def clear_cel(filename: str, layer_name: str, frame_index: int) -> str:
    """Delete a cel on a layer/frame.

    Args:
        filename: Sprite file to modify
        layer_name: Layer of the cel
        frame_index: Frame index starting at 1
    """
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            frame = frame0(spr, frame_index)
            get_cel(spr, layer, frame)
            if layer.is_background:
                raise ToolError("Cannot delete a background layer's cel; erase_region it instead")
            del layer.cels[frame]
    except ToolError as e:
        return f"Failed to clear cel: {e}"
    return f"Cel cleared on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def copy_cel(
    filename: str,
    layer_name: str,
    source_frame: int,
    target_frame: int,
    replace: bool = True
) -> str:
    """Copy a cel from one frame to another on the same layer.

    Args:
        filename: Sprite file to modify
        layer_name: Layer of the cel
        source_frame: Frame to copy from (1-based)
        target_frame: Frame to copy to (1-based)
        replace: Overwrite an existing cel at the target (default True)
    """
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            src, dst = frame0(spr, source_frame), frame0(spr, target_frame)
            cel = get_cel(spr, layer, src)
            if dst in layer.cels and not replace:
                raise ToolError(f"'{layer_name}' already has a cel at frame {target_frame} (replace is off)")
            if src != dst:
                layer.cels[dst] = cel.copy()
    except ToolError as e:
        return f"Failed to copy cel: {e}"
    return f"Cel copied on '{layer_name}' from frame {source_frame} to {target_frame} in {filename}"


def _copy_frame_cels(spr: ase.Sprite, src: int, dst: int, overwrite: bool) -> None:
    for layer in spr.layers:
        if overwrite and dst in layer.cels and not layer.is_background:
            del layer.cels[dst]
        if src in layer.cels and (overwrite or dst not in layer.cels):
            layer.cels[dst] = layer.cels[src].copy()


@mcp.tool()
async def copy_frame(
    filename: str,
    source_frame: int,
    target_frame: int | None = None,
    overwrite: bool = True
) -> str:
    """Copy all cels from a source frame to a target frame (or append a new frame).

    Args:
        filename: Sprite file to modify
        source_frame: Frame to copy (1-based)
        target_frame: Destination frame (1-based); omit to append a new frame
        overwrite: Clear the target frame first (default True); when False
            only layers without a cel at the target receive one
    """
    try:
        with edit(filename) as spr:
            src = frame0(spr, source_frame)
            if target_frame is None:
                spr.durations.append(spr.durations[src])
                dst = spr.frame_count - 1
            else:
                dst = frame0(spr, target_frame)
            if src != dst:
                _copy_frame_cels(spr, src, dst, overwrite)
    except ToolError as e:
        return f"Failed to copy frame: {e}"
    if target_frame is None:
        return f"Frame {source_frame} copied to new frame in {filename}"
    return f"Frame {source_frame} copied to frame {target_frame} in {filename}"


@mcp.tool()
async def propagate_frame_to_range(
    filename: str,
    source_frame: int,
    start_frame: int,
    end_frame: int,
    overwrite: bool = True
) -> str:
    """Copy all layers from a source frame to a range of frames.

    Args:
        filename: Sprite file to modify
        source_frame: Frame to copy (1-based)
        start_frame: First destination frame (1-based)
        end_frame: Last destination frame (1-based, inclusive)
        overwrite: Clear each destination frame first (default True); when
            False only layers without a cel there receive one
    """
    try:
        with edit(filename) as spr:
            src = frame0(spr, source_frame)
            for dst in frame_range(spr, start_frame, end_frame):
                if dst != src:
                    _copy_frame_cels(spr, src, dst, overwrite)
    except ToolError as e:
        return f"Failed to propagate frame range: {e}"
    return f"Propagated frame {source_frame} to frames {start_frame}-{end_frame} in {filename}"


@mcp.tool()
async def set_tag(filename: str, name: str, from_frame: int, to_frame: int, direction: str = "forward") -> str:
    """Create or update an animation tag on the sprite.

    Args:
        filename: Sprite file to modify
        name: Tag name (an existing tag with this name is updated)
        from_frame: First frame (1-based)
        to_frame: Last frame (1-based, inclusive)
        direction: "forward", "reverse" or "pingpong" (LibreSprite has no
            pingpong_reverse)
    """
    key = direction.strip().lower()
    if key not in _DIRECTIONS:
        return f"Unsupported direction '{direction}' (forward, reverse, pingpong)"
    if not name:
        return "Tag name cannot be empty"
    try:
        with edit(filename) as spr:
            frames = frame_range(spr, from_frame, to_frame)
            tag = next((t for t in spr.tags if t.name == name), None)
            if tag is None:
                tag = ase.Tag(name, frames.start, frames.stop - 1)
                spr.tags.append(tag)
            tag.from_frame, tag.to_frame = frames.start, frames.stop - 1
            tag.direction = _DIRECTIONS[key]
    except ToolError as e:
        return f"Failed to set tag: {e}"
    return f"Tag '{name}' set to frames {from_frame}-{to_frame} (direction={ase.DIRECTIONS[_DIRECTIONS[key]]}) in {filename}"


def _libresprite_ini() -> str:
    base = os.getenv("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "LibreSprite", "libresprite.ini")


@mcp.tool()
async def set_onion_skin(filename: str, enabled: bool = True, before: int = 2, after: int = 2, opacity: int = 128) -> str:
    """Configure LibreSprite's onion skin defaults (editor display only).

    LibreSprite keeps onion skin settings in its preferences, not in the
    sprite, so this updates the [onionskin] section of libresprite.ini; it
    applies the next time LibreSprite starts (close LibreSprite first or it
    overwrites the file on exit). For a batch preview use render_onion_skin.

    Args:
        filename: Sprite the settings are meant for (checked to exist)
        enabled: Turn onion skin on
        before: Frames shown before the current one
        after: Frames shown after the current one
        opacity: Opacity of the nearest ghost (0-255)
    """
    if not os.path.isfile(filename):
        return f"File {filename} not found"
    if not 0 <= opacity <= 255 or before < 0 or after < 0:
        return "before/after must be >= 0 and opacity 0-255"
    path = _libresprite_ini()
    values = {"active": str(enabled).lower(), "prev_frames": str(before),
              "next_frames": str(after), "opacity_base": str(opacity)}
    lines = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    out, in_section, seen = [], False, set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_section:
                out += [f"{k} = {v}" for k, v in values.items() if k not in seen]
            in_section = stripped.lower() == "[onionskin]"
        elif in_section and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                line = f"{key} = {values[key]}"
                seen.add(key)
        out.append(line)
    if in_section:
        out += [f"{k} = {v}" for k, v in values.items() if k not in seen]
    elif "[onionskin]" not in (l.strip().lower() for l in lines):
        out += ["", "[onionskin]"] + [f"{k} = {v}" for k, v in values.items()]
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    except OSError as e:
        return f"Failed to set onion skin: {e}"
    state = "enabled" if enabled else "disabled"
    return f"Onion skin {state} (before={before}, after={after}, opacity={opacity}) in {path}"


@mcp.tool()
async def propagate_cels(
    filename: str,
    layer_names: List[str],
    source_frame: int,
    start_frame: int,
    end_frame: int,
    replace: bool = True
) -> str:
    """Copy cels from a source frame to a range of frames for specific layers.

    Args:
        filename: Sprite file to modify
        layer_names: Layers whose cel is copied
        source_frame: Frame to copy from (1-based)
        start_frame: First destination frame (1-based)
        end_frame: Last destination frame (1-based, inclusive)
        replace: Whether to overwrite existing cels (default: True)
    """
    if not layer_names:
        return "layer_names cannot be empty"
    try:
        with edit(filename) as spr:
            layers = [get_layer(spr, name) for name in layer_names]
            src = frame0(spr, source_frame)
            copied = 0
            for layer in layers:
                cel = get_cel(spr, layer, src)
                for dst in frame_range(spr, start_frame, end_frame):
                    if dst != src and (replace or dst not in layer.cels):
                        layer.cels[dst] = cel.copy()
                        copied += 1
    except ToolError as e:
        return f"Failed to propagate cels: {e}"
    return f"Propagated {copied} cels from frame {source_frame} to frames {start_frame}-{end_frame} in {filename}"


@mcp.tool()
async def tween_cel_positions_eased(
    filename: str,
    layer_name: str,
    start_frame: int,
    end_frame: int,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    easing: str = "smoothstep",
    create_missing_cels: bool = False,
    source_frame_index: int | None = None
) -> str:
    """Tween cel positions with easing across a frame range.

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        start_frame: First frame (1-based)
        end_frame: Last frame (1-based, inclusive)
        start_x: Cel x at start_frame
        start_y: Cel y at start_frame
        end_x: Cel x at end_frame
        end_y: Cel y at end_frame
        easing: linear, ease_in, ease_out, ease_in_out or smoothstep (default)
        create_missing_cels: Create missing cels (copied from the source frame)
        source_frame_index: Frame to copy missing cels from (default: start_frame)
    """
    easing = (easing or "smoothstep").lower().strip()
    if easing not in _EASINGS:
        return "Unsupported easing (linear, ease_in, ease_out, ease_in_out, smoothstep)"
    try:
        _tween_positions(filename, layer_name, start_frame, end_frame, start_x, start_y, end_x, end_y,
                         easing, create_missing_cels, source_frame_index)
    except ToolError as e:
        return f"Failed to tween cel positions with easing: {e}"
    return (f"Tweened cel positions ({easing}) on '{layer_name}' frames {start_frame}-{end_frame} "
            f"in {filename}")


@mcp.tool()
async def oscillate_cel_positions(
    filename: str,
    layer_name: str,
    start_frame: int,
    end_frame: int,
    amplitude_x: int = 0,
    amplitude_y: int = 0,
    cycles: float = 1.0,
    phase_deg: float = 0.0,
    create_missing_cels: bool = False,
    source_frame_index: int | None = None
) -> str:
    """Oscillate cel positions across a frame range using a sine wave.

    Adds dx = amplitude_x * sin(angle), dy = amplitude_y * cos(angle) to each
    cel's current position, with angle running through `cycles` full turns.
    Good for bobbing, breathing and hovering.

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        start_frame: First frame (1-based)
        end_frame: Last frame (1-based, inclusive)
        amplitude_x: Horizontal amplitude in pixels
        amplitude_y: Vertical amplitude in pixels
        cycles: Number of full oscillations across the range
        phase_deg: Phase offset in degrees
        create_missing_cels: Create missing cels (copied from the source frame)
        source_frame_index: Frame to copy missing cels from (default: start_frame)
    """
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            frames = frame_range(spr, start_frame, end_frame)
            source = _source0(spr, source_frame_index, frames.start)
            span = len(frames) - 1
            phase = math.radians(phase_deg)
            moved = 0
            for i, frame in enumerate(frames):
                angle = 2 * math.pi * cycles * (i / span if span > 0 else 0) + phase
                if _cel_for(spr, layer, frame, create_missing_cels, source) is None:
                    continue
                cel = _unlink(layer, frame)
                cel.x += _round(amplitude_x * math.sin(angle))
                cel.y += _round(amplitude_y * math.cos(angle))
                moved += 1
            if moved == 0:
                raise ToolError(f"'{layer_name}' has no cels in frames {start_frame}-{end_frame}")
    except ToolError as e:
        return f"Failed to oscillate cel positions: {e}"
    return (f"Oscillated cel positions on '{layer_name}' frames {start_frame}-{end_frame} "
            f"(amp {amplitude_x},{amplitude_y}, {cycles:g} cycles) in {filename}")


@mcp.tool()
async def tween_cel_opacity_eased(
    filename: str,
    layer_name: str,
    start_frame: int,
    end_frame: int,
    start_opacity: int,
    end_opacity: int,
    easing: str = "smoothstep",
    create_missing_cels: bool = False,
    source_frame_index: int | None = None
) -> str:
    """Tween cel opacity with easing across a frame range (fades).

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        start_frame: First frame (1-based)
        end_frame: Last frame (1-based, inclusive)
        start_opacity: Opacity at start_frame (0-255)
        end_opacity: Opacity at end_frame (0-255)
        easing: linear, ease_in, ease_out, ease_in_out or smoothstep (default)
        create_missing_cels: Create missing cels (copied from the source frame)
        source_frame_index: Frame to copy missing cels from (default: start_frame)
    """
    easing = (easing or "smoothstep").lower().strip()
    if easing not in _EASINGS:
        return "Unsupported easing (linear, ease_in, ease_out, ease_in_out, smoothstep)"
    if not (0 <= start_opacity <= 255 and 0 <= end_opacity <= 255):
        return "Opacity must be between 0 and 255"
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            frames = frame_range(spr, start_frame, end_frame)
            source = _source0(spr, source_frame_index, frames.start)
            span = len(frames) - 1
            changed = 0
            for i, frame in enumerate(frames):
                e = _ease(easing, i / span if span > 0 else 0)
                if _cel_for(spr, layer, frame, create_missing_cels, source) is None:
                    continue
                cel = _unlink(layer, frame)
                cel.opacity = max(0, min(255, _round(start_opacity + (end_opacity - start_opacity) * e)))
                changed += 1
            if changed == 0:
                raise ToolError(f"'{layer_name}' has no cels in frames {start_frame}-{end_frame}")
    except ToolError as e:
        return f"Failed to tween cel opacity with easing: {e}"
    return (f"Tweened cel opacity ({easing}) on '{layer_name}' frames {start_frame}-{end_frame} "
            f"in {filename}")


@mcp.tool()
async def tween_cel_scale_eased(
    filename: str,
    layer_name: str,
    start_frame: int,
    end_frame: int,
    start_scale: float,
    end_scale: float,
    easing: str = "smoothstep",
    anchor: str = "center",
    replace: bool = True,
    create_missing_cels: bool = True,
    source_frame_index: int | None = None
) -> str:
    """Tween cel scale with easing across a frame range (nearest-neighbor).

    Each frame gets a scaled copy of the source frame's cel.

    Args:
        filename: Sprite file to modify
        layer_name: Layer name to target
        start_frame: First frame (1-based)
        end_frame: Last frame (1-based, inclusive)
        start_scale: Scale at start_frame (1.0 = original size)
        end_scale: Scale at end_frame
        easing: linear, ease_in, ease_out, ease_in_out or smoothstep (default)
        anchor: "center" (scale around the cel's center) or "topleft"
        replace: Replace existing cels in the range (default True)
        create_missing_cels: Create cels where the range has none (default True)
        source_frame_index: Frame whose cel is scaled (default: start_frame)
    """
    easing = (easing or "smoothstep").lower().strip()
    if easing not in _EASINGS:
        return "Unsupported easing (linear, ease_in, ease_out, ease_in_out, smoothstep)"
    anchor = (anchor or "center").lower().strip()
    if anchor not in ("center", "topleft"):
        return "Unsupported anchor (center, topleft)"
    if start_scale <= 0 or end_scale <= 0:
        return "Scales must be > 0"
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            frames = frame_range(spr, start_frame, end_frame)
            base = get_cel(spr, layer, _source0(spr, source_frame_index, frames.start)).copy()
            span = len(frames) - 1
            written = 0
            for i, frame in enumerate(frames):
                exists = frame in layer.cels
                if (exists and not replace) or (not exists and not create_missing_cels):
                    continue
                scale = start_scale + (end_scale - start_scale) * _ease(easing, i / span if span > 0 else 0)
                new_w = max(1, _round(base.width * scale))
                new_h = max(1, _round(base.height * scale))
                x, y = base.x, base.y
                if anchor == "center":
                    x = _round(base.x + base.width / 2 - new_w / 2)
                    y = _round(base.y + base.height / 2 - new_h / 2)
                layer.cels[frame] = ase.Cel(_scale_nearest(base.image, new_w, new_h), x, y, base.opacity)
                written += 1
    except ToolError as e:
        return f"Failed to tween cel scale with easing: {e}"
    return (f"Tweened cel scale ({easing}) on '{layer_name}' frames {start_frame}-{end_frame} "
            f"({written} cels) in {filename}")


@mcp.tool()
async def delete_frame(filename: str, frame_index: int) -> str:
    """Delete a frame by index; later frames and tags shift down.

    Args:
        filename: Sprite file to modify
        frame_index: Frame to delete (1-based)
    """
    try:
        with edit(filename) as spr:
            frame = frame0(spr, frame_index)
            if spr.frame_count == 1:
                raise ToolError("Cannot delete the only frame")
            del spr.durations[frame]
            for layer in spr.layers:
                layer.cels = {(f - 1 if f > frame else f): c for f, c in layer.cels.items() if f != frame}
            kept = []
            for tag in spr.tags:
                if tag.from_frame == tag.to_frame == frame:
                    continue
                if tag.from_frame > frame:
                    tag.from_frame -= 1
                if tag.to_frame >= frame:
                    tag.to_frame -= 1
                kept.append(tag)
            spr.tags = kept
    except ToolError as e:
        return f"Failed to delete frame: {e}"
    return f"Frame {frame_index} deleted from {filename}"


@mcp.tool()
async def delete_tag(filename: str, name: str) -> str:
    """Delete an animation tag by name.

    Args:
        filename: Sprite file to modify
        name: Tag to delete
    """
    try:
        with edit(filename) as spr:
            tag = next((t for t in spr.tags if t.name == name), None)
            if tag is None:
                raise ToolError(f"Tag not found: {name!r}")
            spr.tags.remove(tag)
    except ToolError as e:
        return f"Failed to delete tag: {e}"
    return f"Tag '{name}' deleted from {filename}"


@mcp.tool()
async def set_cel_opacity(filename: str, layer_name: str, frame_index: int, opacity: int) -> str:
    """Set the opacity of a single cel (0-255).

    Args:
        filename: Sprite file to modify
        layer_name: Layer of the cel
        frame_index: Frame index starting at 1
        opacity: Opacity 0-255
    """
    if opacity < 0 or opacity > 255:
        return "Opacity must be between 0 and 255"
    try:
        with edit(filename) as spr:
            layer = get_layer(spr, layer_name)
            frame = frame0(spr, frame_index)
            get_cel(spr, layer, frame)
            _unlink(layer, frame).opacity = opacity
    except ToolError as e:
        return f"Failed to set cel opacity: {e}"
    return f"Cel opacity set to {opacity} on '{layer_name}' frame {frame_index} in {filename}"
