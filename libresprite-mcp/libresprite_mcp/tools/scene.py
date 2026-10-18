from typing import List

from ..core import ase
from ..core.colors import image_to_rgba, rgba_to_image
from ..core.sprite import ToolError, edit, find_layer, load, reject_traversal
from .. import mcp


def _copy_cel(src_spr: ase.Sprite, dst_spr: ase.Sprite, cel: ase.Cel) -> ase.Cel:
    """Copy a cel into another sprite, converting pixels if the color modes differ."""
    copy = cel.copy()
    if src_spr.color_mode != dst_spr.color_mode or (
            src_spr.color_mode == ase.INDEXED and src_spr.palette != dst_spr.palette):
        copy.image = rgba_to_image(dst_spr, image_to_rgba(src_spr, cel.image))
    return copy


@mcp.tool()
async def copy_layers_between_sprites(
    source_filename: str,
    target_filename: str,
    layer_names: List[str],
    replace: bool = True,
    create_missing_frames: bool = True
) -> str:
    """Copy layers by name from a source sprite to a target sprite.

    Cels keep their position, opacity and links; pixels are converted when
    the two sprites use different color modes (or, for indexed sprites,
    different palettes). A layer missing from the target is created on top
    with the source layer's visibility, opacity and blend mode.

    Args:
        source_filename: Source sprite file
        target_filename: Target sprite file
        layer_names: List of layer names to copy
        replace: Clear the target layer's existing cels first (default True);
            when False, frames where the target layer already has a cel keep it
        create_missing_frames: Add empty frames to the target if the source
            has more (they take the source frames' durations)
    """
    if not layer_names:
        return "Layer names list cannot be empty"
    err = reject_traversal(source_filename) or reject_traversal(target_filename)
    if err:
        return err

    missing: list[str] = []
    try:
        src = load(source_filename)
        valid = [name for name in layer_names if find_layer(src, name)]
        missing = [name for name in layer_names if not find_layer(src, name)]
        if not valid:
            raise ToolError("None of the requested layers exist in the source: " + ", ".join(missing))
        with edit(target_filename) as dst:
            if create_missing_frames:
                dst.durations.extend(src.durations[dst.frame_count:])
            for name in valid:
                src_layer = find_layer(src, name)
                dst_layer = find_layer(dst, name)
                if dst_layer is None:
                    dst_layer = ase.Layer(name, flags=src_layer.flags & ~(ase.FLAG_BACKGROUND | ase.FLAG_LOCK_MOVEMENT),
                                          blend_mode=src_layer.blend_mode, opacity=src_layer.opacity)
                    dst.layers.append(dst_layer)
                if replace:
                    dst_layer.cels.clear()
                copies: dict[int, ase.Cel] = {}  # id(source cel) -> copy, keeps links linked
                for frame, cel in sorted(src_layer.cels.items()):
                    if frame >= dst.frame_count or frame in dst_layer.cels:
                        continue
                    if id(cel) not in copies:
                        copies[id(cel)] = _copy_cel(src, dst, cel)
                    dst_layer.cels[frame] = copies[id(cel)]
    except ToolError as e:
        return f"Failed to copy layers: {e}"

    msg = f"Layers copied from {source_filename} to {target_filename}"
    if missing:
        msg += f" (skipped missing layers: {', '.join(missing)})"
    return msg
