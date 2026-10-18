import json
import os

import numpy as np

from ..core import render
from ..core.sprite import ToolError, frame0, load, reject_traversal
from .. import mcp


def _tinted(rgba: np.ndarray, tint: str) -> np.ndarray:
    """LibreSprite's onion-skin tints (rgba_blender_red_tint / blue_tint)."""
    r, g, b, a = (rgba[..., i].astype(np.int32) for i in range(4))
    v = (r * 2126 + g * 7152 + b * 722) // 10000
    if tint == "red":
        channels = ((255 + v) // 2, v // 2, v // 2)
    else:
        channels = (v // 2, v // 2, (255 + v) // 2)
    return np.stack(channels + (a,), axis=-1)


@mcp.tool()
async def render_onion_skin(
    filename: str,
    frame_index: int,
    output_filename: str,
    before: int = 1,
    after: int = 1,
    scale: int = 4,
    ghost_opacity: int = 100,
    tint: bool = False,
) -> str:
    """Render a frame with neighboring frames as translucent onion-skin ghosts.

    Produces a PNG of the given frame (on white) composited over ghosted
    copies of the surrounding frames: the file equivalent of LibreSprite's
    onion skinning. Essential for checking motion continuity while
    animating: export it, open the PNG, and verify the in-between
    positions line up.

    Args:
        filename: Sprite file to read
        frame_index: Frame to render, starting at 1
        output_filename: Output PNG path
        before: Number of previous frames to ghost (default 1)
        after: Number of following frames to ghost (default 1)
        scale: Integer nearest-neighbor scale factor (default 4)
        ghost_opacity: Opacity of ghost frames, 0-255 (default 100)
        tint: Tint previous frames red and following frames blue, like
            LibreSprite's "red/blue tint" onion skin (default False)
    """
    if scale < 1 or scale > 64:
        return "scale must be between 1 and 64"
    if not (0 <= ghost_opacity <= 255):
        return "ghost_opacity must be between 0 and 255"
    if before < 0 or after < 0:
        return "before and after must be >= 0"
    err = reject_traversal(output_filename)
    if err:
        return err
    if not output_filename.lower().endswith(".png"):
        output_filename = f"{output_filename}.png"

    try:
        spr = load(filename)
        idx = frame0(spr, frame_index)
        comp = np.full((spr.height, spr.width, 4), 255, np.int32)
        # Ghosts: oldest first so closer frames draw on top
        for offset in range(before, 0, -1):
            if idx - offset >= 0:
                ghost = render.composite(spr, idx - offset).astype(np.int32)
                comp = render.blend_normal(comp, _tinted(ghost, "red") if tint else ghost, ghost_opacity)
        for offset in range(after, 0, -1):
            if idx + offset < spr.frame_count:
                ghost = render.composite(spr, idx + offset).astype(np.int32)
                comp = render.blend_normal(comp, _tinted(ghost, "blue") if tint else ghost, ghost_opacity)
        comp = render.blend_normal(comp, render.composite(spr, idx).astype(np.int32), 255)
        out_dir = os.path.dirname(os.path.abspath(output_filename))
        os.makedirs(out_dir, exist_ok=True)
        render.to_pil(render.scale_nearest(comp.astype(np.uint8), scale)).save(output_filename)
    except ToolError as e:
        return f"Failed to render onion skin: {e}"
    except OSError as e:
        return f"Failed to render onion skin: {e}"
    return (
        f"Onion-skin render of frame {frame_index} "
        f"(-{before}/+{after} ghosts) saved to {output_filename} at {scale}x"
    )


@mcp.tool()
async def compare_frames(filename: str, frame_a: int, frame_b: int) -> str:
    """Compare two frames and report how much they differ.

    Diffs the two flattened frames (all visible layers) pixel by pixel.
    Use while animating to confirm a frame actually changed, or that it
    did not change too much. Two fully transparent pixels count as equal.

    Args:
        filename: Sprite file to read
        frame_a: First frame index, starting at 1
        frame_b: Second frame index, starting at 1

    Returns:
        JSON with changed pixel count, total pixels, percent changed,
        and the bounding box of the changed region.
    """
    try:
        spr = load(filename)
        a = render.composite(spr, frame0(spr, frame_a))
        b = render.composite(spr, frame0(spr, frame_b))
    except ToolError as e:
        return f"Failed to compare frames: {e}"

    both_clear = (a[..., 3] == 0) & (b[..., 3] == 0)
    diff = np.any(a != b, axis=-1) & ~both_clear
    changed = int(diff.sum())
    total = spr.width * spr.height
    result = {
        "frame_a": frame_a,
        "frame_b": frame_b,
        "changed_pixels": changed,
        "total_pixels": total,
        "percent_changed": round(changed / total * 100, 2) if total else 0,
    }
    if changed:
        ys, xs = np.nonzero(diff)
        result["changed_bounds"] = {
            "x": int(xs.min()),
            "y": int(ys.min()),
            "width": int(xs.max() - xs.min()) + 1,
            "height": int(ys.max() - ys.min()) + 1,
        }
    return json.dumps(result)


@mcp.tool()
async def get_color_stats(filename: str, frame_index: int = 1, top: int = 16) -> str:
    """Report the colors used in a frame and how often each appears.

    Histograms the flattened frame (all visible layers). Use to check
    palette discipline: too many near-duplicate colors is a common
    pixel-art mistake.

    Args:
        filename: Sprite file to read
        frame_index: Frame to analyze, starting at 1 (default 1)
        top: How many of the most-used colors to list (default 16)

    Returns:
        JSON with unique color count, opaque pixel count, and the top
        colors with usage counts.
    """
    if top < 1:
        return "top must be >= 1"
    try:
        spr = load(filename)
        flat = render.composite(spr, frame0(spr, frame_index))
    except ToolError as e:
        return f"Failed to get color stats: {e}"

    opaque = flat[flat[..., 3] > 0][:, :3]
    if len(opaque):
        colors, counts = np.unique(opaque, axis=0, return_counts=True)
    else:
        colors, counts = np.zeros((0, 3), np.uint8), np.zeros(0, np.int64)
    ranked = sorted(
        ({"color": "#{:02X}{:02X}{:02X}".format(*(int(c) for c in rgb)), "count": int(n)}
         for rgb, n in zip(colors, counts)),
        key=lambda c: (-c["count"], c["color"]),
    )
    return json.dumps({
        "frame": frame_index,
        "unique_colors": len(ranked),
        "opaque_pixels": int(len(opaque)),
        "top_colors": ranked[:top],
    })
