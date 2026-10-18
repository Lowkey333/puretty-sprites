import glob
import io
import json
import os
from typing import Any, Dict, List

import numpy as np
from mcp.server.mcpserver import Image as McpImage
from PIL import Image

from ..core import ase, pixelize as px, render
from ..core.colors import parse_hex_color, rgba_to_image, to_hex
from ..core.sprite import ToolError, edit, find_layer, frame0, load, reject_traversal, save
from .slices import create_slice, set_slice_pivot
from .. import mcp

# Bridges from generated/rendered images (ComfyUI, Krita, Blender) to
# LibreSprite sprites. See core/pixelize.py for the algorithm.


def _png(rgba: np.ndarray, scale: int) -> bytes:
    buf = io.BytesIO()
    render.to_pil(render.over_checkerboard(render.scale_nearest(rgba, scale), max(4 * scale, 8))).save(buf, "PNG")
    return buf.getvalue()


def _outline(color: str):
    if not color:
        return None
    rgba = parse_hex_color(color)
    if rgba is None:
        raise ToolError(f"Invalid outline_color: {color}")
    return rgba


def _pixelize(rgba, palette_colors, width, height, pixel_size, remove_background, tolerance, outline, clean):
    try:
        return px.pixelize(rgba, width=width, height=height, pixel_size=pixel_size, palette=palette_colors,
                           background=remove_background, tolerance=tolerance, outline=outline, clean=clean)
    except ValueError as e:
        raise ToolError(str(e)) from e


def _palette(spec, images, sprite_palette=None):
    try:
        return px.resolve_palette(spec, images, sprite_palette)
    except ValueError as e:
        raise ToolError(str(e)) from e


@mcp.tool()
async def pixelize_image(
    image_path: str,
    output_path: str,
    width: int = 0,
    height: int = 0,
    pixel_size: int = 0,
    palette: Any = "auto:16",
    remove_background: str = "auto",
    background_tolerance: int = 24,
    outline_color: str = "",
    clean: bool = True,
) -> list[McpImage | str]:
    """Turn an AI-generated or rendered image into real pixel art (PNG).

    Removes a uniform background, snaps colors to a palette, downscales by
    blocks taking each block's dominant color (no blurry in-between
    colors), binarizes alpha and removes stray pixels. Shows you the result.

    Size: give width and/or height of the result (e.g. height=64), or
    pixel_size (how many source pixels make one output pixel), or neither
    to auto-detect the grid of upscaled "fake pixel art".

    Args:
        image_path: Source image (PNG/JPG/WebP)
        output_path: Where to write the pixel-art PNG
        width: Output width in pixels (0 = from height/aspect)
        height: Output height in pixels (0 = from width/aspect)
        pixel_size: Source pixels per output pixel (alternative to width/height)
        palette: "auto:N" (extract N colors, default auto:16), a preset name
            (pico8, endesga32, dawnbringer32, ...), "none" (keep colors), or a
            list of hex colors
        remove_background: "auto" (uniform border color), "none", or a hex color
        background_tolerance: Max per-channel difference counted as background
        outline_color: Optional hex color for a 1px outline around the subject
        clean: Remove isolated single pixels (default True)
    """
    if not os.path.isfile(image_path):
        return [f"Image {image_path} not found"]
    try:
        err = reject_traversal(output_path)
        if err:
            raise ToolError(err)
        src = px.load_rgba(image_path)
        cut = px.remove_background(src, remove_background, background_tolerance) if remove_background != "none" else src
        colors = _palette(palette, [cut])
        result = _pixelize(src, colors, width, height, pixel_size, remove_background,
                           background_tolerance, _outline(outline_color), clean)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        render.to_pil(result.rgba).save(output_path)
    except (ToolError, OSError) as e:
        return [f"Failed to pixelize image: {e}"]
    h, w = result.rgba.shape[:2]
    used = len(np.unique(result.rgba[result.rgba[..., 3] > 0][:, :3], axis=0))
    note = f" (detected {result.block}px grid)" if result.detected else ""
    scale = max(1, min(16, 512 // max(w, h)))
    return [McpImage(data=_png(result.rgba, scale), format="png"),
            f"Pixelized {image_path} -> {output_path}: {w}x{h}, {used} colors{note}"]


@mcp.tool()
async def import_pixelized_image(
    filename: str,
    image_path: str,
    layer_name: str,
    frame_index: int = 1,
    x: int = 0,
    y: int = 0,
    width: int = 0,
    height: int = 0,
    pixel_size: int = 0,
    palette: Any = "sprite",
    remove_background: str = "auto",
    align: str = "none",
    outline_color: str = "",
) -> str:
    """Pixelize an image and place it on a layer of a sprite in one step.

    Same processing as pixelize_image; by default colors snap to the
    sprite's own palette so the import matches the rest of the art. The
    layer is created (on top) when missing and its cel in that frame is
    replaced.

    Args:
        filename: Sprite file to modify
        image_path: Source image
        layer_name: Layer to place the result on
        frame_index: Frame index starting at 1
        x: Left edge of the result (ignored unless align is "none")
        y: Top edge of the result (ignored unless align is "none")
        width: Result width in pixels (0 = from height/aspect)
        height: Result height in pixels (0 = from width/aspect)
        pixel_size: Source pixels per output pixel (alternative to width/height)
        palette: "sprite" (default), "auto:N", a preset name, "none" or hex list
        remove_background: "auto", "none" or a hex color
        align: "none" (use x/y), "bottom" (centered on the bottom edge) or "center"
        outline_color: Optional hex color for a 1px outline
    """
    if not os.path.isfile(image_path):
        return f"Image {image_path} not found"
    if align not in ("none", "bottom", "center"):
        return "align must be 'none', 'bottom' or 'center'"
    try:
        src = px.load_rgba(image_path)
        with edit(filename) as spr:
            frame = frame0(spr, frame_index)
            colors = _palette(palette, [px.remove_background(src, remove_background)], spr.palette)
            result = _pixelize(src, colors, width, height, pixel_size, remove_background, 24,
                               _outline(outline_color), True)
            rgba, cx, cy = px.fit_to_canvas(result.rgba, spr.width, spr.height, align)
            if align == "none":
                cx, cy = x, y
            layer = find_layer(spr, layer_name)
            if layer is None:
                layer = ase.Layer(layer_name)
                spr.layers.append(layer)
            layer.cels[frame] = ase.Cel(rgba_to_image(spr, rgba), cx, cy)
    except (ToolError, OSError) as e:
        return f"Failed to import pixelized image: {e}"
    return (f"Imported {image_path} as {rgba.shape[1]}x{rgba.shape[0]} pixel art on '{layer_name}' "
            f"frame {frame_index} at ({cx},{cy}) in {filename}")


@mcp.tool()
async def extract_palette_from_image(image_path: str, max_colors: int = 16, filename: str = "",
                                     remove_background: str = "auto") -> str:
    """Extract a pixel-art palette (k-means, dark to light) from an image.

    Handy for matching concept art: extract its palette, then draw or
    pixelize with it. With filename, the palette is also set on that sprite.

    Args:
        image_path: Source image (e.g. concept art)
        max_colors: Palette size, 2-256 (default 16)
        filename: Optional sprite to set the palette on
        remove_background: "auto" (ignore a uniform background), "none" or a hex color
    """
    if not os.path.isfile(image_path):
        return f"Image {image_path} not found"
    if not 2 <= max_colors <= 256:
        return "max_colors must be between 2 and 256"
    try:
        rgba = px.remove_background(px.load_rgba(image_path), remove_background)
        colors = px.extract_palette([rgba], max_colors)
        if filename:
            with edit(filename) as spr:
                spr.palette = colors
    except (ToolError, OSError, ValueError) as e:
        return f"Failed to extract palette: {e}"
    applied = f" (set on {filename})" if filename else ""
    return json.dumps([to_hex(c, with_alpha=False) for c in colors]) + applied


def _collect_paths(image_paths, directory, pattern) -> list[str]:
    paths = list(image_paths or [])
    if directory:
        if not os.path.isdir(directory):
            raise ToolError(f"Directory {directory} not found")
        paths += sorted(glob.glob(os.path.join(directory, pattern or "*.png")))
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise ToolError(f"Images not found: {', '.join(missing[:5])}")
    if not paths:
        raise ToolError("No frames: pass image_paths, or a directory containing images")
    return paths


def build_animated_sprite(output_filename: str, sources: list[np.ndarray], *, width: int, height: int,
                          pixel_size: int, palette: Any, remove_background: str, align: str,
                          outline_color: str, durations: list[int], tags: list[dict],
                          layer_name: str, clean: bool = True) -> ase.Sprite:
    """Pixelize frames with one shared palette and save them as an animated sprite."""
    cut = [px.remove_background(s, remove_background) for s in sources]
    colors = _palette(palette, cut)
    outline = _outline(outline_color)
    # With pixel_size the frames keep their own size (width/height is only the canvas).
    target_w, target_h = (0, 0) if pixel_size else (width, height)
    frames = [_pixelize(s, colors, target_w, target_h, pixel_size, remove_background, 24,
                        outline, clean).rgba for s in sources]
    canvas_w = width or max(f.shape[1] for f in frames)
    canvas_h = height or max(f.shape[0] for f in frames)
    spr = ase.Sprite(canvas_w, canvas_h, ase.RGB,
                     palette=colors[:256] if colors else _used_palette(frames),
                     durations=list(durations))
    layer = ase.Layer(layer_name)
    for i, rgba in enumerate(frames):
        img, x, y = px.fit_to_canvas(rgba, canvas_w, canvas_h, align)
        if img[..., 3].any() or i == 0:
            layer.cels[i] = ase.Cel(img.copy(), x, y)
    spr.layers.append(layer)
    for t in tags:
        direction = {"forward": 0, "reverse": 1, "pingpong": 2}.get(str(t.get("direction", "forward")).lower(), 0)
        start, end = int(t["from"]) - 1, int(t["to"]) - 1
        if not 0 <= start <= end < len(frames):
            raise ToolError(f"Tag {t.get('name')!r} covers frames outside 1-{len(frames)}")
        spr.tags.append(ase.Tag(str(t["name"]), start, end, direction))
    save(spr, output_filename)
    return spr


def _used_palette(frames: list[np.ndarray]) -> list:
    colors = np.unique(np.concatenate([f[f[..., 3] > 0][:, :3] for f in frames]), axis=0)[:256]
    return [tuple(int(v) for v in c) + (255,) for c in colors] or [(0, 0, 0, 255)]


def _read_manifest(path: str) -> dict:
    """Frames, tags, per-frame durations, size and pivot from a sprite_render manifest."""
    if not os.path.isfile(path):
        raise ToolError(f"Manifest {path} not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "sequence" not in data:
        raise ToolError(f"{path} is not a sprite_render manifest (no 'sequence')")
    base = os.path.dirname(os.path.abspath(path))
    sequence = [p if os.path.isabs(p) else os.path.join(base, p) for p in data["sequence"]]
    durations = []
    for anim in data.get("animations", []):
        for _direction in data.get("directions", ["S"]):
            durations += [int(anim.get("duration_ms", 100))] * len(anim.get("frames", []))
    if len(durations) != len(sequence):
        durations = [int(data.get("duration_ms", 100))] * len(sequence)
    return {"sequence": sequence, "tags": data.get("tags", []), "durations": durations,
            "frame_size": int(data.get("frame_size", 0)), "pivot": data.get("pivot")}


@mcp.tool()
async def frames_to_sprite(
    output_filename: str,
    image_paths: List[str] | None = None,
    directory: str = "",
    pattern: str = "*.png",
    width: int = 0,
    height: int = 0,
    pixel_size: int = 0,
    palette: Any = "auto:16",
    remove_background: str = "auto",
    align: str = "none",
    duration_ms: int = 100,
    tags: List[Dict[str, Any]] | None = None,
    layer_name: str = "Layer 1",
    outline_color: str = "",
    manifest: str = "",
) -> list[McpImage | str]:
    """Build an animated LibreSprite sprite from a sequence of images, one frame each.

    For Blender turnarounds/animations, ComfyUI batches or any folder of
    frames. Every frame is pixelized with ONE shared palette (so colors
    don't flicker between frames) and the same scale. Shows you the frames.

    Sizing: width/height set the sprite canvas. With pixel_size, each frame
    is downscaled by that factor and placed on the canvas by `align`;
    without it, each image is scaled to exactly width x height (best for
    renders that are already framed consistently, e.g. Blender output).

    Args:
        output_filename: .ase file to create (overwritten if it exists)
        image_paths: Frame images in order
        directory: Alternatively, a folder of frames (sorted by name)
        pattern: Glob for directory (default "*.png")
        width: Canvas width (0 = from the frames)
        height: Canvas height (0 = from the frames)
        pixel_size: Source pixels per output pixel (alternative sizing, see above)
        palette: "auto:N" (default auto:16, shared by all frames), a preset,
            "none" or a list of hex colors
        remove_background: "auto", "none" or a hex color
        align: "none" (default), "bottom" (feet on the bottom edge) or "center";
            use bottom/center for frames whose framing varies
        duration_ms: Duration of every frame (default 100)
        tags: Optional tags, e.g. [{"name": "walk_S", "from": 1, "to": 8, "direction": "forward"}]
        layer_name: Name of the layer holding the frames
        outline_color: Optional hex color for a 1px outline around the subject
        manifest: A manifest.json from blender_render_sprites: supplies the
            frames, tags (Walk_S, Walk_SW, ...), durations and size, and
            stores the character's ground pivot as a "pivot" slice
    """
    if align not in ("none", "bottom", "center"):
        return ["align must be 'none', 'bottom' or 'center'"]
    if not 1 <= duration_ms <= 65535:
        return ["duration_ms must be between 1 and 65535"]
    err = reject_traversal(output_filename)
    if err:
        return [err]
    try:
        durations = None
        pivot = None
        if manifest:
            info = _read_manifest(manifest)
            image_paths = info["sequence"]
            tags = tags or info["tags"]
            durations = info["durations"]
            width = width or info["frame_size"]
            height = height or info["frame_size"]
            pivot = info["pivot"]
        paths = _collect_paths(image_paths, directory, pattern)
        sources = [px.load_rgba(p) for p in paths]
        if not pixel_size and not (width or height):
            raise ToolError("Give width/height (canvas size) or pixel_size")
        spr = build_animated_sprite(output_filename, sources, width=width, height=height,
                                    pixel_size=pixel_size, palette=palette,
                                    remove_background=remove_background, align=align,
                                    outline_color=outline_color,
                                    durations=durations or [duration_ms] * len(paths),
                                    tags=tags or [], layer_name=layer_name)
        if pivot is not None:
            await create_slice(output_filename, "pivot", 0, 0, spr.width, spr.height)
            await set_slice_pivot(output_filename, "pivot", round(pivot["x"]), round(pivot["y"]))
    except (ToolError, OSError, KeyError, ValueError) as e:
        return [f"Failed to build sprite: {e}"]
    strip = np.concatenate([render.composite(spr, f) for f in range(min(spr.frame_count, 16))], axis=1)
    scale = max(1, min(8, 1024 // max(strip.shape[1], 1)))
    return [McpImage(data=_png(strip, scale), format="png"),
            f"Created {output_filename}: {spr.width}x{spr.height}, {spr.frame_count} frames, "
            f"{len(spr.palette)}-color palette, {len(spr.tags)} tags"
            + (" (first 16 frames shown)" if spr.frame_count > 16 else "")]


@mcp.tool()
async def import_spritesheet_image(
    image_path: str,
    output_filename: str,
    frame_width: int,
    frame_height: int,
    columns: int = 0,
    rows: int = 0,
    frame_count: int = 0,
    margin: int = 0,
    spacing: int = 0,
    duration_ms: int = 100,
    skip_empty: bool = True,
    pixelize_to: int = 0,
    palette: Any = "none",
) -> str:
    """Slice a sprite-sheet image (a grid of frames) into an animated sprite.

    Works for existing sheets (asset packs, old projects) and for AI
    character sheets. Frames are read left to right, top to bottom.

    Args:
        image_path: The sheet image
        output_filename: .ase file to create
        frame_width: Width of one cell in the sheet
        frame_height: Height of one cell in the sheet
        columns: Cells per row (0 = as many as fit)
        rows: Number of rows (0 = as many as fit)
        frame_count: Stop after this many frames (0 = all cells)
        margin: Pixels around the whole grid
        spacing: Pixels between cells
        duration_ms: Duration of every frame (default 100)
        skip_empty: Skip fully transparent cells (default True)
        pixelize_to: Optional height to pixelize each cell to (0 = keep pixels as they are)
        palette: Palette when pixelizing: "none" (default), "auto:N", preset or hex list
    """
    if frame_width <= 0 or frame_height <= 0:
        return "frame_width and frame_height must be > 0"
    if not os.path.isfile(image_path):
        return f"Image {image_path} not found"
    try:
        sheet = px.load_rgba(image_path)
        h, w = sheet.shape[:2]
        cols = columns or max(1, (w - 2 * margin + spacing) // (frame_width + spacing))
        nrows = rows or max(1, (h - 2 * margin + spacing) // (frame_height + spacing))
        cells = []
        for r in range(nrows):
            for c in range(cols):
                x0 = margin + c * (frame_width + spacing)
                y0 = margin + r * (frame_height + spacing)
                cell = sheet[y0:y0 + frame_height, x0:x0 + frame_width]
                if cell.shape[:2] != (frame_height, frame_width):
                    continue
                if skip_empty and not cell[..., 3].any():
                    continue
                cells.append(cell)
        if frame_count:
            cells = cells[:frame_count]
        if not cells:
            raise ToolError("No frames found with that grid")
        if pixelize_to:
            spr = build_animated_sprite(output_filename, cells, width=0, height=pixelize_to, pixel_size=0,
                                        palette=palette,
                                        remove_background="none", align="none", outline_color="",
                                        durations=[duration_ms] * len(cells), tags=[], layer_name="Layer 1")
        else:
            spr = ase.Sprite(frame_width, frame_height, ase.RGB, palette=_used_palette(cells),
                             durations=[duration_ms] * len(cells))
            layer = ase.Layer("Layer 1")
            for i, cell in enumerate(cells):
                layer.cels[i] = ase.Cel(cell.copy())
            spr.layers.append(layer)
            save(spr, output_filename)
    except (ToolError, OSError) as e:
        return f"Failed to import sprite sheet: {e}"
    return f"Created {output_filename}: {spr.width}x{spr.height}, {spr.frame_count} frames from {image_path}"
