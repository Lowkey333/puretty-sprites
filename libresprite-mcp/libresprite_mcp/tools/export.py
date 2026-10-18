import io
import json
import math
import os
import shutil

import numpy as np
from mcp.server.mcpserver import Image as McpImage
from PIL import Image, ImageDraw

from .. import __version__, mcp
from ..core import ase, render
from ..core.colors import parse_hex_color, rgba_to_image
from ..core.sprite import (ToolError, edit, find_layer, frame0, frame_range, get_layer, load,
                           reject_traversal)
from .slices import aseprite_meta

_SHEET_TYPES = ("horizontal", "vertical", "rows", "columns", "packed")


def _frame_image(spr: ase.Sprite, frame: int, scale: int = 1, only_layers=None,
                 include_hidden: bool = False) -> Image.Image:
    rgba = render.composite(spr, frame, include_hidden=include_hidden, only_layers=only_layers)
    return render.to_pil(render.scale_nearest(rgba, scale))


def _check_output(path: str) -> None:
    err = reject_traversal(path)
    if err:
        raise ToolError(err)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)


def _save_gif(frames: list[Image.Image], durations: list[int], path: str, pingpong: bool = False) -> None:
    """Animated GIF; pixels with alpha < 128 become the transparent color."""
    if pingpong and len(frames) > 2:
        frames = frames + frames[-2:0:-1]
        durations = durations + durations[-2:0:-1]
    paletted = []
    for im in frames:
        alpha = np.array(im.getchannel("A"))
        rgb = im.convert("RGB").quantize(colors=255, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        idx = np.array(rgb)
        idx[alpha < 128] = 255
        p = Image.fromarray(idx.astype(np.uint8), "P")
        pal = rgb.getpalette()[:255 * 3]
        p.putpalette(pal + [0] * (768 - len(pal)))
        paletted.append(p)
    paletted[0].save(path, save_all=True, append_images=paletted[1:], duration=durations,
                     loop=0, disposal=2, transparency=255, optimize=False)


def _save_flat(im: Image.Image, path: str, fmt: str) -> None:
    if fmt in ("jpg", "jpeg", "bmp"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    im.save(path, format={"jpg": "JPEG", "tif": "TIFF"}.get(fmt, fmt.upper()))


def _numbered(path: str, n: int, width: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_{n:0{width}d}{ext}"


@mcp.tool()
async def export_sprite(filename: str, output_filename: str, format: str = "png") -> str:
    """Export the sprite to another format.

    .gif writes an animation with the frame durations; png/jpg/bmp/webp
    write one image per frame (name_1.png, name_2.png, ... when there is
    more than one frame); .ase/.aseprite saves a copy of the sprite.

    Args:
        filename: Sprite file to export
        output_filename: Name of the output file
        format: Output format used when output_filename has no extension
            (default "png")
    """
    try:
        spr = load(filename)
        _check_output(output_filename)
        ext = os.path.splitext(output_filename)[1].lower().lstrip(".") or format.lower()
        if not os.path.splitext(output_filename)[1]:
            output_filename += "." + ext
        if ext in ("ase", "aseprite"):
            ase.write(spr, output_filename)
            return f"Exported {filename} to {output_filename}"
        if ext == "gif":
            _save_gif([_frame_image(spr, f) for f in range(spr.frame_count)], spr.durations, output_filename)
            return f"Exported {spr.frame_count}-frame GIF to {output_filename}"
        if ext not in ("png", "jpg", "jpeg", "bmp", "webp", "tga", "tif", "tiff"):
            raise ToolError(f"Unsupported format '{ext}'")
        if spr.frame_count == 1:
            _save_flat(_frame_image(spr, 0), output_filename, ext)
            return f"Exported {filename} to {output_filename}"
        digits = len(str(spr.frame_count))
        for f in range(spr.frame_count):
            _save_flat(_frame_image(spr, f), _numbered(output_filename, f + 1, digits), ext)
    except (ToolError, OSError, ValueError) as e:
        return f"Failed to export sprite: {e}"
    return (f"Exported {spr.frame_count} frames to {_numbered(output_filename, 1, digits)} ... "
            f"{_numbered(output_filename, spr.frame_count, digits)}")


@mcp.tool()
async def copy_sprite(filename: str, output_filename: str, overwrite: bool = False) -> str:
    """Copy a sprite to a new file.

    Args:
        filename: Sprite file to copy
        output_filename: Path of the copy (.ase or .aseprite)
        overwrite: Whether to overwrite if output exists
    """
    if not os.path.isfile(filename):
        return f"File {filename} not found"
    err = reject_traversal(output_filename)
    if err:
        return err
    if os.path.exists(output_filename) and not overwrite:
        return f"Failed to copy sprite: {output_filename} exists (pass overwrite=True)"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        shutil.copyfile(filename, output_filename)
    except OSError as e:
        return f"Failed to copy sprite: {e}"
    return f"Sprite copied to {output_filename}"


@mcp.tool()
async def export_frame(filename: str, frame_index: int, output_filename: str, scale: int = 1) -> str:
    """Export a single frame as a PNG, optionally scaled up.

    For checking your work, view_frame shows the frame to you directly;
    export_frame is for saving a PNG somewhere.

    Args:
        filename: Sprite file to export
        frame_index: Frame index starting at 1
        output_filename: Output PNG path
        scale: Integer nearest-neighbor scale factor (default 1)
    """
    if scale < 1:
        return "Scale must be >= 1"
    try:
        spr = load(filename)
        frame = frame0(spr, frame_index)
        _check_output(output_filename)
        _frame_image(spr, frame, scale).save(output_filename)
    except (ToolError, OSError) as e:
        return f"Failed to export frame: {e}"
    return f"Frame {frame_index} exported to {output_filename} at {scale}x"


def _sheet_layout(n: int, w: int, h: int, sheet_type: str, padding: int) -> tuple[list[tuple[int, int]], int, int]:
    if sheet_type == "horizontal":
        cols, rows = n, 1
    elif sheet_type == "vertical":
        cols, rows = 1, n
    elif sheet_type == "columns":
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
    else:  # rows, packed (every frame has the same size, so packing is a grid)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    positions = []
    for i in range(n):
        c, r = (i // rows, i % rows) if sheet_type == "columns" else (i % cols, i // cols)
        positions.append((c * (w + padding), r * (h + padding)))
    return positions, cols * (w + padding) - padding, rows * (h + padding) - padding


def build_sheet(spr: ase.Sprite, frames: list[int], scale: int, sheet_type: str, padding: int,
                sheet_name: str, title: str, data_format: str, list_tags: bool):
    """Render frames into one sheet image plus LibreSprite-style JSON data."""
    w, h = spr.width * scale, spr.height * scale
    positions, sheet_w, sheet_h = _sheet_layout(len(frames), w, h, sheet_type, padding)
    sheet = Image.new("RGBA", (max(1, sheet_w), max(1, sheet_h)), (0, 0, 0, 0))
    entries = []
    for frame, (x, y) in zip(frames, positions):
        sheet.paste(_frame_image(spr, frame, scale), (x, y))
        entries.append((f"{title} {frame}.ase", {
            "frame": {"x": x, "y": y, "w": w, "h": h},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h},
            "sourceSize": {"w": w, "h": h},
            "duration": spr.durations[frame],
        }))
    if data_format == "json-hash":
        frames_data = {name: entry for name, entry in entries}
    else:
        frames_data = [{"filename": name, **entry} for name, entry in entries]
    meta = {
        "app": "libresprite-mcp (LibreSprite-compatible)",
        "version": __version__,
        "image": sheet_name,
        "format": "RGBA8888",
        "size": {"w": sheet.width, "h": sheet.height},
        "scale": str(scale),
    }
    if list_tags:
        index = {f: i for i, f in enumerate(frames)}
        meta["frameTags"] = [
            {"name": t.name, "from": index[t.from_frame], "to": index[t.to_frame],
             "direction": ase.DIRECTIONS[t.direction]}
            for t in spr.tags if t.from_frame in index and t.to_frame in index
        ]
    meta["layers"] = [
        {"name": l.name, "opacity": l.opacity, "blendMode": ase.BLEND_MODES[l.blend_mode]}
        for l in spr.layers if l.visible
    ]
    return sheet, {"frames": frames_data, "meta": meta}


@mcp.tool()
async def export_spritesheet(
    filename: str,
    output_filename: str,
    sheet_type: str = "horizontal",
    data_filename: str = "",
    scale: int = 1,
    padding: int = 0,
    tag_name: str = "",
    data_format: str = "json-array",
    list_tags: bool = False,
) -> str:
    """Export frames as a sprite sheet, optionally with a JSON data file.

    The JSON matches LibreSprite's own --data output (frames with x/y/w/h
    and duration, meta.frameTags, meta.layers), so engine importers made
    for LibreSprite or Aseprite sheets read it.

    Args:
        filename: Sprite file to export
        output_filename: Output sheet image path (PNG)
        sheet_type: Layout: "horizontal", "vertical", "rows", "columns", or "packed"
        data_filename: Optional path for a JSON metadata file
        scale: Integer scale factor applied before packing (default 1)
        padding: Padding in pixels between frames (default 0)
        tag_name: Only include frames of this animation tag (default: all frames)
        data_format: JSON format for the data file: "json-array" (default) or "json-hash"
        list_tags: Include animation tag metadata in the JSON data file (default: False)
    """
    if sheet_type not in _SHEET_TYPES:
        return f"Unknown sheet_type '{sheet_type}': use {', '.join(_SHEET_TYPES)}"
    if data_format not in ("json-array", "json-hash"):
        return "data_format must be 'json-array' or 'json-hash'"
    if scale < 1 or padding < 0:
        return "scale must be >= 1 and padding >= 0"
    try:
        spr = load(filename)
        frames = list(range(spr.frame_count))
        if tag_name:
            tag = next((t for t in spr.tags if t.name == tag_name), None)
            if tag is None:
                raise ToolError(f"Tag not found: {tag_name!r}")
            frames = list(range(tag.from_frame, tag.to_frame + 1))
        _check_output(output_filename)
        title = os.path.splitext(os.path.basename(filename))[0]
        sheet, data = build_sheet(spr, frames, scale, sheet_type, padding,
                                  os.path.basename(output_filename), title, data_format,
                                  list_tags or bool(tag_name))
        sheet_slices = aseprite_meta(filename)
        if sheet_slices:
            data["meta"]["slices"] = sheet_slices
        sheet.save(output_filename)
        if data_filename:
            _check_output(data_filename)
            with open(data_filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
    except (ToolError, OSError) as e:
        return f"Failed to export spritesheet: {e}"
    data_note = f" with data {data_filename}" if data_filename else ""
    return (f"Exported {len(frames)}-frame {sheet_type} sheet ({sheet.width}x{sheet.height}) "
            f"to {output_filename}{data_note}")


@mcp.tool()
async def export_layers(filename: str, output_directory: str, include_hidden: bool = False, frame_index: int = 1) -> str:
    """Export each layer as its own PNG file named <layer>.png.

    Args:
        filename: Sprite file to export
        output_directory: Directory for the per-layer PNGs (created if missing)
        include_hidden: Also export hidden layers (default False)
        frame_index: Frame to export (default 1)
    """
    try:
        spr = load(filename)
        frame = frame0(spr, frame_index)
        _check_output(os.path.join(output_directory, "x.png"))
        written = []
        for layer in spr.layers:
            if not layer.visible and not include_hidden:
                continue
            safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in layer.name).strip() or "layer"
            path = os.path.join(output_directory, f"{safe}.png")
            _frame_image(spr, frame, only_layers={layer.name}, include_hidden=True).save(path)
            written.append(os.path.basename(path))
    except (ToolError, OSError) as e:
        return f"Failed to export layers: {e}"
    return f"Exported {len(written)} layers to {output_directory}: {', '.join(written)}"


@mcp.tool()
async def export_tag(filename: str, tag_name: str, output_filename: str, scale: int = 1) -> str:
    """Export the frames of an animation tag as a GIF or PNG sequence.

    Args:
        filename: Sprite file to export
        tag_name: Animation tag to export
        output_filename: Output path; .gif gives an animation (honoring the
            tag's direction), .png a sequence name_1.png, name_2.png, ...
        scale: Integer scale factor (default 1)
    """
    if scale < 1:
        return "Scale must be >= 1"
    try:
        spr = load(filename)
        tag = next((t for t in spr.tags if t.name == tag_name), None)
        if tag is None:
            raise ToolError(f"Tag not found: {tag_name!r} (tags: {', '.join(t.name for t in spr.tags) or 'none'})")
        frames = list(range(tag.from_frame, tag.to_frame + 1))
        if tag.direction == 1:
            frames.reverse()
        _check_output(output_filename)
        images = [_frame_image(spr, f, scale) for f in frames]
        if output_filename.lower().endswith(".gif"):
            _save_gif(images, [spr.durations[f] for f in frames], output_filename, pingpong=tag.direction == 2)
            return f"Exported tag '{tag_name}' ({len(frames)} frames) to {output_filename}"
        digits = len(str(len(frames)))
        for i, im in enumerate(images, 1):
            im.save(_numbered(output_filename, i, digits))
    except (ToolError, OSError) as e:
        return f"Failed to export tag: {e}"
    return f"Exported tag '{tag_name}' as {len(frames)} PNGs ({_numbered(output_filename, 1, digits)} ...)"


@mcp.tool()
async def import_image_as_layer(
    filename: str,
    image_path: str,
    layer_name: str,
    frame_index: int = 1,
    x: int = 0,
    y: int = 0,
) -> str:
    """Import an image file (PNG, etc.) into a layer of the sprite.

    Useful for bringing in reference images or composing pre-made parts.
    The layer is created (on top) if it does not exist, and the image
    replaces that layer's cel in the frame, with its top-left at (x, y).
    Colors are converted to the sprite's mode (nearest palette entry when
    indexed). To turn a large AI-generated image into pixel art first,
    use pixelize_image or import_pixelized_image.

    Args:
        filename: Sprite file to modify
        image_path: Image file to import
        layer_name: Layer to place the image on
        frame_index: Frame index starting at 1 (default 1)
        x: X position for the image's top-left corner (default 0)
        y: Y position for the image's top-left corner (default 0)
    """
    if not os.path.isfile(image_path):
        return f"Image {image_path} not found"
    try:
        with Image.open(image_path) as im:
            rgba = np.array(im.convert("RGBA"))
    except OSError as e:
        return f"Failed to import image: {e}"
    try:
        with edit(filename) as spr:
            frame = frame0(spr, frame_index)
            layer = find_layer(spr, layer_name)
            if layer is None:
                layer = ase.Layer(layer_name)
                spr.layers.append(layer)
            layer.cels[frame] = ase.Cel(rgba_to_image(spr, rgba), x, y)
    except ToolError as e:
        return f"Failed to import image: {e}"
    return (f"Imported {image_path} ({rgba.shape[1]}x{rgba.shape[0]}) into '{layer_name}' "
            f"frame {frame_index} at ({x},{y}) in {filename}")


def _auto_scale(w: int, h: int, target: int = 512) -> int:
    return max(1, min(32, target // max(w, h, 1)))


def _background(rgba: np.ndarray, background: str, cell: int) -> np.ndarray:
    if background == "transparent":
        return rgba
    if background == "checker":
        return render.over_checkerboard(rgba, cell)
    color = parse_hex_color(background)
    if color is None:
        raise ToolError(f"background must be 'checker', 'transparent' or a hex color, not {background!r}")
    base = np.zeros(rgba.shape, np.int32)
    base[...] = color
    return render.blend_normal(base, rgba.astype(np.int32), 255).astype(np.uint8)


def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@mcp.tool()
async def view_frame(
    filename: str,
    frame_index: int = 1,
    scale: int = 0,
    background: str = "checker",
    layer_name: str = "",
    grid: bool = False,
) -> list[McpImage | str]:
    """Show a frame to you as an image, so you can check your work.

    The core feedback loop: draw, view_frame, fix, repeat. Renders exactly
    what LibreSprite shows (layer opacity, blend modes, hidden layers off).

    Args:
        filename: Sprite file to view
        frame_index: Frame index starting at 1 (default 1)
        scale: Integer zoom; 0 (default) picks one that makes the image ~512px
        background: "checker" (default), "transparent" or a hex color
        layer_name: Show only this layer (default: all visible layers)
        grid: Draw a 1-pixel grid (needs scale >= 4) to help count pixels
    """
    try:
        spr = load(filename)
        frame = frame0(spr, frame_index)
        only = None
        if layer_name:
            get_layer(spr, layer_name)
            only = {layer_name}
        zoom = scale if scale > 0 else _auto_scale(spr.width, spr.height)
        rgba = render.composite(spr, frame, include_hidden=bool(layer_name), only_layers=only)
        img = _background(render.scale_nearest(rgba, zoom), background, max(zoom * 4, 8))
        pil = render.to_pil(img)
        if grid and zoom >= 4:
            draw = ImageDraw.Draw(pil)
            for gx in range(0, pil.width, zoom):
                draw.line([(gx, 0), (gx, pil.height)], fill=(0, 0, 0, 60))
            for gy in range(0, pil.height, zoom):
                draw.line([(0, gy), (pil.width, gy)], fill=(0, 0, 0, 60))
    except ToolError as e:
        return [f"Failed to view frame: {e}"]
    caption = (f"{os.path.basename(filename)} frame {frame_index}/{spr.frame_count}, "
               f"{spr.width}x{spr.height} shown at {zoom}x")
    return [McpImage(data=_png_bytes(pil), format="png"), caption]


@mcp.tool()
async def view_animation(
    filename: str,
    tag_name: str = "",
    start_frame: int = 1,
    end_frame: int = 0,
    scale: int = 0,
    background: str = "checker",
) -> list[McpImage | str]:
    """Show several frames side by side (a numbered contact sheet) as one image.

    Use it to check an animation's poses and spacing at a glance.

    Args:
        filename: Sprite file to view
        tag_name: Show this tag's frames (overrides start/end)
        start_frame: First frame (1-based, default 1)
        end_frame: Last frame (default 0 = last frame)
        scale: Integer zoom; 0 (default) fits the sheet in about 1024px
        background: "checker" (default), "transparent" or a hex color
    """
    try:
        spr = load(filename)
        if tag_name:
            tag = next((t for t in spr.tags if t.name == tag_name), None)
            if tag is None:
                raise ToolError(f"Tag not found: {tag_name!r}")
            frames = list(range(tag.from_frame, tag.to_frame + 1))
        else:
            frames = list(frame_range(spr, start_frame, end_frame))
        cols = min(len(frames), 8)
        rows = math.ceil(len(frames) / cols)
        zoom = scale if scale > 0 else max(1, min(16, 1024 // max(cols * spr.width, rows * spr.height, 1)))
        w, h = spr.width * zoom, spr.height * zoom
        label_h = 14
        sheet = Image.new("RGBA", (cols * (w + 4) + 4, rows * (h + label_h + 4) + 4), (40, 40, 48, 255))
        draw = ImageDraw.Draw(sheet)
        for i, frame in enumerate(frames):
            cx, cy = 4 + (i % cols) * (w + 4), 4 + (i // cols) * (h + label_h + 4)
            rgba = render.scale_nearest(render.composite(spr, frame), zoom)
            sheet.paste(render.to_pil(_background(rgba, background, max(zoom * 4, 8))), (cx, cy + label_h))
            draw.text((cx + 2, cy + 1), f"{frame + 1} ({spr.durations[frame]}ms)", fill=(230, 230, 230, 255))
    except ToolError as e:
        return [f"Failed to view animation: {e}"]
    caption = (f"{os.path.basename(filename)} frames {frames[0] + 1}-{frames[-1] + 1} "
               f"({len(frames)} frames) at {zoom}x")
    return [McpImage(data=_png_bytes(sheet), format="png"), caption]
