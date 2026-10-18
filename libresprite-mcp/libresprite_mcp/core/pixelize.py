"""Turn generated or rendered images into clean pixel art.

Images from ComfyUI, Krita AI Diffusion or a Blender render are not pixel
art: they have far too many colors, soft edges, an inconsistent "pixel"
grid and usually a background. pixelize() fixes that in order:

1. remove the background (flood fill from the border, so interior pixels
   that happen to share the background color survive),
2. snap colors to a palette (a preset, an existing sprite's palette, or
   one extracted with k-means; frames of one animation share a palette),
3. downscale by blocks, giving each output pixel the most common palette
   color of its block (no blended in-between colors),
4. binarize alpha, remove isolated stray pixels and optionally outline.

Everything works on (h, w, 4) uint8 RGBA arrays.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import raster
from .colors import parse_hex_color
from .palettes import PRESETS


def load_rgba(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"))


# ---------------------------------------------------------------- background


def border_color(rgba: np.ndarray, tolerance: int = 24) -> tuple[np.ndarray, float]:
    """Dominant border color (median) and the share of border pixels within tolerance of it."""
    edge = np.concatenate([rgba[0], rgba[-1], rgba[:, 0], rgba[:, -1]])[:, :3].astype(np.int32)
    median = np.median(edge, axis=0)
    close = np.abs(edge - median).max(axis=1) <= tolerance
    if not close.any():
        return median.round().astype(np.int32), 0.0
    return edge[close].mean(axis=0).round().astype(np.int32), float(close.mean())


def remove_background(rgba: np.ndarray, mode: str = "auto", tolerance: int = 24) -> np.ndarray:
    """Make the background transparent.

    mode: "auto" (detect a uniform border color; skipped if the image already
    has transparency or the border isn't uniform), "none", or a hex color.
    Only pixels connected to the border are removed.
    """
    out = rgba.copy()
    if mode == "none":
        return out
    if mode == "auto":
        if (rgba[..., 3] < 255).mean() > 0.05:
            return out  # already cut out
        color, share = border_color(rgba, tolerance)
        if share < 0.6:
            return out  # no uniform background to remove
    else:
        parsed = parse_hex_color(mode)
        if parsed is None:
            raise ValueError(f"remove_background must be 'auto', 'none' or a hex color, not {mode!r}")
        color = np.array(parsed[:3], np.int32)
    diff = np.abs(rgba[..., :3].astype(np.int32) - color).max(axis=-1)
    out[_flood_from_border((diff <= tolerance) | (rgba[..., 3] == 0))] = 0
    return out


def _flood_from_border(candidate: np.ndarray) -> np.ndarray:
    """Candidate pixels 4-connected to the image border (one multi-seed BFS)."""
    h, w = candidate.shape
    flat = candidate.ravel()
    region = np.zeros(h * w, bool)
    border = np.unique(np.concatenate([np.arange(w), np.arange(w) + (h - 1) * w,
                                       np.arange(h) * w, np.arange(h) * w + w - 1]))
    seeds = border[flat[border]]
    region[seeds] = True
    queue = deque(seeds.tolist())
    while queue:
        i = queue.popleft()
        y, x = divmod(i, w)
        for j, ok in ((i - 1, x > 0), (i + 1, x < w - 1), (i - w, y > 0), (i + w, y < h - 1)):
            if ok and flat[j] and not region[j]:
                region[j] = True
                queue.append(j)
    return region.reshape(h, w)


# ---------------------------------------------------------------- grid size


def detect_pixel_size(rgba: np.ndarray, max_size: int = 64, threshold: int = 24) -> tuple[int, int, int, float]:
    """Estimate the "fake pixel" size of upscaled pixel art.

    Returns (size, offset_x, offset_y, confidence). Color edges in upscaled
    pixel art fall on a regular grid. For each candidate size we measure how
    much of the edge energy lands on the best-aligned comb, above what a
    random comb would catch (1/size), on both axes; the largest size that
    still catches most edges wins (divisors of the true size catch them all
    too, multiples miss some). confidence is that excess share, 0-1;
    below ~0.6 there is no clear grid.
    """
    # Per-pixel color steps (largest channel change, alpha included). Steps
    # under the threshold are noise (AI jitter, JPEG) and don't count.
    c = rgba.astype(np.int16)
    c[..., :3] *= (c[..., 3:4] > 0)
    dx = np.abs(np.diff(c, axis=1)).max(axis=-1)
    dy = np.abs(np.diff(c, axis=0)).max(axis=-1)
    gx = np.where(dx > threshold, dx, 0).sum(axis=0).astype(np.float64)
    gy = np.where(dy > threshold, dy, 0).sum(axis=1).astype(np.float64)

    def capture(profile: np.ndarray, size: int) -> tuple[float, int]:
        total = profile.sum()
        if total <= 0:
            return 0.0, 0
        sums = [profile[off::size].sum() for off in range(size)]
        off = int(np.argmax(sums))
        share = sums[off] / total
        return (share - 1 / size) / (1 - 1 / size), (off + 1) % size

    limit = max(2, min(max_size, len(gx) // 3, len(gy) // 3))
    scores = []
    for size in range(2, limit + 1):
        cx, ox = capture(gx, size)
        cy, oy = capture(gy, size)
        scores.append((size, ox, oy, float(min(cx, cy))))
    top = max((s[3] for s in scores), default=0.0)
    # Largest size explaining (nearly) as much as the best one; a multiple
    # of the true size that misses even a few faint edges falls below it.
    best = (1, 0, 0, 0.0)
    for size, ox, oy, confidence in scores:
        if confidence >= max(0.6, 0.95 * top):
            best = (size, ox, oy, confidence)
    return best


# ---------------------------------------------------------------- palettes


def extract_palette(images: list[np.ndarray], max_colors: int = 16, samples: int = 60000,
                    seed: int = 1) -> list[tuple[int, int, int, int]]:
    """A palette of up to max_colors opaque colors shared by all images (k-means)."""
    pixels = np.concatenate([im[im[..., 3] >= 128][:, :3] for im in images]) if images else np.zeros((0, 3))
    if len(pixels) == 0:
        return [(0, 0, 0, 255)]
    rng = np.random.default_rng(seed)
    if len(pixels) > samples:
        pixels = pixels[rng.choice(len(pixels), samples, replace=False)]
    pixels = pixels.astype(np.float32)
    uniq = np.unique(pixels, axis=0)
    if len(uniq) <= max_colors:
        return [tuple(int(v) for v in c) + (255,) for c in uniq]
    # k-means++ initialisation, then Lloyd iterations.
    centers = [pixels[rng.integers(len(pixels))]]
    for _ in range(1, max_colors):
        d = np.min(((pixels[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(pixels[rng.choice(len(pixels), p=d / d.sum())] if d.sum() > 0 else pixels[0])
    centers = np.array(centers)
    for _ in range(25):
        labels = ((pixels[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
        moved = np.array([pixels[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                          for k in range(len(centers))])
        if np.allclose(moved, centers, atol=0.5):
            break
        centers = moved
    centers = np.unique(np.clip(centers.round(), 0, 255).astype(int), axis=0)
    # Dark to light, which is how palettes are usually laid out.
    order = np.argsort(centers @ np.array([299, 587, 114]))
    return [tuple(int(v) for v in centers[i]) + (255,) for i in order]


def resolve_palette(spec, images: list[np.ndarray], sprite_palette=None) -> list[tuple[int, int, int, int]] | None:
    """Turn a palette spec into colors.

    spec: "auto" / "auto:N" (k-means over the images), a preset name, "sprite"
    (sprite_palette), "none" (keep colors), or a list of hex colors.
    """
    if isinstance(spec, (list, tuple)):
        colors = [parse_hex_color(c) for c in spec]
        if not colors or any(c is None for c in colors):
            raise ValueError("palette list must contain hex colors")
        return [c[:3] + (255,) for c in colors]  # type: ignore[index]
    spec = (spec or "auto").strip().lower()
    if spec == "none":
        return None
    if spec == "sprite":
        if not sprite_palette:
            raise ValueError("palette 'sprite' needs a target sprite")
        return [tuple(c[:3]) + (255,) for c in sprite_palette if c[3] > 0]
    if spec.startswith("auto"):
        n = int(spec.split(":", 1)[1]) if ":" in spec else 16
        if not 2 <= n <= 256:
            raise ValueError("auto palette size must be between 2 and 256")
        return extract_palette(images, n)
    if spec in PRESETS:
        return list(PRESETS[spec])
    raise ValueError(f"Unknown palette '{spec}': use auto, auto:N, sprite, none, a preset "
                     f"({', '.join(PRESETS)}) or a list of hex colors")


def nearest_indices(rgb: np.ndarray, palette: list) -> np.ndarray:
    """Palette index of every pixel (weighted RGB distance, cached per unique color)."""
    pal = np.array([c[:3] for c in palette], np.int32)
    flat = rgb.reshape(-1, 3).astype(np.int32)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    weights = np.array([2, 4, 3], np.int32)
    dist = (((uniq[:, None, :] - pal[None]) ** 2) * weights).sum(-1)
    return dist.argmin(axis=1)[inverse.reshape(-1)].reshape(rgb.shape[:-1])


# ---------------------------------------------------------------- downscale


def block_mode(indices: np.ndarray, alpha_on: np.ndarray, block: int, ncolors: int,
               coverage: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Downscale an index image by `block`, taking each block's most common color.

    A block is opaque when at least `coverage` of it is opaque.
    """
    h, w = indices.shape[0] // block, indices.shape[1] // block
    idx = indices[:h * block, :w * block].reshape(h, block, w, block).transpose(0, 2, 1, 3).reshape(h, w, -1)
    on = alpha_on[:h * block, :w * block].reshape(h, block, w, block).transpose(0, 2, 1, 3).reshape(h, w, -1)
    counts = np.stack([((idx == k) & on).sum(-1) for k in range(ncolors)], axis=-1)
    return counts.argmax(-1), on.mean(-1) >= coverage


def despeckle(indices: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Replace opaque pixels that match none of their 4 neighbours with the most common neighbour color."""
    out = indices.copy()
    h, w = indices.shape
    pad_i = np.pad(indices, 1, constant_values=-1)
    pad_o = np.pad(opaque, 1, constant_values=False)
    neigh = [(pad_i[1 + dy:1 + dy + h, 1 + dx:1 + dx + w], pad_o[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
             for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    same = sum(((n == indices) & o).astype(int) for n, o in neigh)
    opaque_neigh = sum(o.astype(int) for _, o in neigh)
    lonely = opaque & (same == 0) & (opaque_neigh >= 3)
    for y, x in zip(*np.nonzero(lonely)):
        values = [int(n[y, x]) for n, o in neigh if o[y, x]]
        out[y, x] = max(set(values), key=values.count)
    return out


@dataclass
class PixelizeResult:
    rgba: np.ndarray
    palette: list
    block: int
    detected: bool


def pixelize(rgba: np.ndarray, *, width: int = 0, height: int = 0, pixel_size: int = 0,
             palette: list | None = None, background: str = "auto", tolerance: int = 24,
             alpha_threshold: int = 128, coverage: float = 0.5, clean: bool = True,
             outline: tuple | None = None) -> PixelizeResult:
    """Convert an RGBA image to pixel art (see the module docstring).

    Size: pixel_size (block size in source pixels), width and/or height of
    the result, or neither to detect the grid of upscaled pixel art.
    palette None keeps the original colors (per-block mean instead of mode).
    """
    src = remove_background(rgba, background, tolerance)
    detected = False
    ox = oy = 0
    if pixel_size <= 0 and width <= 0 and height <= 0:
        pixel_size, ox, oy, confidence = detect_pixel_size(src)
        detected = confidence >= 0.6 and pixel_size > 1
        if not detected:
            raise ValueError("Could not detect a pixel grid; pass width, height or pixel_size")
    if pixel_size > 0:
        block = pixel_size
        src = src[oy:, ox:]
        out_w, out_h = src.shape[1] // block, src.shape[0] // block
    else:
        h, w = src.shape[:2]
        if width <= 0:
            width = max(1, round(w * height / h))
        if height <= 0:
            height = max(1, round(h * width / w))
        out_w, out_h = width, height
        block = max(1, min(w // out_w, h // out_h, 16))
        if (w, h) != (out_w * block, out_h * block):
            # Resample to an exact multiple first; alpha keeps hard edges.
            im = Image.fromarray(src, "RGBA")
            rgb = im.convert("RGB").resize((out_w * block, out_h * block), Image.Resampling.LANCZOS)
            a = im.getchannel("A").resize((out_w * block, out_h * block), Image.Resampling.BILINEAR)
            src = np.dstack([np.array(rgb), np.array(a)])
    if out_w < 1 or out_h < 1:
        raise ValueError("The target size is smaller than one pixel")
    opaque_src = src[..., 3] >= alpha_threshold

    if palette is None:
        # Keep the original colors: mean of each block's opaque pixels.
        blocks = src[:out_h * block, :out_w * block].reshape(out_h, block, out_w, block, 4).astype(np.float64)
        on_px = opaque_src[:out_h * block, :out_w * block].reshape(out_h, block, out_w, block)
        count = on_px.sum(axis=(1, 3))
        rgb = (blocks[..., :3] * on_px[..., None]).sum(axis=(1, 3)) / np.maximum(count, 1)[..., None]
        on = count >= coverage * block * block
        out = np.zeros((out_h, out_w, 4), np.uint8)
        out[..., :3] = np.clip(rgb.round(), 0, 255)
        out[..., 3] = np.where(on, 255, 0)
        out[~on] = 0
        if outline is not None:
            out[raster.outline_mask(on)] = outline
        return PixelizeResult(out, [], block, detected)

    indices = nearest_indices(src[..., :3], palette)
    small_idx, on = block_mode(indices, opaque_src, block, len(palette), coverage)
    if clean:
        small_idx = despeckle(small_idx, on)
    pal = np.array([c[:3] for c in palette], np.uint8)
    out = np.zeros((out_h, out_w, 4), np.uint8)
    out[..., :3] = pal[small_idx]
    out[..., 3] = np.where(on, 255, 0)
    out[~on] = 0
    if outline is not None:
        mask = raster.outline_mask(on)
        out[mask] = outline
    return PixelizeResult(out, palette, block, detected)


def fit_to_canvas(rgba: np.ndarray, width: int, height: int, align: str = "bottom") -> tuple[np.ndarray, int, int]:
    """Crop to content and return (image, x, y) placing it on a width x height canvas.

    align: "bottom" (feet on the bottom edge, centered), "center" or "none"
    (keep the image as is at 0,0).
    """
    if align == "none":
        return rgba, 0, 0
    ys, xs = np.nonzero(rgba[..., 3])
    if len(xs) == 0:
        return rgba[:1, :1], 0, 0
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape[:2]
    x = (width - cw) // 2
    y = height - ch if align == "bottom" else (height - ch) // 2
    return crop, x, y
