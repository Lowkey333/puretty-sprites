"""Shared colour parsing and conversion for tool modules.

Accepts #RGB, #RGBA, #RRGGBB, #RRGGBBAA (with or without the leading '#')
and returns (r, g, b, a). Alpha defaults to 255 when not supplied.
"""
from __future__ import annotations

import numpy as np

from . import ase


def parse_hex_color(value: str) -> tuple[int, int, int, int] | None:
    """Parse a hex colour string into an (r, g, b, a) tuple, or None."""
    if not value:
        return None
    h = value.strip().lstrip("#")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    if len(h) not in (6, 8):
        return None
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        a = int(h[6:8], 16) if len(h) == 8 else 255
    except ValueError:
        return None
    return r, g, b, a


def to_hex(rgba, with_alpha: bool = True) -> str:
    r, g, b, a = (int(v) for v in rgba)
    return f"#{r:02X}{g:02X}{b:02X}{a:02X}" if with_alpha else f"#{r:02X}{g:02X}{b:02X}"


def luma(r: int, g: int, b: int) -> int:
    """LibreSprite's rgb_luma(): Rec. 709 weights in integer math."""
    return (r * 2126 + g * 7152 + b * 722) // 10000


def nearest_index(palette, rgba, transparent_index: int | None = None) -> int:
    """Index of the palette colour closest to rgba (squared RGBA distance).

    Fully transparent colours map to transparent_index when given.
    """
    if rgba[3] == 0 and transparent_index is not None:
        return transparent_index
    pal = np.asarray(palette, np.int32)
    diff = pal - np.asarray(rgba, np.int32)
    weights = np.array([2, 4, 3, 1], np.int32)  # weighted RGB distance (green matters most)
    dist = (diff * diff * weights).sum(axis=1)
    if transparent_index is not None and 0 <= transparent_index < len(pal) and rgba[3] > 0:
        dist[transparent_index] = np.iinfo(np.int32).max
    return int(np.argmin(dist))


def to_pixel(spr: ase.Sprite, rgba) -> tuple[int, ...]:
    """Convert an (r, g, b, a) colour to a pixel value in the sprite's mode."""
    r, g, b, a = (int(v) for v in rgba)
    if spr.color_mode == ase.RGB:
        return (r, g, b, a)
    if spr.color_mode == ase.GRAY:
        return (luma(r, g, b), a)
    return (nearest_index(spr.palette, (r, g, b, a), spr.transparent_index),)


def transparent_pixel(spr: ase.Sprite) -> tuple[int, ...]:
    if spr.color_mode == ase.RGB:
        return (0, 0, 0, 0)
    if spr.color_mode == ase.GRAY:
        return (0, 0)
    return (spr.transparent_index,)


def pixel_to_rgba(spr: ase.Sprite, pixel) -> tuple[int, int, int, int]:
    """Convert a stored pixel value back to (r, g, b, a)."""
    p = [int(v) for v in np.atleast_1d(pixel)]
    if spr.color_mode == ase.RGB:
        return tuple(p)  # type: ignore[return-value]
    if spr.color_mode == ase.GRAY:
        return (p[0], p[0], p[0], p[1])
    index = p[0]
    if index == spr.transparent_index or index >= len(spr.palette):
        return (0, 0, 0, 0)
    return tuple(spr.palette[index])  # type: ignore[return-value]


def image_to_rgba(spr: ase.Sprite, image: np.ndarray) -> np.ndarray:
    """Convert a cel image of the sprite's mode to (h, w, 4) uint8 RGBA."""
    if spr.color_mode == ase.RGB:
        return image.copy()
    if spr.color_mode == ase.GRAY:
        v = image[..., 0:1]
        return np.concatenate([v, v, v, image[..., 1:2]], axis=-1)
    pal = np.zeros((256, 4), np.uint8)
    pal[:len(spr.palette)] = spr.palette
    out = pal[image[..., 0]]
    out[image[..., 0] == spr.transparent_index] = 0
    return out


def rgba_to_image(spr: ase.Sprite, rgba: np.ndarray) -> np.ndarray:
    """Convert (h, w, 4) RGBA to a cel image in the sprite's mode."""
    rgba = rgba.astype(np.uint8)
    if spr.color_mode == ase.RGB:
        return rgba.copy()
    if spr.color_mode == ase.GRAY:
        r, g, b = (rgba[..., i].astype(np.int32) for i in range(3))
        v = ((r * 2126 + g * 7152 + b * 722) // 10000).astype(np.uint8)
        return np.stack([v, rgba[..., 3]], axis=-1)
    return quantize_rgba(rgba, spr.palette, spr.transparent_index)[..., None]


def quantize_rgba(rgba: np.ndarray, palette, transparent_index: int | None) -> np.ndarray:
    """Map every pixel to its nearest palette index, caching unique colours."""
    flat = rgba.reshape(-1, 4)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    lut = np.array([nearest_index(palette, tuple(c), transparent_index) for c in uniq], np.uint8)
    return lut[inverse.reshape(-1)].reshape(rgba.shape[:2])


# RGB <-> HSL. h in [0,360), s and l in [0,1].

def rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    r, g, b = r / 255, g / 255, b / 255
    maxc, minc = max(r, g, b), min(r, g, b)
    l = (maxc + minc) / 2
    if maxc == minc:
        return 0.0, 0.0, l
    d = maxc - minc
    s = d / (2 - maxc - minc) if l > 0.5 else d / (maxc + minc)
    if maxc == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif maxc == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h * 60, s, l


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    h = h % 360
    if s <= 0:
        v = int(l * 255 + 0.5)
        return v, v, v
    c = (1 - abs(2 * l - 1)) * s
    hp = h / 60
    x = c * (1 - abs(hp % 2 - 1))
    r1, g1, b1 = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][min(int(hp), 5)]
    m = l - c / 2
    return tuple(max(0, min(255, int((v + m) * 255 + 0.5))) for v in (r1, g1, b1))  # type: ignore[return-value]


def rgb_to_hsl_array(rgb: np.ndarray):
    """Vectorised rgb_to_hsl over an (..., 3) array; returns h, s, l arrays."""
    c = rgb.astype(np.float64) / 255
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    maxc, minc = c.max(axis=-1), c.min(axis=-1)
    l = (maxc + minc) / 2
    d = maxc - minc
    gray = d == 0
    safe_d = np.where(gray, 1, d)
    s = np.where(l > 0.5, d / np.where(2 - maxc - minc == 0, 1, 2 - maxc - minc),
                 d / np.where(maxc + minc == 0, 1, maxc + minc))
    h = np.where(maxc == r, (g - b) / safe_d + np.where(g < b, 6, 0),
                 np.where(maxc == g, (b - r) / safe_d + 2, (r - g) / safe_d + 4))
    return np.where(gray, 0, h * 60), np.where(gray, 0, s), l


def hsl_to_rgb_array(h, s, l) -> np.ndarray:
    h = np.mod(h, 360)
    c = (1 - np.abs(2 * l - 1)) * s
    hp = h / 60
    x = c * (1 - np.abs(np.mod(hp, 2) - 1))
    zero = np.zeros_like(c)
    sector = np.minimum(hp.astype(int), 5)
    r1 = np.choose(sector, [c, x, zero, zero, x, c])
    g1 = np.choose(sector, [x, c, c, x, zero, zero])
    b1 = np.choose(sector, [zero, zero, x, c, c, x])
    m = l - c / 2
    out = np.stack([r1 + m, g1 + m, b1 + m], axis=-1)
    return np.clip(np.floor(out * 255 + 0.5), 0, 255).astype(np.uint8)
