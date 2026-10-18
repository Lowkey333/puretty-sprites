"""Flatten sprite frames into RGBA images the way LibreSprite renders them.

The integer math follows LibreSprite v1.3's src/doc/blend_funcs.cpp and
src/render/render.cpp, so previews and exports match what LibreSprite
shows and exports pixel for pixel: a cel is drawn with opacity
MUL_UN8(cel.opacity, layer.opacity) through its layer's blend mode, and
indexed sprites skip the transparent color index.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import ase


def _mul_un8(a, b):
    t = a * b + 128
    return ((t >> 8) + t) >> 8


mul_un8 = _mul_un8  # opacity * opacity / 255, rounded like LibreSprite


def _div_trunc(n, d):
    """C integer division: truncates toward zero for negative numerators."""
    q = np.abs(n) // d
    return np.where(n < 0, -q, q)


def blend_normal(dst: np.ndarray, src: np.ndarray, opacity: int) -> np.ndarray:
    """rgba_blender_normal over (..., 4) int32 arrays."""
    Ba = dst[..., 3]
    S_alpha = src[..., 3]
    Sa = _mul_un8(S_alpha, opacity)
    Ra = Ba + Sa - _mul_un8(Ba, Sa)
    safe_Ra = np.where(Ra == 0, 1, Ra)[..., None]
    mixed = dst[..., :3] + _div_trunc((src[..., :3] - dst[..., :3]) * Sa[..., None], safe_Ra)
    out = np.concatenate([mixed, Ra[..., None]], axis=-1)
    over_empty = np.concatenate([src[..., :3], Sa[..., None]], axis=-1)
    out = np.where((Ba == 0)[..., None], over_empty, out)
    keep = (Ba != 0) & (S_alpha == 0)
    return np.where(keep[..., None], dst, out)


def _screen(b, s):
    return b + s - _mul_un8(b, s)


def _hard_light(b, s):
    return np.where(s < 128, _mul_un8(b, s << 1), _screen(b, np.maximum((s << 1) - 255, 0)))


def _color_dodge(b, s):
    inv = 255 - s
    safe = np.where(inv == 0, 1, inv)
    out = np.where(b >= inv, 255, (b * 255 + safe // 2) // safe)
    return np.where(b == 0, 0, out)


def _color_burn(b, s):
    inv = 255 - b
    safe = np.where(s == 0, 1, s)
    out = np.where(inv >= s, 0, 255 - (inv * 255 + safe // 2) // safe)
    return np.where(b == 255, 255, out)


def _soft_light(b, s):
    # LibreSprite's two branches are identical; kept as-is to match it.
    bf = b / 255.0
    sf = s / 255.0
    r = bf - (1.0 - 2.0 * sf) * bf * (1.0 - bf)
    return (r * 255 + 0.5).astype(np.int32)


_SEPARABLE = {
    1: lambda b, s: _mul_un8(b, s),
    2: _screen,
    3: lambda b, s: _hard_light(s, b),
    4: np.minimum,
    5: np.maximum,
    6: _color_dodge,
    7: _color_burn,
    8: _hard_light,
    9: _soft_light,
    10: lambda b, s: np.abs(b - s),
    11: lambda b, s: b + s - 2 * _mul_un8(b, s),
}


# HSL modes, per pixel. set_sat() binds C++ references through the MIN, MID
# and MAX macros, so ties make two names alias one channel; emulating the
# references by index (with the macros' tie-breaking) matches LibreSprite.

def _lum(c):
    return 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]


def _sat(c):
    return max(c) - min(c)


def _clip_color(c):
    l = _lum(c)
    n = min(c)
    x = max(c)
    if n < 0:
        c = [l + ((v - l) * l) / (l - n) for v in c]
    if x > 1:
        c = [l + ((v - l) * (1 - l)) / (x - l) for v in c]
    return c


def _set_lum(c, l):
    d = l - _lum(c)
    return _clip_color([v + d for v in c])


def _min_ref(c):
    gb = 1 if c[1] < c[2] else 2
    return 0 if c[0] < c[gb] else gb


def _max_ref(c):
    gb = 1 if c[1] > c[2] else 2
    return 0 if c[0] > c[gb] else gb


def _mid_ref(c):
    r, g, b = c
    if r > g:
        if g > b:
            return 1
        return 2 if r > b else 0
    if g > b:
        return 2 if b > r else 0
    return 1


def _set_sat(c, s):
    c = list(c)
    lo, mid, hi = _min_ref(c), _mid_ref(c), _max_ref(c)
    if c[hi] > c[lo]:
        c[mid] = ((c[mid] - c[lo]) * s) / (c[hi] - c[lo])
        c[hi] = s
    else:
        c[mid] = 0
        c[hi] = 0
    c[lo] = 0
    return c


def _hsl_pixel(mode, b, s):
    b = [v / 255.0 for v in b]
    s = [v / 255.0 for v in s]
    if mode == 12:  # hue
        r = _set_lum(_set_sat(s, _sat(b)), _lum(b))
    elif mode == 13:  # saturation
        r = _set_lum(_set_sat(b, _sat(s)), _lum(b))
    elif mode == 14:  # color
        r = _set_lum(s, _lum(b))
    else:  # luminosity
        r = _set_lum(b, _lum(s))
    return [int(255.0 * v) for v in r]


def blend(dst: np.ndarray, src: np.ndarray, opacity: int, mode: int) -> np.ndarray:
    """Composite src over dst ((..., 4) int32 arrays) with a blend mode.

    Source pixels equal to the image mask color (0, fully transparent
    black) are skipped, as in LibreSprite's BlenderHelper: the backdrop
    keeps its RGB even where its alpha is 0, and later layers read it.
    """
    masked = ~np.any(src, axis=-1)
    if mode in _SEPARABLE:
        rgb = _SEPARABLE[mode](dst[..., :3], src[..., :3])
        src = np.concatenate([np.clip(rgb, 0, 255), src[..., 3:4]], axis=-1)
    elif mode in (12, 13, 14, 15):
        src = src.copy()
        for idx in zip(*np.nonzero(~masked)):
            src[idx][:3] = np.clip(_hsl_pixel(mode, dst[idx][:3].tolist(), src[idx][:3].tolist()), 0, 255)
    return np.where(masked[..., None], dst, blend_normal(dst, src, opacity))


def cel_rgba(spr: ase.Sprite, image: np.ndarray) -> np.ndarray:
    """Convert a cel image to (h, w, 4) RGBA int32, transparent where masked."""
    if spr.color_mode == ase.RGB:
        return image.astype(np.int32)
    if spr.color_mode == ase.GRAY:
        v = image[..., 0:1].astype(np.int32)
        return np.concatenate([v, v, v, image[..., 1:2].astype(np.int32)], axis=-1)
    palette = np.array(spr.palette, np.int32)
    idx = image[..., 0].astype(np.intp)
    out = palette[np.clip(idx, 0, len(palette) - 1)]
    out[idx >= len(palette)] = 0
    out[idx == spr.transparent_index] = 0
    return out


def composite(spr: ase.Sprite, frame: int, *, include_hidden: bool = False,
              only_layers: set[str] | None = None) -> np.ndarray:
    """Flatten one 0-based frame into an (h, w, 4) uint8 RGBA array."""
    canvas = np.zeros((spr.height, spr.width, 4), np.int32)
    if spr.color_mode == ase.INDEXED and spr.layers and spr.layers[0].is_background \
            and (spr.layers[0].visible or include_hidden) \
            and spr.transparent_index < len(spr.palette):
        canvas[:] = spr.palette[spr.transparent_index]
    for layer in spr.layers:
        if only_layers is not None and layer.name not in only_layers:
            continue
        if not layer.visible and not include_hidden:
            continue
        cel = layer.cels.get(frame)
        if cel is None:
            continue
        # A background layer always has opacity 255 and normal blending.
        opacity = cel.opacity if layer.is_background else _mul_un8(cel.opacity, layer.opacity)
        mode = 0 if layer.is_background else layer.blend_mode
        x0, y0 = max(cel.x, 0), max(cel.y, 0)
        x1, y1 = min(cel.x + cel.width, spr.width), min(cel.y + cel.height, spr.height)
        if x0 >= x1 or y0 >= y1:
            continue
        src = cel_rgba(spr, cel.image[y0 - cel.y:y1 - cel.y, x0 - cel.x:x1 - cel.x])
        if spr.color_mode == ase.INDEXED:
            # Masked (transparent-index) pixels leave the canvas untouched.
            masked = cel.image[y0 - cel.y:y1 - cel.y, x0 - cel.x:x1 - cel.x, 0] == spr.transparent_index
            region = canvas[y0:y1, x0:x1]
            canvas[y0:y1, x0:x1] = np.where(masked[..., None], region, blend(region, src, opacity, mode))
        else:
            canvas[y0:y1, x0:x1] = blend(canvas[y0:y1, x0:x1], src, opacity, mode)
    return canvas.astype(np.uint8)


def to_pil(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(rgba, np.uint8), "RGBA")


def scale_nearest(rgba: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return rgba
    return np.repeat(np.repeat(rgba, factor, axis=0), factor, axis=1)


def checkerboard(width: int, height: int, cell: int = 8,
                 light=(204, 204, 204, 255), dark=(153, 153, 153, 255)) -> np.ndarray:
    """LibreSprite's default transparency checkerboard, for previews."""
    yy, xx = np.mgrid[0:height, 0:width]
    mask = ((xx // cell) + (yy // cell)) % 2 == 0
    out = np.empty((height, width, 4), np.int32)
    out[mask] = light
    out[~mask] = dark
    return out


def over_checkerboard(rgba: np.ndarray, cell: int = 8) -> np.ndarray:
    h, w = rgba.shape[:2]
    board = checkerboard(w, h, cell)
    return blend_normal(board, rgba.astype(np.int32), 255).astype(np.uint8)
