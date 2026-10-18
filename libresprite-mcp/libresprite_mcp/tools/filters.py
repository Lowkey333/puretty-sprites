"""Image filters computed in the server.

LibreSprite 1.3 cannot run filter commands from a headless script, so the
filters are reimplemented here: convolution follows LibreSprite's own
ConvolutionMatrixFilter (with its built-in matrices from data/convmatr.def)
and brightness/contrast follows the Aseprite filter the upstream tool used.
Grayscale and indexed cels are filtered as RGBA and converted back (nearest
palette color for indexed sprites).
"""
import json
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..core import ase, render
from ..core.colors import to_hex
from ..core.sprite import (ToolError, active_target, cel_rgba, drawing_cel, edit, frame0, get_cel,
                           get_layer, opaque, update_cel_rgba)
from .. import mcp

# LibreSprite's data/convmatr.def, verbatim minus comments. Format:
#   name w h cx cy { data } div bias target
# div/bias "auto" are derived from the matrix (see _parse_matrices).
_CONVMATR_DEF = r"""
brightness 1 1 0 0 { 1 } 1 +8 rgb
contrast 1 1 0 0 { 1 } 1 -8 rgb
negative 1 1 0 0 { -1 } auto auto rgb

blur-3x3 3 3 1 1
  { 1 2 1
    2 4 2
    1 2 1 } auto auto rgba

blur-3x3-hard 3 3 1 1
  { 0 1 0
    1 8 1
    0 1 0 } auto auto rgba

blur-5x5 5 5 2 2
  { 1 2 3 2 1
    2 3 4 3 2
    3 4 5 4 3
    2 3 4 3 2
    1 2 3 2 1 } auto auto rgba

blur-7x7 7 7 3 3
  { 1 2 3 4 3 2 1
    2 3 4 5 4 3 2
    3 4 5 6 5 4 3
    4 5 6 7 6 5 4
    3 4 5 6 5 4 3
    2 3 4 5 4 3 2
    1 2 3 4 3 2 1 } auto auto rgba

blur-9x9 9 9 4 4
  { 1 2 3 4 5 4 3 2 1
    2 3 4 5 6 5 4 3 2
    3 4 5 6 7 6 5 4 3
    4 5 6 7 8 7 6 5 4
    5 6 7 8 9 8 7 6 5
    4 5 6 7 8 7 6 5 4
    3 4 5 6 7 6 5 4 3
    2 3 4 5 6 5 4 3 2
    1 2 3 4 5 4 3 2 1 } auto auto rgba

blur-17x17 17 17 8 8
  {  1  2  3  4  5  6  7  8  9  8  7  6  5  4  3  2  1
     2  3  4  5  6  7  8  9 10  9  8  7  6  5  4  3  2
     3  4  5  6  7  8  9 10 11 10  9  8  7  6  5  4  3
     4  5  6  7  8  9 10 11 12 11 10  9  8  7  6  5  4
     5  6  7  8  9 10 11 12 13 12 11 10  9  8  7  6  5
     6  7  8  9 10 11 12 13 14 13 12 11 10  9  8  7  6
     7  8  9 10 11 12 13 14 15 14 13 12 11 10  9  8  7
     8  9 10 11 12 13 14 15 16 15 14 13 12 11 10  9  8
     9 10 11 12 13 14 15 16 17 16 15 14 13 12 11 10  9
     8  9 10 11 12 13 14 15 16 15 14 13 12 11 10  9  8
     7  8  9 10 11 12 13 14 15 14 13 12 11 10  9  8  7
     6  7  8  9 10 11 12 13 14 13 12 11 10  9  8  7  6
     5  6  7  8  9 10 11 12 13 12 11 10  9  8  7  6  5
     4  5  6  7  8  9 10 11 12 11 10  9  8  7  6  5  4
     3  4  5  6  7  8  9 10 11 10  9  8  7  6  5  4  3
     2  3  4  5  6  7  8  9 10  9  8  7  6  5  4  3  2
     1  2  3  4  5  6  7  8  9  8  7  6  5  4  3  2  1 } auto auto rgba

blur-5x3-left 5 3 0 1
  { 2 3 2 1 0
    6 4 3 2 1
    2 3 2 1 0 } auto auto rgba

blur-17x3-left 17 3 0 1
  { 14 16 13 12 10  8  6  4 3 2 1 0 0 0 0 0 0
    24 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
    14 16 13 12 10  8  6  4 3 2 1 0 0 0 0 0 0 } auto auto rgba

blur-3x17-top 3 17 1 0
  { 14 24 14
    16 16 16
    13 15 13
    12 14 12
    10 13 10
     8 12  8
     6 11  6
     4 10  4
     3  9  3
     2  8  2
     1  7  1
     0  6  0
     0  5  0
     0  4  0
     0  3  0
     0  2  0
     0  1  0 } auto auto rgba

blur-5x5-diagonal(\) 5 5 2 2
  { 1 1 1 0 0
    1 2 2 1 0
    1 2 3 2 1
    0 1 2 2 1
    0 0 1 1 1 } auto auto rgba

blur-5x5-diagonal(/) 5 5 2 2
  { 0 0 1 1 1
    0 1 2 2 1
    1 2 3 2 1
    1 2 2 1 0
    1 1 1 0 0 } auto auto rgba

sharpen-3x3 3 3 1 1
  { -1 -1 -1
    -1 16 -1
    -1 -1 -1 } 8 0 rgba

sharpen-5x5 5 5 2 2
  {  0 -1 -2 -1  0
    -1 -2 -4 -2 -1
    -2 -4 48 -4 -2
    -1 -2 -4 -2 -1
     0 -1 -2 -1  0 } 8 0 rgba

sharpen-7x7 7 7 3 3
  {  0 -1  -2  -4  -2 -1  0
    -1 -2  -4  -8  -4 -2 -1
    -2 -4  -8 -16  -8 -4 -2
    -4 -8 -16 224 -16 -8 -4
    -2 -4  -8 -16  -8 -4 -2
    -1 -2  -4  -8  -4 -2 -1
     0 -1  -2  -4  -2 -1 -0 } 16 0 rgba

edges-find 3 3 1 1
  { -1 -1 -1
    -1  8 -1
    -1 -1 -1 } 1 0 rgba

edges-find-horizontal 3 3 1 1
  { -1 -2 -1
     0  0  0
     1  2  1 } 1 0 rgba

edges-find-vertical 3 3 1 1
  { -1  0  1
    -2  0  2
    -1  0  1 } 1 0 rgba

misc-contour 3 3 1 1
  { 1  1  1
    1 -8  1
    1  1  1 } 1 255 rgb

misc-texturize 3 3 1 1
  { -2 -2  1
    -2  1  2
    -1  2  2 } 1 0 rgba

misc-emboss 3 3 1 1
  { -4 -2 -1
    -2  1  2
     1  2  4 } 1 0 rgba

misc-marmolize 3 3 1 1
  { -1 -1  1
    -1  0  1
    -1  1  1 } 2 128 rgb

misc-rock 3 3 1 1
  { -4 -2 -1
    -2  0  2
     1  2  4 } auto auto rgb

misc-rock-edges 3 3 1 1
  { -1 -1 -1
    -1  8 -1
    -1 -1 -1 } auto auto rgb

drunk-3x3_x 3 3 1 1
  { 1 0 1
    0 1 0
    1 0 1 } auto auto rgba

drunk-3x3_+ 3 3 1 1
  { 0 1 0
    1 1 1
    0 1 0 } auto auto rgba

drunk-5x5_x 5 5 2 2
  { 1 0 0 0 1
    0 0 0 0 0
    0 0 1 0 0
    0 0 0 0 0
    1 0 0 0 1 } auto auto rgba

drunk-5x5_+ 5 5 2 2
  { 0 0 1 0 0
    0 0 0 0 0
    1 0 1 0 1
    0 0 0 0 0
    0 0 1 0 0 } auto auto rgba

drunk-7x7_x 7 7 3 3
  { 1 0 0 0 0 0 1
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 1 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    1 0 0 0 0 0 1 } auto auto rgba

drunk-7x7_+ 7 7 3 3
  { 0 0 0 1 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    1 0 0 1 0 0 1
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 1 0 0 0 } auto auto rgba

drunk-9x9_x 9 9 4 4
  { 1 0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 0 1 } auto auto rgba

drunk-9x9_+ 9 9 4 4
  { 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    1 0 0 0 1 0 0 0 1
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 } auto auto rgba

drunk-17x17_x 17 17 8 8
  {  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 } auto auto rgba

drunk-17x17_+ 17 17 8 8
  {  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     1  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  1
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0 } auto auto rgba

drunk-17x17_o 17 17 8 8
  {  0  0  0  0  0  0  0  1  1  1  0  0  0  0  0  0  0
     0  0  0  0  0  1  1  0  0  0  1  1  0  0  0  0  0
     0  0  0  0  1  0  0  0  0  0  0  0  1  0  0  0  0
     0  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0
     0  0  1  0  0  0  0  0  0  0  0  0  0  0  1  0  0
     0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0
     0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
     0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0
     0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0
     0  0  1  0  0  0  0  0  0  0  0  0  0  0  1  0  0
     0  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0
     0  0  0  0  1  0  0  0  0  0  0  0  1  0  0  0  0
     0  0  0  0  0  1  1  0  0  0  1  1  0  0  0  0  0
     0  0  0  0  0  0  0  1  1  1  0  0  0  0  0  0  0 } auto auto rgba

outline-transparent-layer-(cross) 3 3 1 1
  { 0 255 0
    255 255 255
    0 255 0 } 1 0 a

outline-transparent-layer-(square) 3 3 1 1
  { 255 255 255
    255 255 255
    255 255 255 } 1 0 a
"""

_PRECISION = 256  # ConvolutionMatrix::Precision


@dataclass
class _Matrix:
    name: str
    w: int
    h: int
    cx: int
    cy: int
    data: list[int]  # row-major, scaled by _PRECISION
    div: int
    bias: int
    target: str  # subset of "rgba"


def _parse_matrices(text: str) -> dict[str, _Matrix]:
    """Parse convmatr.def the way ConvolutionMatrixStock::reloadStock() does."""
    tokens = text.split()
    matrices = {}
    i = 0
    while i < len(tokens):
        name = tokens[i]
        w, h, cx, cy = (int(t) for t in tokens[i + 1:i + 5])
        if tokens[i + 5] != "{":
            raise ValueError(f"Bad convolution matrix definition: {name}")
        values = [int(float(t) * _PRECISION) for t in tokens[i + 6:i + 6 + w * h]]
        i += 6 + w * h
        if tokens[i] != "}":
            raise ValueError(f"Bad convolution matrix definition: {name}")
        div_tok, bias_tok, target = tokens[i + 1:i + 4]
        i += 4
        total = sum(values)
        if total > 0:
            div, bias = total, 0
        elif total == 0:
            div, bias = _PRECISION, 128
        else:
            div, bias = abs(total), 255
        if div_tok != "auto":
            div = int(float(div_tok) * _PRECISION)
        if bias_tok != "auto":
            bias = int(float(bias_tok))
        matrices[name] = _Matrix(name, w, h, cx, cy, values, div, bias, target)
    return matrices


CONVOLUTION_MATRICES = _parse_matrices(_CONVMATR_DEF)


def _div_trunc(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    """C integer division (truncates toward zero) for any signs."""
    q = np.abs(n) // np.abs(d)
    return np.where((n < 0) != (d < 0), -q, q)


def _convolve(rgba: np.ndarray, m: _Matrix) -> np.ndarray:
    """ConvolutionMatrixFilter::applyToRgba over a whole (h, w, 4) image.

    Fully transparent neighbours are left out of the weighted sum and their
    weight is taken off the divisor, so shapes blur into transparency
    instead of darkening; edges repeat the border pixels.
    """
    h, w = rgba.shape[:2]
    src = rgba.astype(np.int64)
    padded = np.pad(src, ((m.cy, m.h - 1 - m.cy), (m.cx, m.w - 1 - m.cx), (0, 0)), mode="edge")
    acc = np.zeros((h, w, 4), np.int64)
    div = np.full((h, w), m.div, np.int64)
    for j in range(m.h):
        for i in range(m.w):
            v = m.data[j * m.w + i]
            if v == 0:
                continue
            window = padded[j:j + h, i:i + w]
            clear = window[..., 3] == 0
            div -= np.where(clear, v, 0)
            acc += np.where(clear[..., None], 0, window * v)
    zero = div == 0
    result = np.clip(_div_trunc(acc, np.where(zero, 1, div)[..., None]) + m.bias, 0, 255)
    out = src.copy()
    for channel, letter in enumerate("rgba"):
        if letter in m.target:
            out[..., channel] = result[..., channel]
    out[zero] = src[zero]
    return out.astype(np.uint8)


def _brightness_contrast_map(brightness: int, contrast: int) -> np.ndarray:
    """BrightnessContrastFilter::updateMap() with -100..100 inputs."""
    b, c = brightness / 100.0, contrast / 100.0
    x = np.arange(256) / 255.0
    y = np.clip(((c + 1.0) * (x - 0.5) + 0.5) * (1.0 + b), 0.0, 1.0)
    return (255.5 * y).astype(np.int64).astype(np.uint8)


def _region_mask(cel: ase.Cel, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Pixels of the cel inside a sprite-coordinate rectangle (all when width/height are 0)."""
    if width <= 0 or height <= 0:
        return np.ones(cel.image.shape[:2], bool)
    ys, xs = np.mgrid[0:cel.height, 0:cel.width]
    xs, ys = xs + cel.x, ys + cel.y
    return (xs >= x) & (xs < x + width) & (ys >= y) & (ys < y + height)


def _target_layer(spr: ase.Sprite, filename: str, layer_name: str) -> str:
    return get_layer(spr, layer_name).name if layer_name else active_target(spr, filename)[0]


@mcp.tool()
async def adjust_brightness_contrast(
    filename: str,
    layer_name: str = "",
    frame_index: int = 1,
    brightness: int = 0,
    contrast: int = 0,
    x: int = 0,
    y: int = 0,
    width: int = 0,
    height: int = 0,
) -> str:
    """Brightness/Contrast filter on a cel (the Aseprite filter's curve).

    Args:
        filename: Sprite file to modify
        layer_name: layer to adjust (empty = active layer)
        frame_index: 1-based frame
        brightness: -100..100
        contrast: -100..100
        x, y, width, height: optional region (width>0 & height>0 to scope)
    """
    if not (-100 <= brightness <= 100) or not (-100 <= contrast <= 100):
        return "brightness and contrast must be -100..100"
    try:
        with edit(filename) as spr:
            layer = _target_layer(spr, filename, layer_name)
            cel = get_cel(spr, get_layer(spr, layer), frame0(spr, frame_index))
            rgba = cel_rgba(spr, cel)
            rgba[..., :3] = _brightness_contrast_map(brightness, contrast)[rgba[..., :3]]
            update_cel_rgba(spr, cel, rgba, opaque(spr, cel.image) & _region_mask(cel, x, y, width, height))
    except ToolError as e:
        return f"Failed to adjust brightness/contrast: {e}"
    return f"Adjusted brightness/contrast on {layer} in {filename}"


@mcp.tool()
async def invert_colors(
    filename: str,
    layer_name: str = "",
    frame_index: int = 1,
    x: int = 0,
    y: int = 0,
    width: int = 0,
    height: int = 0,
) -> str:
    """Invert the RGB channels of a cel's opaque pixels (alpha is kept).

    Args:
        filename: Sprite file to modify
        layer_name: layer to invert (empty = active layer)
        frame_index: 1-based frame
        x, y, width, height: optional region (width>0 & height>0 to scope)
    """
    try:
        with edit(filename) as spr:
            layer = _target_layer(spr, filename, layer_name)
            cel = get_cel(spr, get_layer(spr, layer), frame0(spr, frame_index))
            rgba = cel_rgba(spr, cel)
            rgba[..., :3] ^= 0xFF
            update_cel_rgba(spr, cel, rgba, opaque(spr, cel.image) & _region_mask(cel, x, y, width, height))
    except ToolError as e:
        return f"Failed to invert: {e}"
    return f"Inverted colours on {layer} in {filename}"


@mcp.tool()
async def apply_convolution(
    filename: str,
    matrix: str,
    layer_name: str = "",
    frame_index: int = 1,
    x: int = 0,
    y: int = 0,
    width: int = 0,
    height: int = 0,
) -> str:
    """Convolution filter (blur / sharpen / edge / emboss ...) with LibreSprite's matrices.

    Computed like LibreSprite's Convolution Matrix filter: transparent
    neighbours are ignored, so a blur also softens a shape's edge outward.
    The cel is made canvas-sized first so effects can spread past its old
    bounds.

    Args:
        filename: Sprite file to modify
        matrix: a built-in matrix name (see list_convolution_matrices),
            e.g. "blur-3x3", "sharpen-3x3", "edges-find", "misc-emboss"
        layer_name: layer to filter (empty = active layer)
        frame_index: 1-based frame
        x, y, width, height: optional region (width>0 & height>0 to scope)
    """
    if matrix not in CONVOLUTION_MATRICES:
        return (f"Unknown matrix {matrix!r}; call list_convolution_matrices for "
                f"the {len(CONVOLUTION_MATRICES)} valid names")
    try:
        with edit(filename) as spr:
            layer = _target_layer(spr, filename, layer_name)
            cel = drawing_cel(spr, layer, frame_index, create=False)
            filtered = _convolve(cel_rgba(spr, cel), CONVOLUTION_MATRICES[matrix])
            update_cel_rgba(spr, cel, filtered, _region_mask(cel, x, y, width, height))
    except ToolError as e:
        return f"Failed to apply convolution: {e}"
    return f"Applied convolution '{matrix}' on {layer} in {filename}"


@mcp.tool()
async def list_convolution_matrices() -> str:
    """List the built-in convolution-matrix names usable with apply_convolution.

    Returns:
        JSON array of matrix names (LibreSprite's data/convmatr.def).
    """
    return json.dumps(sorted(CONVOLUTION_MATRICES))


def _optimal_palette(pixels: np.ndarray, max_colors: int, with_alpha: bool) -> list[tuple[int, int, int, int]]:
    """Palette of at most max_colors for (n, 4) RGBA pixels, ordered dark to light."""
    if not with_alpha:
        pixels = np.concatenate([pixels[:, :3], np.full((len(pixels), 1), 255, np.uint8)], axis=1)
    unique = np.unique(pixels, axis=0)
    if len(unique) <= max_colors:
        colors = [tuple(int(v) for v in c) for c in unique]
    else:
        mode = "RGBA" if with_alpha else "RGB"
        strip = Image.fromarray(np.ascontiguousarray(pixels[:, :4 if with_alpha else 3][None, :, :]), mode)
        method = Image.Quantize.FASTOCTREE if with_alpha else Image.Quantize.MEDIANCUT
        quantized = strip.quantize(colors=max_colors, method=method, kmeans=2)
        entries = quantized.getpalette(mode) or []
        step = len(mode)
        used = sorted(index for _, index in quantized.getcolors(256) or [])
        colors = []
        for index in used:
            c = entries[index * step:(index + 1) * step]
            colors.append((c[0], c[1], c[2], c[3] if with_alpha else 255))
        colors = list(dict.fromkeys(colors))
    return sorted(colors, key=lambda c: ((c[0] * 2126 + c[1] * 7152 + c[2] * 722) // 10000, c))


@mcp.tool()
async def extract_palette(
    filename: str,
    max_colors: int = 16,
    with_alpha: bool = False,
) -> str:
    """Build an OPTIMAL palette from the sprite's visible pixels and set it.

    True palette extraction (vs the nearest-snap quantize_to_palette):
    every frame is flattened, the visible colors are reduced to at most
    max_colors (exactly the used colors when there are few enough, else
    median cut), and the result becomes the sprite's palette, ordered dark
    to light. NOTE: replaces the sprite's palette; pixels are unchanged.
    Needs an RGB or grayscale sprite.

    Args:
        filename: Sprite file to modify
        max_colors: palette size cap, 1..256 (fewer if the art has fewer)
        with_alpha: keep semi-transparent colors as such instead of
            treating every visible pixel as opaque

    Returns:
        JSON {colors: [#RRGGBB, ...], count}
    """
    if not (1 <= max_colors <= 256):
        return "max_colors must be 1..256"
    try:
        with edit(filename) as spr:
            if spr.color_mode == ase.INDEXED:
                raise ToolError("extract_palette needs an RGB or grayscale sprite (indexed pixels "
                                "are palette indices); convert it with set_color_mode first")
            pixels = np.concatenate([render.composite(spr, f).reshape(-1, 4) for f in range(spr.frame_count)])
            pixels = pixels[pixels[:, 3] > 0]
            if not len(pixels):
                raise ToolError("The sprite has no visible pixels to build a palette from")
            colors = _optimal_palette(pixels, max_colors, with_alpha)
            spr.palette = colors
    except ToolError as e:
        return f"Failed to extract palette: {e}"
    hexes = [to_hex(c, with_alpha=c[3] != 255) for c in colors]
    return json.dumps({"colors": hexes, "count": len(hexes)})
