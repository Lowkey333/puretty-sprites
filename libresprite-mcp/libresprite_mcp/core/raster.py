"""Drawing primitives on cel images.

Images are (height, width, channels) uint8 arrays in the sprite's pixel
format and a pixel value is a tuple with one entry per channel (see
colors.to_pixel). Everything is bounds-checked: shapes may extend past
the image and are clipped. Drawing replaces pixels; nothing is blended.
"""
from __future__ import annotations

from collections import deque

import numpy as np

BAYER_4X4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
])


def put_points(img: np.ndarray, points, value) -> int:
    """Set every in-bounds (x, y) in points; returns how many were set."""
    if not points:
        return 0
    pts = np.asarray(list(points), dtype=np.int64).reshape(-1, 2)
    h, w = img.shape[:2]
    keep = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < w) & (pts[:, 1] < h)
    pts = pts[keep]
    img[pts[:, 1], pts[:, 0]] = value
    return int(len(pts))


def line_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham line, both endpoints included."""
    points = []
    dx, sx = abs(x1 - x0), (1 if x0 < x1 else -1)
    dy, sy = -abs(y1 - y0), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return points
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def thicken(points, thickness: int) -> set[tuple[int, int]]:
    """Stamp a thickness x thickness square brush centred on each point."""
    if thickness <= 1:
        return set(points)
    lo = -((thickness - 1) // 2)
    offsets = [(ox, oy) for oy in range(lo, lo + thickness) for ox in range(lo, lo + thickness)]
    return {(x + ox, y + oy) for x, y in points for ox, oy in offsets}


def line(img, x0, y0, x1, y1, value, thickness: int = 1) -> int:
    return put_points(img, thicken(line_points(x0, y0, x1, y1), thickness), value)


def polyline(img, points, value, thickness: int = 1, closed: bool = False) -> int:
    pts = [(int(p[0]), int(p[1])) for p in points]
    if closed and len(pts) > 2:
        pts = pts + [pts[0]]
    stroke: list[tuple[int, int]] = []
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        stroke.extend(line_points(ax, ay, bx, by))
    return put_points(img, thicken(stroke, thickness), value)


def rect(img, x: int, y: int, w: int, h: int, value, fill: bool = False) -> int:
    x1, y1 = x + w - 1, y + h - 1
    if fill:
        pts = [(px, py) for py in range(y, y1 + 1) for px in range(x, x1 + 1)]
    else:
        pts = [(px, y) for px in range(x, x1 + 1)] + [(px, y1) for px in range(x, x1 + 1)]
        pts += [(x, py) for py in range(y, y1 + 1)] + [(x1, py) for py in range(y, y1 + 1)]
    return put_points(img, pts, value)


def ellipse_points(x0: int, y0: int, x1: int, y1: int) -> set[tuple[int, int]]:
    """Outline of the ellipse inscribed in a rectangle (corners inclusive).

    Alois Zingl's plotEllipseRect, the algorithm Aseprite's ellipse tool
    uses: symmetric, gap-free and good-looking at pixel-art sizes,
    including even widths and heights.
    """
    pts: set[tuple[int, int]] = set()
    a, b = abs(x1 - x0), abs(y1 - y0)
    b1 = b & 1
    dx = 4 * (1 - a) * b * b
    dy = 4 * (b1 + 1) * a * a
    err = dx + dy + b1 * a * a
    if x0 > x1:
        x0, x1 = x1, x1 + a
    if y0 > y1:
        y0 = y1
    y0 += (b + 1) // 2
    y1 = y0 - b1
    a8, b8 = 8 * a * a, 8 * b * b
    while True:
        pts.update({(x1, y0), (x0, y0), (x0, y1), (x1, y1)})
        e2 = 2 * err
        if e2 <= dy:
            y0 += 1
            y1 -= 1
            dy += a8
            err += dy
        if e2 >= dx or 2 * err > dy:
            x0 += 1
            x1 -= 1
            dx += b8
            err += dx
        if x0 > x1:
            break
    while y0 - y1 <= b:
        pts.update({(x0 - 1, y0), (x1 + 1, y0), (x0 - 1, y1), (x1 + 1, y1)})
        y0 += 1
        y1 -= 1
    return pts


def _fill_rows(outline: set[tuple[int, int]]) -> set[tuple[int, int]]:
    rows: dict[int, list[int]] = {}
    for x, y in outline:
        rows.setdefault(y, []).append(x)
    return {(x, y) for y, xs in rows.items() for x in range(min(xs), max(xs) + 1)}


def ellipse(img, x0, y0, x1, y1, value, fill: bool = False) -> int:
    pts = ellipse_points(x0, y0, x1, y1)
    return put_points(img, _fill_rows(pts) if fill else pts, value)


def polygon_fill_points(points) -> set[tuple[int, int]]:
    """Even-odd scanline fill, rounding span ends like the upstream tool."""
    pts = [(int(p[0]), int(p[1])) for p in points]
    filled: set[tuple[int, int]] = set()
    ys = [p[1] for p in pts]
    for y in range(min(ys), max(ys) + 1):
        nodes = []
        j = len(pts) - 1
        for i, (xi, yi) in enumerate(pts):
            xj, yj = pts[j]
            if (yi < y <= yj) or (yj < y <= yi):
                nodes.append(xi + (y - yi) * (xj - xi) / (yj - yi))
            j = i
        nodes.sort()
        for k in range(0, len(nodes) - 1, 2):
            start = int(np.floor(nodes[k] + 0.5))
            end = int(np.floor(nodes[k + 1] + 0.5))
            filled.update((x, y) for x in range(start, end + 1))
    return filled


def polygon(img, points, value, fill: bool = False) -> int:
    count = put_points(img, polygon_fill_points(points), value) if fill else 0
    return count + polyline(img, points, value, closed=True)


def same_color_mask(img: np.ndarray, value, tolerance: int = 0, alpha_channel: int | None = None) -> np.ndarray:
    """Pixels whose every channel is within tolerance of value.

    For images with an alpha channel all fully transparent pixels count as
    the same colour, as in LibreSprite's paint bucket.
    """
    value = np.asarray(value, np.int32).reshape(-1)
    mask = np.all(np.abs(img.astype(np.int32) - value) <= tolerance, axis=-1)
    if alpha_channel is not None:
        clear = img[..., alpha_channel] == 0
        mask = clear if value[alpha_channel] == 0 else mask & ~clear
    return mask


def flood_region(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """4-connected region of True pixels in mask that contains (x, y)."""
    h, w = mask.shape
    region = np.zeros_like(mask)
    if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
        return region
    queue = deque([(x, y)])
    region[y, x] = True
    while queue:
        cx, cy = queue.popleft()
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not region[ny, nx]:
                region[ny, nx] = True
                queue.append((nx, ny))
    return region


def flood_fill(img, x: int, y: int, value, alpha_channel: int | None) -> int:
    """Paint-bucket fill from (x, y); returns the number of pixels filled."""
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return 0
    region = flood_region(same_color_mask(img, img[y, x], 0, alpha_channel), x, y)
    img[region] = value
    return int(region.sum())


def opaque_mask(img: np.ndarray, alpha_channel: int | None, transparent_index: int | None) -> np.ndarray:
    if alpha_channel is not None:
        return img[..., alpha_channel] > 0
    return img[..., 0] != transparent_index


def outline_mask(opaque: np.ndarray, diagonals: bool = False, inside: bool = False) -> np.ndarray:
    """Pixels to paint for a 1px outline around (or just inside) opaque pixels."""
    padded = np.pad(opaque, 1)
    h, w = opaque.shape
    neighbours = [(0, 1), (2, 1), (1, 0), (1, 2)]
    if diagonals:
        neighbours += [(0, 0), (2, 0), (0, 2), (2, 2)]
    if inside:
        clear = ~padded
        touch = np.zeros_like(opaque)
        for dy, dx in neighbours:
            touch |= clear[dy:dy + h, dx:dx + w]
        return opaque & touch
    touch = np.zeros_like(opaque)
    for dy, dx in neighbours:
        touch |= padded[dy:dy + h, dx:dx + w]
    return ~opaque & touch


def bayer_threshold(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Ordered-dither threshold in (0, 1) for sprite coordinates."""
    return (BAYER_4X4[ys % 4, xs % 4] + 0.5) / 16
