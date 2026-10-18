"""Filters computed by the server (filters.py)."""
import json
import os

import numpy as np

from conftest import fails, ok, run
from libresprite_mcp.core import ase
from libresprite_mcp.tools import canvas, drawing, filters, pixel_read

RED = (0xD0, 0x46, 0x48, 255)


def _rgba(path, layer="body", frame=1):
    """The layer's cel as a canvas-sized RGBA array."""
    spr = ase.read(path)
    cel = next(l for l in spr.layers if l.name == layer).cels[frame - 1]
    out = np.zeros((spr.height, spr.width, 4), np.uint8)
    out[cel.y:cel.y + cel.height, cel.x:cel.x + cel.width] = cel.image
    return out


def _dot_sprite(base_dir, name, color="#00FF00", size=9):
    """A transparent canvas with one opaque pixel in the middle."""
    path = os.path.join(base_dir, name)
    ok(run(canvas.create_canvas(size, size, path)))
    ok(run(drawing.draw_pixels_at(path, "Layer 1", 1, [{"x": size // 2, "y": size // 2, "color": color}])))
    return path


def test_list_convolution_matrices():
    names = json.loads(run(filters.list_convolution_matrices()))
    assert len(names) == 39 and names == sorted(names)
    for name in ("blur-3x3", "sharpen-3x3", "misc-emboss", "blur-5x5-diagonal(\\)", "drunk-17x17_o"):
        assert name in names


def test_apply_convolution_rejects_unknown_matrix(sprite):
    res = run(filters.apply_convolution(sprite, "not-a-matrix", "body", 1))
    assert res.startswith("Unknown matrix")


def test_invert_colors_is_exact(sprite):
    before = run(pixel_read.get_pixel_color(sprite, 12, 12, "body", 1))
    r, g, b = (int(before.split(f"{c}=")[1].split(",")[0]) for c in "rgb")
    ok(run(filters.invert_colors(sprite, "body", 1)))
    after = run(pixel_read.get_pixel_color(sprite, 12, 12, "body", 1))
    assert after.startswith(f"#{255 - r:02x}{255 - g:02x}{255 - b:02x}"), (before, after)
    assert _rgba(sprite)[0, 0].tolist() == [0, 0, 0, 0], "transparent pixels stay clean"
    ok(run(filters.invert_colors(sprite, "body", 1)))  # back to red


def test_invert_region_only(sprite):
    ok(run(filters.invert_colors(sprite, "body", 1, 8, 8, 2, 2)))
    img = _rgba(sprite)
    assert img[8, 8].tolist() == [255 - RED[0], 255 - RED[1], 255 - RED[2], 255]
    assert img[10, 10].tolist() == list(RED)
    ok(run(filters.invert_colors(sprite, "body", 1, 8, 8, 2, 2)))


def test_brightness_contrast_follows_the_curve(sprite):
    ok(run(filters.adjust_brightness_contrast(sprite, "body", 1, 30, 10)))

    def curve(u):
        y = ((0.1 + 1.0) * (u / 255 - 0.5) + 0.5) * (1.0 + 0.3)
        return int(255.5 * min(1.0, max(0.0, y)))

    assert _rgba(sprite)[12, 12].tolist() == [curve(RED[0]), curve(RED[1]), curve(RED[2]), 255]
    assert run(filters.adjust_brightness_contrast(sprite, "body", 1, 101, 0)).startswith("brightness and contrast")


def test_blur_spreads_a_dot_into_transparency(base_dir):
    path = _dot_sprite(base_dir, "blur.ase")
    ok(run(filters.apply_convolution(path, "blur-3x3", "Layer 1", 1)))
    img = _rgba(path, "Layer 1")
    # Transparent neighbours carry no weight, so every pixel next to the dot
    # becomes exactly its colour and the rest stays empty.
    opaque = img[..., 3] > 0
    assert opaque[3:6, 3:6].all() and opaque.sum() == 9
    assert {tuple(c) for c in img[3:6, 3:6].reshape(-1, 4).tolist()} == {(0, 255, 0, 255)}


def test_negative_and_edges(base_dir):
    path = os.path.join(base_dir, "negative.ase")
    ok(run(canvas.create_canvas(6, 6, path)))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 0, 0, 6, 6, "#204060", True)))
    ok(run(filters.apply_convolution(path, "negative", "Layer 1", 1)))
    assert _rgba(path, "Layer 1")[2, 2].tolist() == [255 - 0x20, 255 - 0x40, 255 - 0x60, 255]
    ok(run(filters.apply_convolution(path, "sharpen-3x3", "Layer 1", 1)))
    assert _rgba(path, "Layer 1")[2, 2].tolist() == [255 - 0x20, 255 - 0x40, 255 - 0x60, 255], \
        "sharpening a flat area changes nothing"
    ok(run(filters.apply_convolution(path, "edges-find", "Layer 1", 1)))
    assert _rgba(path, "Layer 1")[2, 2].tolist() == [0, 0, 0, 0], "edges-find clears flat interiors (rgba target)"


def test_convolution_region_scope(base_dir):
    path = _dot_sprite(base_dir, "blur_region.ase")
    ok(run(filters.apply_convolution(path, "blur-3x3", "Layer 1", 1, 0, 0, 5, 9)))
    opaque = _rgba(path, "Layer 1")[..., 3] > 0
    assert opaque[3:6, 3:5].all() and not opaque[:, 5:].any(), "pixels right of x=4 were outside the region"


def test_filters_on_grayscale(base_dir):
    path = os.path.join(base_dir, "gray_filter.ase")
    ok(run(canvas.create_canvas(4, 4, path, "gray")))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 0, 0, 4, 4, "#404040", True)))
    ok(run(filters.invert_colors(path, "Layer 1", 1)))
    cel = ase.read(path).layers[0].cels[0]
    assert cel.image.shape[2] == 2 and cel.image[0, 0].tolist() == [255 - 0x40, 255]


def test_filter_errors(sprite):
    fails(run(filters.invert_colors(sprite, "nope", 1)), "layer not found")
    fails(run(filters.apply_convolution(sprite, "blur-3x3", "body", 5)), "frame index out of range")
    ok(run(canvas.add_frame(sprite)))
    fails(run(filters.invert_colors(sprite, "body", 2)), "no cel")


def test_extract_palette_exact_when_few_colors(base_dir):
    path = os.path.join(base_dir, "palette_exact.ase")
    ok(run(canvas.create_canvas(8, 8, path)))
    for x, color in ((0, "#FFFFFF"), (1, "#000000"), (2, "#FF0000")):
        ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, x, 0, 1, 2, color, True)))
    result = json.loads(ok(run(filters.extract_palette(path, 16))))
    assert result == {"colors": ["#000000", "#FF0000", "#FFFFFF"], "count": 3}
    assert ase.read(path).palette == [(0, 0, 0, 255), (255, 0, 0, 255), (255, 255, 255, 255)]


def test_extract_palette_reduces_many_colors(base_dir):
    path = os.path.join(base_dir, "palette_many.ase")
    spr = ase.Sprite(16, 16, ase.RGB, palette=[(0, 0, 0, 255)])
    layer = ase.Layer("art")
    yy, xx = np.mgrid[0:16, 0:16]
    img = np.stack([xx * 16, yy * 16, np.full_like(xx, 128), np.full_like(xx, 255)], axis=-1).astype(np.uint8)
    layer.cels[0] = ase.Cel(img)
    spr.layers.append(layer)
    ase.write(spr, path)
    result = json.loads(ok(run(filters.extract_palette(path, 8))))
    assert 1 <= result["count"] <= 8
    assert all(c.startswith("#") and len(c) == 7 for c in result["colors"])
    assert len(ase.read(path).palette) == result["count"]


def test_extract_palette_alpha_and_errors(base_dir):
    path = os.path.join(base_dir, "palette_alpha.ase")
    ok(run(canvas.create_canvas(4, 4, path)))
    ok(run(drawing.draw_pixels_at(path, "Layer 1", 1, [{"x": 0, "y": 0, "color": "#FF000080"}])))
    assert json.loads(ok(run(filters.extract_palette(path, 4, with_alpha=True))))["colors"] == ["#FF000080"]
    assert json.loads(ok(run(filters.extract_palette(path, 4))))["colors"] == ["#FF0000"]
    assert run(filters.extract_palette(path, 0)) == "max_colors must be 1..256"

    empty = os.path.join(base_dir, "palette_empty.ase")
    ok(run(canvas.create_canvas(4, 4, empty)))
    fails(run(filters.extract_palette(empty)), "no visible pixels")
    indexed = os.path.join(base_dir, "palette_indexed.ase")
    ok(run(canvas.create_canvas(4, 4, indexed, "indexed")))
    fails(run(filters.extract_palette(indexed)), "rgb or grayscale")
