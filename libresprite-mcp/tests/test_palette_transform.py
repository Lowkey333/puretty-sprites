import json

import numpy as np

from conftest import fails, ok, run
from libresprite_mcp.core import ase, render
from libresprite_mcp.tools import canvas, drawing, palette, transform


def test_palette_roundtrip(sprite):
    ok(run(palette.set_palette(sprite, ["#000000", "#FF0000", "#00FF0080"])))
    assert json.loads(run(palette.get_palette(sprite))) == ["#000000", "#FF0000", "#00FF0080"]
    ok(run(palette.apply_palette_preset(sprite, "pico8")))
    assert len(json.loads(run(palette.get_palette(sprite)))) == 16
    fails(run(palette.apply_palette_preset(sprite, "nope")), "unknown preset")


def test_ramp_and_quantize(sprite):
    ramp = json.loads(run(palette.generate_color_ramp("#D04648", 5)))
    assert len(ramp) == 5 and ramp[2] != ramp[0]
    ok(run(palette.set_palette(sprite, ["#000000", "#FFFFFF", "#C83C3C"])))
    out = ok(run(palette.quantize_to_palette(sprite, "body")))
    spr = ase.read(sprite)
    body = spr.layers[1].cels[0].image
    assert tuple(body[10, 10]) == (0xC8, 0x3C, 0x3C, 255)
    assert "Quantized" in out


def test_color_mode_conversions(base_dir):
    path = base_dir + "/modes.ase"
    ok(run(canvas.create_canvas(6, 6, path)))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 0, 0, 3, 3, "#000000", True)))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 3, 3, 3, 3, "#AC3232", True)))
    before = render.composite(ase.read(path), 0)
    out = ok(run(palette.set_color_mode(path, "indexed")))
    assert "transparent palette entry" in out
    spr = ase.read(path)
    assert spr.color_mode == ase.INDEXED and spr.palette[0][3] == 0
    assert np.array_equal(render.composite(spr, 0), before)  # black stayed black, not transparent
    ok(run(palette.set_color_mode(path, "rgb")))
    assert np.array_equal(render.composite(ase.read(path), 0), before)
    ok(run(palette.set_color_mode(path, "grayscale")))
    assert ase.read(path).color_mode == ase.GRAY


def test_flip_rotate(base_dir):
    path = base_dir + "/xf.ase"
    ok(run(canvas.create_canvas(4, 2, path)))
    ok(run(drawing.draw_pixels_at(path, "Layer 1", 1, [{"x": 0, "y": 0, "color": "#FF0000"}])))
    ok(run(transform.flip_layer(path, "Layer 1", 1, "horizontal")))
    assert tuple(ase.read(path).layers[0].cels[0].image[0, 3]) == (255, 0, 0, 255)
    ok(run(transform.rotate_layer(path, "Layer 1", 1, 90)))
    c = ase.read(path).layers[0].cels[0]
    assert (c.width, c.height) == (2, 4)
    assert tuple(c.image[3, 1]) == (255, 0, 0, 255)


def test_resize_crop_canvas_size(base_dir):
    path = base_dir + "/size.ase"
    ok(run(canvas.create_canvas(4, 4, path)))
    ok(run(drawing.draw_pixels_at(path, "Layer 1", 1, [{"x": 1, "y": 1, "color": "#00FF00"}])))
    ok(run(transform.resize_canvas(path, 8, 8)))
    spr = ase.read(path)
    assert (spr.width, spr.height) == (8, 8)
    assert render.composite(spr, 0)[2:4, 2:4, 1].min() == 255
    ok(run(transform.crop_canvas(path, 2, 2, 4, 4)))
    spr = ase.read(path)
    assert (spr.width, spr.height) == (4, 4) and render.composite(spr, 0)[0, 0, 1] == 255
    ok(run(transform.canvas_size(path, 6, 8, "bottom")))
    spr = ase.read(path)
    rgba = render.composite(spr, 0)
    assert (spr.width, spr.height) == (6, 8) and rgba[4, 1, 1] == 255  # moved down 4, right 1
    fails(run(transform.canvas_size(path, 6, 8, "middle")), "unknown anchor")
