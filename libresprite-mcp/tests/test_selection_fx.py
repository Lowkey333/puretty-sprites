import json

import numpy as np

from conftest import fails, ok, run
from libresprite_mcp.core import ase
from libresprite_mcp.tools import canvas, drawing, fx, selection


def img(path, layer="body", frame=1):
    spr = ase.read(path)
    l = next(l for l in spr.layers if l.name == layer)
    return l.cels[frame - 1].image


def test_move_copy_erase(sprite):
    ok(run(selection.move_region(sprite, "body", 1, 8, 8, 4, 4, 0, 0)))
    a = img(sprite)
    assert tuple(a[0, 0]) == (0xD0, 0x46, 0x48, 255)
    assert a[8, 8, 3] == 0 and a[12, 12, 3] == 255  # source area cleared, rest kept
    ok(run(selection.copy_region(sprite, "body", 1, 0, 0, 4, 4, 28, 28, target_layer_name="Layer 1")))
    assert tuple(img(sprite, "Layer 1")[28, 28]) == (0xD0, 0x46, 0x48, 255)
    ok(run(selection.erase_region(sprite, "body", 1, 0, 0, 2, 2)))
    assert img(sprite)[0, 0, 3] == 0 and img(sprite)[2, 2, 3] == 255


def test_erase_and_replace_color(sprite):
    ok(run(fx.replace_color(sprite, "body", 1, "#D04648", "#00FF00", 4)))
    a = img(sprite)
    assert tuple(a[15, 15]) == (0, 255, 0, 255)
    out = ok(run(selection.erase_color(sprite, "body", 1, "#00FF00")))
    assert img(sprite)[15, 15, 3] == 0
    assert "Erased" in out


def test_outline_inside_and_out(base_dir):
    path = base_dir + "/outline.ase"
    ok(run(canvas.create_canvas(10, 10, path)))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 3, 3, 4, 4, "#FFFFFF", True)))
    ok(run(fx.outline_cel(path, "Layer 1", 1, "#000000")))
    a = img(path, "Layer 1")
    assert tuple(a[2, 4]) == (0, 0, 0, 255) and a[2, 2, 3] == 0  # no diagonal corner
    ok(run(fx.outline_cel(path, "Layer 1", 1, "#FF0000", include_diagonals=True)))
    a = img(path, "Layer 1")
    assert tuple(a[2, 2]) == (255, 0, 0, 255)  # corner next to the black outline
    assert tuple(a[2, 1]) == (255, 0, 0, 255)  # only reached diagonally
    assert a[1, 1, 3] == 0
    fails(run(fx.outline_cel(path, "Layer 1", 1, place="middle")), "place")


def test_hsl_and_dither(base_dir):
    path = base_dir + "/hsl.ase"
    ok(run(canvas.create_canvas(8, 8, path)))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 0, 0, 8, 8, "#FF0000", True)))
    ok(run(fx.adjust_hsl(path, "Layer 1", 1, hue_shift=120, x=0, y=0, width=4, height=8)))
    a = img(path, "Layer 1")
    assert tuple(a[0, 0]) == (0, 255, 0, 255) and tuple(a[0, 7]) == (255, 0, 0, 255)
    ok(run(fx.apply_dither_pattern(path, "Layer 1", 1, 0, 0, 4, 4, "#000000", "#FFFFFF", 0.5)))
    block = img(path, "Layer 1")[:4, :4, 0]
    assert (block == 255).sum() == 8  # half and half
    ok(run(fx.apply_dither_gradient(path, "Layer 1", 1, 0, 0, 8, 8, "#000000", "#FFFFFF")))
    g = img(path, "Layer 1")[..., 0]
    assert (g[0] == 255).sum() < (g[7] == 255).sum()
    assert set(np.unique(g)) <= {0, 255}


def test_indexed_effects_keep_palette(base_dir):
    path = base_dir + "/idx.ase"
    ok(run(canvas.create_canvas(4, 4, path, color_mode="indexed")))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 0, 0, 4, 4, "#AC3232", True)))
    spr = ase.read(path)
    index = spr.layers[0].cels[0].image[0, 0, 0]
    assert spr.palette[index][:3] == (0xAC, 0x32, 0x32)
    ok(run(fx.replace_color(path, "Layer 1", 1, "#AC3232", "#639BFF")))
    spr = ase.read(path)
    assert spr.palette[spr.layers[0].cels[0].image[0, 0, 0]][:3] == (0x63, 0x9B, 0xFF)
