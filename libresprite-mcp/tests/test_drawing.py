import numpy as np

from conftest import ok, run, fails
from libresprite_mcp.core import ase
from libresprite_mcp.tools import canvas, drawing


def px(path, layer, frame, x, y):
    spr = ase.read(path)
    l = next(l for l in spr.layers if l.name == layer)
    cel = l.cels[frame - 1]
    return tuple(int(v) for v in cel.image[y - cel.y, x - cel.x])


def test_fixture_rectangle(sprite):
    assert px(sprite, "body", 1, 8, 8) == (0xD0, 0x46, 0x48, 255)
    assert px(sprite, "body", 1, 23, 23) == (0xD0, 0x46, 0x48, 255)
    assert px(sprite, "body", 1, 24, 24) == (0, 0, 0, 0)


def test_draw_pixels_at_and_alpha(sprite):
    ok(run(drawing.draw_pixels_at(sprite, "body", 1, [{"x": 0, "y": 0, "color": "#11223380"}])))
    assert px(sprite, "body", 1, 0, 0) == (0x11, 0x22, 0x33, 0x80)


def test_active_cel_tools_target_top_layer(sprite):
    ok(run(drawing.draw_line(sprite, 0, 31, 31, 31, "#00FF00")))
    assert px(sprite, "body", 1, 15, 31) == (0, 255, 0, 255)
    ok(run(canvas.set_layer(sprite, "Layer 1")))
    ok(run(drawing.draw_pixels(sprite, [{"x": 1, "y": 1, "color": "#0000FF"}])))
    assert px(sprite, "Layer 1", 1, 1, 1) == (0, 0, 255, 255)


def test_shapes(sprite):
    ok(run(drawing.draw_circle_at(sprite, "body", 1, 16, 16, 5, "#FFFFFF", False)))
    spr = ase.read(sprite)
    img = next(l for l in spr.layers if l.name == "body").cels[0].image
    white = np.all(img == (255, 255, 255, 255), axis=-1)
    ys, xs = np.nonzero(white)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (11, 21, 11, 21)
    assert white[11, 16] and white[21, 16] and white[16, 11] and white[16, 21]
    assert not white[16, 16]  # outline only
    ok(run(drawing.draw_ellipse_at(sprite, "body", 1, 16, 16, 6, 3, "#FFFF00", True)))
    ok(run(drawing.draw_polygon(sprite, "body", 1, [{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 0, "y": 6}], "#123456", True)))
    assert px(sprite, "body", 1, 1, 1) == (0x12, 0x34, 0x56, 255)
    ok(run(drawing.draw_path(sprite, "body", 1, [{"x": 0, "y": 10}, {"x": 5, "y": 10}, {"x": 5, "y": 15}], "#ABCDEF", 2)))
    assert px(sprite, "body", 1, 5, 15) == (0xAB, 0xCD, 0xEF, 255)


def test_fill_area_contiguous(sprite, base_dir):
    path = base_dir + "/fill.ase"
    ok(run(canvas.create_canvas(10, 10, path)))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 2, 2, 6, 6, "#000000", False)))
    ok(run(drawing.fill_area_at(path, "Layer 1", 1, 4, 4, "#FF0000")))
    assert px(path, "Layer 1", 1, 4, 4) == (255, 0, 0, 255)
    assert px(path, "Layer 1", 1, 0, 0) == (0, 0, 0, 0)  # outside the box untouched
    assert px(path, "Layer 1", 1, 2, 2) == (0, 0, 0, 255)


def test_gradient(sprite):
    ok(run(drawing.apply_gradient_rect(sprite, "body", 1, 0, 20, 5, 1, "#000000", "#FF0000", True)))
    assert px(sprite, "body", 1, 0, 20) == (0, 0, 0, 255)
    assert px(sprite, "body", 1, 4, 20) == (255, 0, 0, 255)
    assert px(sprite, "body", 1, 2, 20)[0] == 128


def test_errors(sprite):
    fails(run(drawing.draw_rectangle_at(sprite, "nope", 1, 0, 0, 2, 2)), "layer not found")
    fails(run(drawing.draw_rectangle_at(sprite, "body", 9, 0, 0, 2, 2)), "frame index out of range")
    fails(run(drawing.draw_line_at(sprite, "body", 1, 0, 0, 1, 1, "#XYZ")), "invalid color")
    fails(run(drawing.draw_pixels_at(sprite + ".missing", "body", 1, [])), "not found")
