import numpy as np

from conftest import fails, ok, run
from libresprite_mcp.core import ase, render
from libresprite_mcp.tools import canvas, drawing, layers


def names(path):
    return [l.name for l in ase.read(path).layers]


def test_rename_duplicate_reorder(sprite):
    ok(run(layers.rename_layer(sprite, "body", "torso")))
    ok(run(layers.duplicate_layer(sprite, "torso")))
    assert names(sprite) == ["Layer 1", "torso", "torso copy"]
    spr = ase.read(sprite)
    assert np.array_equal(spr.layers[1].cels[0].image, spr.layers[2].cels[0].image)
    ok(run(layers.reorder_layer(sprite, "torso copy", 1)))
    assert names(sprite) == ["torso copy", "Layer 1", "torso"]
    fails(run(layers.rename_layer(sprite, "torso", "Layer 1")), "already exists")
    fails(run(layers.reorder_layer(sprite, "torso", 9)), "position")


def test_blend_mode(sprite):
    ok(run(layers.set_layer_blend_mode(sprite, "torso", "Multiply")))
    assert ase.read(sprite).layers[2].blend_mode == ase.BLEND_MODES.index("multiply")
    fails(run(layers.set_layer_blend_mode(sprite, "torso", "addition")), "unknown blend mode")


def test_merge_down_matches_render(base_dir):
    path = base_dir + "/merge.ase"
    ok(run(canvas.create_canvas(8, 8, path)))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 0, 0, 8, 8, "#336699", True)))
    ok(run(canvas.add_layer(path, "top")))
    ok(run(drawing.draw_rectangle_at(path, "top", 1, 2, 2, 4, 4, "#FF8800", True)))
    ok(run(layers.set_layer_blend_mode(path, "top", "screen")))
    before = render.composite(ase.read(path), 0)
    ok(run(layers.merge_layer_down(path, "top")))
    spr = ase.read(path)
    assert [l.name for l in spr.layers] == ["Layer 1"]
    assert np.array_equal(render.composite(spr, 0), before)
    fails(run(layers.merge_layer_down(path, "Layer 1")), "bottom layer")


def test_flatten_and_delete(base_dir):
    path = base_dir + "/flat.ase"
    ok(run(canvas.create_canvas(8, 8, path)))
    ok(run(canvas.add_layer(path, "a")))
    ok(run(canvas.add_layer(path, "hidden")))
    ok(run(drawing.draw_pixels_at(path, "a", 1, [{"x": 1, "y": 1, "color": "#FF0000"}])))
    ok(run(drawing.draw_pixels_at(path, "hidden", 1, [{"x": 2, "y": 2, "color": "#00FF00"}])))
    from libresprite_mcp.tools import animation
    ok(run(animation.set_layer_visibility(path, "hidden", False)))
    before = render.composite(ase.read(path), 0)
    out = ok(run(layers.flatten_sprite(path)))
    assert "hidden" in out
    spr = ase.read(path)
    assert [l.name for l in spr.layers] == ["Flattened"]
    assert np.array_equal(render.composite(spr, 0), before)
    fails(run(layers.delete_layer(path, "Flattened")), "only layer")
