import json

import numpy as np

from conftest import fails, ok, run
from libresprite_mcp.core import ase
from libresprite_mcp.tools import animation, canvas


def info(path):
    return json.loads(run(animation.get_sprite_info(path)))


def cel(path, layer, frame):
    spr = ase.read(path)
    return next(l for l in spr.layers if l.name == layer).cels.get(frame - 1)


def test_frames_and_durations(sprite):
    ok(run(animation.add_frames(sprite, 3, 80)))
    i = info(sprite)
    assert i["frames"] == 4 and i["durations_ms"] == [100, 80, 80, 80]
    assert i["layers"][1]["cel_frames"] == [1]  # new frames are empty
    ok(run(animation.set_frame_duration_all(sprite, 120)))
    assert info(sprite)["durations_ms"] == [120] * 4


def test_copy_and_propagate(sprite):
    ok(run(animation.copy_cel(sprite, "body", 1, 2)))
    assert np.array_equal(cel(sprite, "body", 2).image, cel(sprite, "body", 1).image)
    fails(run(animation.copy_cel(sprite, "body", 1, 2, replace=False)), "already has a cel")
    ok(run(animation.propagate_cels(sprite, ["body"], 1, 3, 4)))
    assert info(sprite)["layers"][1]["cel_frames"] == [1, 2, 3, 4]
    ok(run(animation.copy_frame(sprite, 1)))
    assert info(sprite)["frames"] == 5


def test_tweens(sprite):
    ok(run(animation.tween_cel_positions(sprite, "body", 1, 4, 0, 0, 6, -3)))
    assert [(cel(sprite, "body", f).x, cel(sprite, "body", f).y) for f in (1, 2, 3, 4)] == \
        [(0, 0), (2, -1), (4, -2), (6, -3)]
    ok(run(animation.tween_cel_positions_eased(sprite, "body", 1, 4, 0, 0, 0, 9, "ease_in")))
    assert [cel(sprite, "body", f).y for f in (1, 2, 3, 4)] == [0, 1, 4, 9]
    ok(run(animation.offset_cel_positions(sprite, "body", 2, 3, 5, 0)))
    assert cel(sprite, "body", 2).x == 5 and cel(sprite, "body", 4).x == 0
    ok(run(animation.tween_cel_opacity_eased(sprite, "body", 1, 4, 255, 0, "linear")))
    assert [cel(sprite, "body", f).opacity for f in (1, 4)] == [255, 0]
    ok(run(animation.oscillate_cel_positions(sprite, "body", 1, 4, amplitude_y=2, cycles=0.5)))


def test_scale_tween(sprite):
    ok(run(animation.tween_cel_scale_eased(sprite, "body", 1, 3, 1.0, 0.5, "linear", source_frame_index=5)))
    assert cel(sprite, "body", 3).width == 16 and cel(sprite, "body", 3).x == 8


def test_tags_and_delete_frame(sprite):
    ok(run(animation.set_tag(sprite, "walk", 2, 4, "pingpong")))
    ok(run(animation.set_tag(sprite, "hit", 5, 5)))
    fails(run(animation.set_tag(sprite, "x", 1, 2, "pingpong_reverse")), "unsupported direction")
    ok(run(animation.delete_frame(sprite, 3)))
    tags = {t["name"]: t for t in info(sprite)["tags"]}
    assert (tags["walk"]["from"], tags["walk"]["to"], tags["walk"]["direction"]) == (2, 3, "pingpong")
    assert (tags["hit"]["from"], tags["hit"]["to"]) == (4, 4)
    ok(run(animation.delete_tag(sprite, "hit")))
    assert [t["name"] for t in info(sprite)["tags"]] == ["walk"]


def test_cel_lifecycle_and_layer_props(sprite):
    ok(run(animation.clear_cel(sprite, "body", 2)))
    assert cel(sprite, "body", 2) is None
    fails(run(animation.clear_cel(sprite, "body", 2)), "no cel")
    ok(run(animation.create_cel(sprite, "body", 2, 1, 1)))
    assert (cel(sprite, "body", 2).x, cel(sprite, "body", 2).width) == (1, 32)
    ok(run(animation.set_cel_opacity(sprite, "body", 2, 77)))
    assert cel(sprite, "body", 2).opacity == 77
    ok(run(animation.set_layer_opacity(sprite, "body", 128)))
    ok(run(animation.set_layer_visibility(sprite, "body", False)))
    layer = info(sprite)["layers"][1]
    assert (layer["opacity"], layer["visible"]) == (128, False)


def test_linked_cels_unlink_on_move(base_dir):
    path = base_dir + "/linked.ase"
    ok(run(canvas.create_canvas(4, 4, path)))
    spr = ase.read(path)
    spr.durations = [100, 100]
    shared = spr.layers[0].cels[0]
    spr.layers[0].cels[1] = shared
    ase.write(spr, path)
    ok(run(animation.set_cel_position(path, "Layer 1", 2, 3, 3)))
    spr = ase.read(path)
    assert spr.layers[0].cels[0] is not spr.layers[0].cels[1]
    assert (spr.layers[0].cels[0].x, spr.layers[0].cels[1].x) == (0, 3)
