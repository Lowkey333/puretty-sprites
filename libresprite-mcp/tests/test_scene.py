"""copy_layers_between_sprites (scene.py)."""
import os

import numpy as np

from conftest import fails, ok, run
from libresprite_mcp.core import ase
from libresprite_mcp.tools import canvas, scene


def _source(base_dir):
    """3 frames: 'hero' has a linked cel on frames 1-2 and its own cel on 3."""
    path = os.path.join(base_dir, "scene_src.ase")
    spr = ase.Sprite(8, 8, ase.RGB, palette=[(0, 0, 0, 255)], durations=[100, 150, 200])
    hero = ase.Layer("hero", opacity=200, blend_mode=1)
    shared = ase.Cel(np.full((2, 2, 4), (255, 0, 0, 255), np.uint8), x=3, y=4, opacity=190)
    hero.cels[0] = hero.cels[1] = shared
    hero.cels[2] = ase.Cel(np.full((1, 1, 4), (0, 0, 255, 255), np.uint8), x=7, y=7)
    fx = ase.Layer("fx")
    fx.cels[1] = ase.Cel(np.full((8, 8, 4), (0, 255, 0, 255), np.uint8))
    spr.layers = [hero, fx]
    ase.write(spr, path)
    return path


def test_copy_creates_layer_with_cels_links_and_frames(base_dir):
    src = _source(base_dir)
    dst = os.path.join(base_dir, "scene_dst.ase")
    ok(run(canvas.create_canvas(8, 8, dst)))
    msg = ok(run(scene.copy_layers_between_sprites(src, dst, ["hero", "ghost"])))
    assert msg.endswith("(skipped missing layers: ghost)")

    out = ase.read(dst)
    assert out.durations == [100, 150, 200], "missing frames take the source durations"
    hero = next(l for l in out.layers if l.name == "hero")
    assert out.layers[-1] is hero, "new layers go on top"
    assert (hero.opacity, hero.blend_mode) == (200, 1)
    assert hero.cels[0] is hero.cels[1], "linked cels stay linked"
    assert (hero.cels[0].x, hero.cels[0].y, hero.cels[0].opacity) == (3, 4, 190)
    assert hero.cels[2].image[0, 0].tolist() == [0, 0, 255, 255]


def test_replace_false_keeps_existing_target_cels(base_dir):
    src = _source(base_dir)
    dst = os.path.join(base_dir, "scene_keep.ase")
    spr = ase.Sprite(8, 8, ase.RGB, palette=[(0, 0, 0, 255)], durations=[100, 100, 100])
    fx = ase.Layer("fx")
    fx.cels[1] = ase.Cel(np.full((1, 1, 4), 9, np.uint8))
    spr.layers = [fx]
    ase.write(spr, dst)

    ok(run(scene.copy_layers_between_sprites(src, dst, ["fx"], replace=False)))
    assert ase.read(dst).layers[0].cels[1].image.shape == (1, 1, 4)
    ok(run(scene.copy_layers_between_sprites(src, dst, ["fx"], replace=True)))
    assert ase.read(dst).layers[0].cels[1].image.shape == (8, 8, 4)


def test_no_frames_are_added_when_disabled(base_dir):
    src = _source(base_dir)
    dst = os.path.join(base_dir, "scene_short.ase")
    ok(run(canvas.create_canvas(8, 8, dst)))
    ok(run(scene.copy_layers_between_sprites(src, dst, ["hero"], create_missing_frames=False)))
    out = ase.read(dst)
    assert out.frame_count == 1
    assert sorted(next(l for l in out.layers if l.name == "hero").cels) == [0]


def test_pixels_are_converted_between_color_modes(base_dir):
    src = _source(base_dir)
    dst = os.path.join(base_dir, "scene_indexed.ase")
    spr = ase.Sprite(8, 8, ase.INDEXED, palette=[(0, 0, 0, 255), (250, 10, 10, 255), (10, 10, 250, 255)],
                     durations=[100, 100, 100])
    spr.layers = [ase.Layer("base")]
    ase.write(spr, dst)
    ok(run(scene.copy_layers_between_sprites(src, dst, ["hero"])))
    hero = next(l for l in ase.read(dst).layers if l.name == "hero")
    assert hero.cels[0].image.shape == (2, 2, 1)
    assert hero.cels[0].image[0, 0, 0] == 1, "red maps to the nearest palette entry"
    assert hero.cels[2].image[0, 0, 0] == 2


def test_errors(base_dir):
    src = _source(base_dir)
    dst = os.path.join(base_dir, "scene_err.ase")
    ok(run(canvas.create_canvas(8, 8, dst)))
    fails(run(scene.copy_layers_between_sprites(src, dst, ["nothing", "here"])), "none of the requested layers")
    fails(run(scene.copy_layers_between_sprites(src + ".nope", dst, ["hero"])), "not found")
    fails(run(scene.copy_layers_between_sprites(src, dst + ".nope", ["hero"])), "not found")
    assert run(scene.copy_layers_between_sprites(src, dst, [])) == "Layer names list cannot be empty"
    assert run(scene.copy_layers_between_sprites("../a.ase", dst, ["hero"])).startswith("Invalid filename")
