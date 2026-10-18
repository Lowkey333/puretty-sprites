"""Visual feedback tools (analysis.py)."""
import json
import os

import numpy as np
from PIL import Image

from conftest import fails, ok, run
from libresprite_mcp.core import ase, render
from libresprite_mcp.tools import analysis, canvas, drawing


def _add_frame_copying(path, source_frame):
    """Append a frame whose cels are copies of source_frame (1-based)."""
    spr = ase.read(path)
    spr.durations.append(100)
    for layer in spr.layers:
        cel = layer.cels.get(source_frame - 1)
        if cel is not None:
            layer.cels[spr.frame_count - 1] = cel.copy()
    ase.write(spr, path)


def test_render_onion_skin(sprite, base_dir):
    # frame 2 = frame 1 with the rect moved 8px right; frame 3 = a copy of 2.
    _add_frame_copying(sprite, 1)
    _add_frame_copying(sprite, 1)
    spr = ase.read(sprite)
    body = next(l for l in spr.layers if l.name == "body")
    for frame in (1, 2):
        body.cels[frame].x = 8
    ase.write(spr, sprite)

    out = os.path.join(base_dir, "onion.png")
    ok(run(analysis.render_onion_skin(sprite, 2, out, 1, 1, 4, 100)))
    img = np.array(Image.open(out).convert("RGBA"))
    assert img.shape == (128, 128, 4)
    # (2,2) is empty in every frame: white background.
    assert img[2 * 4, 2 * 4].tolist() == [255, 255, 255, 255]
    # (10,10) is only in frame 1 (the ghost before): red at opacity 100 over white.
    ghost = render.blend_normal(np.array([[255, 255, 255, 255]]), np.array([[0xD0, 0x46, 0x48, 255]]), 100)[0]
    assert img[10 * 4, 10 * 4].tolist() == ghost.tolist()
    # (20,20) is in frame 2 itself: solid red.
    assert img[20 * 4, 20 * 4].tolist() == [0xD0, 0x46, 0x48, 255]


def test_onion_skin_tint_and_extension(sprite, base_dir):
    out = os.path.join(base_dir, "onion_tint")
    msg = ok(run(analysis.render_onion_skin(sprite, 2, out, 1, 0, 1, 255, tint=True)))
    assert msg.endswith("at 1x") and "onion_tint.png" in msg
    img = np.array(Image.open(out + ".png").convert("RGBA"))
    # LibreSprite's red tint of #D04648: luma v -> ((255+v)/2, v/2, v/2).
    v = (0xD0 * 2126 + 0x46 * 7152 + 0x48 * 722) // 10000
    assert img[10, 10].tolist() == [(255 + v) // 2, v // 2, v // 2, 255]


def test_onion_skin_rejects_bad_arguments(sprite, base_dir):
    out = os.path.join(base_dir, "x.png")
    assert run(analysis.render_onion_skin(sprite, 1, out, scale=0)).startswith("scale must")
    assert run(analysis.render_onion_skin(sprite, 1, out, ghost_opacity=300)).startswith("ghost_opacity")
    assert run(analysis.render_onion_skin(sprite, 1, "../x.png")).startswith("Invalid filename")
    fails(run(analysis.render_onion_skin(sprite, 9, out)), "frame index out of range")


def test_compare_identical_frames(sprite):
    # frames 2 and 3 are identical copies
    diff = json.loads(ok(run(analysis.compare_frames(sprite, 2, 3))))
    assert diff["total_pixels"] == 1024
    assert diff["changed_pixels"] == 0
    assert diff["percent_changed"] == 0
    assert "changed_bounds" not in diff


def test_compare_differing_frames(sprite):
    ok(run(drawing.draw_rectangle_at(sprite, "body", 3, 0, 0, 4, 4, "#FFFFFF", True)))
    diff = json.loads(ok(run(analysis.compare_frames(sprite, 2, 3))))
    assert diff["changed_pixels"] == 16
    assert diff["changed_bounds"] == {"x": 0, "y": 0, "width": 4, "height": 4}
    # the 8px move between frames 1 and 2 changes two 8x16 strips
    moved = json.loads(ok(run(analysis.compare_frames(sprite, 1, 2))))
    assert moved["changed_pixels"] == 2 * 8 * 16
    assert moved["changed_bounds"] == {"x": 8, "y": 8, "width": 24, "height": 16}


def test_get_color_stats(sprite):
    stats = json.loads(ok(run(analysis.get_color_stats(sprite, 1))))
    assert stats == {"frame": 1, "unique_colors": 1, "opaque_pixels": 256,
                     "top_colors": [{"color": "#D04648", "count": 256}]}
    stats3 = json.loads(ok(run(analysis.get_color_stats(sprite, 3, top=1))))
    assert stats3["unique_colors"] == 2 and stats3["opaque_pixels"] == 256 + 16
    assert stats3["top_colors"] == [{"color": "#D04648", "count": 256}]


def test_color_stats_errors(sprite):
    assert run(analysis.get_color_stats(sprite, 1, top=0)) == "top must be >= 1"
    fails(run(analysis.get_color_stats(sprite, 99)), "frame index out of range")
    fails(run(analysis.compare_frames(sprite, 1, 99)), "frame index out of range")


def test_hidden_layers_are_ignored(base_dir):
    path = os.path.join(base_dir, "hidden_stats.ase")
    ok(run(canvas.create_canvas(4, 4, path)))
    ok(run(drawing.draw_rectangle_at(path, "Layer 1", 1, 0, 0, 4, 4, "#00FF00", True)))
    spr = ase.read(path)
    spr.layers[0].visible = False
    ase.write(spr, path)
    stats = json.loads(ok(run(analysis.get_color_stats(path, 1))))
    assert stats["opaque_pixels"] == 0 and stats["top_colors"] == []
