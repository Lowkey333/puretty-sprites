import json
import os

import numpy as np
from PIL import Image

from conftest import fails, ok, run
from libresprite_mcp.core import ase, pixelize, render
from libresprite_mcp.tools import animation, canvas, drawing, export, pipeline, slices


def test_exports(sprite, base_dir):
    ok(run(animation.add_frames(sprite, 2, 50, copy_previous=True)))
    ok(run(animation.tween_cel_positions(sprite, "body", 1, 3, 0, 0, 4, 0)))
    ok(run(animation.set_tag(sprite, "loop", 1, 3, "pingpong")))
    sheet = base_dir + "/sheet.png"
    data = base_dir + "/sheet.json"
    ok(run(slices.create_slice(sprite, "feet", 8, 20, 16, 4)))
    ok(run(slices.set_slice_pivot(sprite, "feet", 8, 4)))
    ok(run(export.export_spritesheet(sprite, sheet, "horizontal", data, scale=2, padding=1, list_tags=True)))
    im = Image.open(sheet)
    assert im.size == (3 * 64 + 2, 64)
    meta = json.load(open(data))
    assert meta["frames"][1]["frame"] == {"x": 65, "y": 0, "w": 64, "h": 64}
    assert meta["meta"]["frameTags"][0]["direction"] == "pingpong"
    assert meta["meta"]["slices"][0]["keys"][0]["pivot"] == {"x": 8, "y": 4}
    gif = base_dir + "/loop.gif"
    ok(run(export.export_tag(sprite, "loop", gif, 2)))
    assert Image.open(gif).n_frames == 4  # pingpong 1,2,3,2
    ok(run(export.export_sprite(sprite, base_dir + "/seq.png")))
    assert os.path.isfile(base_dir + "/seq_1.png") and os.path.isfile(base_dir + "/seq_3.png")
    ok(run(export.export_frame(sprite, 1, base_dir + "/f1.png", 4)))
    assert Image.open(base_dir + "/f1.png").size == (128, 128)
    ok(run(export.export_layers(sprite, base_dir + "/layers")))
    assert os.path.isfile(base_dir + "/layers/body.png")


def test_view_tools_return_images(sprite):
    result = run(export.view_frame(sprite, 1, grid=True))
    assert result[0].__class__.__name__ == "Image" and "frame 1" in result[1]
    result = run(export.view_animation(sprite))
    assert result[0].__class__.__name__ == "Image" and "frames" in result[1]
    assert "Failed" in run(export.view_frame(sprite, 99))[0]


def test_import_image(sprite, base_dir):
    path = base_dir + "/stamp.png"
    Image.fromarray(np.full((3, 2, 4), (1, 2, 3, 255), np.uint8)).save(path)
    ok(run(export.import_image_as_layer(sprite, path, "stamp", 1, 5, 6)))
    c = next(l for l in ase.read(sprite).layers if l.name == "stamp").cels[0]
    assert (c.x, c.y, c.width, c.height) == (5, 6, 2, 3)


def fake_ai_image(path, block=8, bg=(250, 250, 250)):
    """An upscaled 'pixel art' blob with soft noise on a flat background."""
    rng = np.random.default_rng(0)
    small = np.zeros((16, 16, 3), np.uint8)
    small[:] = bg
    small[4:12, 5:11] = (200, 60, 60)
    small[6:8, 6:10] = (40, 40, 120)
    big = np.repeat(np.repeat(small, block, 0), block, 1).astype(int)
    big += rng.integers(-6, 7, big.shape)
    Image.fromarray(np.clip(big, 0, 255).astype(np.uint8), "RGB").save(path)
    return small


def test_pixelize_detects_grid_and_removes_background(base_dir):
    src = base_dir + "/ai.png"
    small = fake_ai_image(src)
    size, ox, oy, conf = pixelize.detect_pixel_size(pixelize.load_rgba(src))
    assert size == 8 and conf > 0.6
    out = base_dir + "/ai_px.png"
    result = run(pipeline.pixelize_image(src, out, palette="auto:4"))
    assert "16x16" in result[1] and "detected 8px grid" in result[1]
    px = np.array(Image.open(out))
    assert px[0, 0, 3] == 0  # background removed
    assert tuple(px[8, 8, :3]) == tuple(int(v) for v in small[8, 8]) or np.abs(px[8, 8, :3].astype(int) - small[8, 8]).max() < 12
    assert len(np.unique(px[px[..., 3] > 0][:, :3], axis=0)) <= 4


def test_frames_to_sprite_shared_palette(base_dir):
    paths = []
    for i in range(3):
        p = base_dir + f"/frame_{i}.png"
        small = np.zeros((32, 32, 4), np.uint8)
        small[8 + i:24 + i, 10:22] = (90, 160, 60, 255)
        Image.fromarray(np.repeat(np.repeat(small, 4, 0), 4, 1)).save(p)
        paths.append(p)
    out = base_dir + "/frames.ase"
    result = run(pipeline.frames_to_sprite(out, paths, width=32, height=32, palette="auto:8",
                                           tags=[{"name": "bob", "from": 1, "to": 3}]))
    assert "3 frames" in result[1], result
    spr = ase.read(out)
    assert spr.frame_count == 3 and spr.tags[0].name == "bob"
    assert render.composite(spr, 2)[10, 12, 3] == 255 and render.composite(spr, 0)[8, 12, 3] == 255


def test_import_spritesheet_image(base_dir):
    sheet = np.zeros((8, 24, 4), np.uint8)
    for i in range(3):
        sheet[2:6, i * 8 + 2:i * 8 + 6] = (255, 0, i * 100, 255)
    path = base_dir + "/grid.png"
    Image.fromarray(sheet).save(path)
    out = base_dir + "/grid.ase"
    ok(run(pipeline.import_spritesheet_image(path, out, 8, 8)))
    spr = ase.read(out)
    assert (spr.frame_count, spr.width) == (3, 8)
    assert tuple(render.composite(spr, 2)[3, 3]) == (255, 0, 200, 255)
