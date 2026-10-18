"""Single-cel and composite (flattened) pixel reads (pixel_read.py).

The composite tests prove get_composite_* reads the VISIBLE result (top
layer wins, opacity applied, hidden layers ignored) while get_pixel_color
reads one named cel.
"""
import json
import os

import numpy as np

from conftest import fails, ok, run
from libresprite_mcp.core import ase
from libresprite_mcp.tools import canvas, drawing, pixel_read


def test_reads_a_named_cel(sprite):
    # fixture: 32x32, 'body' layer with a red rect at (8,8,16,16).
    assert run(pixel_read.get_pixel_color(sprite, 12, 12, "body", 1)) == "#d04648 (r=208, g=70, b=72, a=255)"
    assert run(pixel_read.get_pixel_color(sprite, 1, 1, "body", 1)) == "#000000 (r=0, g=0, b=0, a=0)"


def test_empty_layer_name_reads_the_active_layer(sprite):
    # Nothing was selected with set_layer, so the topmost layer ('body') is active.
    assert run(pixel_read.get_pixel_color(sprite, 12, 12)).startswith("#d04648")
    ok(run(canvas.set_layer(sprite, "Layer 1")))
    assert "a=0" in run(pixel_read.get_pixel_color(sprite, 12, 12))
    ok(run(canvas.set_layer(sprite, "body")))


def test_rect_is_row_major_with_sprite_coordinates(sprite):
    pixels = json.loads(ok(run(pixel_read.get_pixels_rect(sprite, 7, 7, 2, 2, "body", 1))))
    assert [(p["x"], p["y"]) for p in pixels] == [(7, 7), (8, 7), (7, 8), (8, 8)]
    assert [p["a"] for p in pixels] == [0, 0, 0, 255]
    assert pixels[3]["hex"] == "#d04648" and pixels[3]["r"] == 208


def test_rect_size_is_capped(sprite):
    fails(run(pixel_read.get_pixels_rect(sprite, 0, 0, 100, 100, "body", 1)), "at most 4096")
    fails(run(pixel_read.get_composite_rect(sprite, 0, 0, 0, 5, 1)), "must be > 0")


def test_composite_reads_top_layer(sprite):
    ok(run(canvas.add_layer(sprite, "overlay")))
    ok(run(drawing.draw_rectangle_at(sprite, "overlay", 1, 8, 8, 16, 16, "#3050D0", True)))
    body = run(pixel_read.get_pixel_color(sprite, 12, 12, "body", 1))
    assert body.lower().startswith("#d04648"), body
    comp = run(pixel_read.get_composite_pixel(sprite, 12, 12, 1))
    assert comp.lower().startswith("#3050d0"), comp


def test_composite_applies_opacity_and_skips_hidden_layers(sprite):
    spr = ase.read(sprite)
    overlay = next(l for l in spr.layers if l.name == "overlay")
    overlay.opacity = 128
    ase.write(spr, sprite)
    r, g, b, a = (int(s.split("=")[1]) for s in
                  run(pixel_read.get_composite_pixel(sprite, 12, 12, 1)).strip(")").split("(")[1].split(", "))
    # rgba_blender_normal: blue at opacity 128 over opaque red (C division truncates toward 0).
    mix = lambda bg, fg: bg + int((fg - bg) * 128 / 255)
    assert (r, g, b, a) == (mix(0xD0, 0x30), mix(0x46, 0x50), mix(0x48, 0xD0), 255) == (0x80, 0x4B, 0x8C, 255)
    overlay.visible = False
    ase.write(spr, sprite)
    assert run(pixel_read.get_composite_pixel(sprite, 12, 12, 1)).startswith("#d04648")


def test_composite_rect_shape(sprite):
    px = json.loads(run(pixel_read.get_composite_rect(sprite, 8, 8, 4, 4, 1)))
    assert len(px) == 16
    assert all("hex" in p and "a" in p for p in px)
    assert {p["hex"] for p in px} == {"#d04648"}


def test_composite_pixel_out_of_bounds_is_transparent(sprite):
    assert "a=0" in run(pixel_read.get_composite_pixel(sprite, 1, 1, 1))
    assert "a=0" in run(pixel_read.get_composite_pixel(sprite, 500, -3, 1))


def test_indexed_sprites_report_color_and_index(base_dir):
    path = os.path.join(base_dir, "indexed_read.ase")
    spr = ase.Sprite(4, 4, ase.INDEXED, palette=[(0, 0, 0, 255), (255, 0, 0, 255), (0, 0, 255, 255)])
    layer = ase.Layer("art")
    img = np.zeros((4, 4, 1), np.uint8)
    img[1, 2] = 2
    layer.cels[0] = ase.Cel(img)
    spr.layers.append(layer)
    ase.write(spr, path)
    assert run(pixel_read.get_pixel_color(path, 2, 1, "art", 1)) == "#0000ff (r=0, g=0, b=255, a=255, index=2)"
    # index 0 is the transparent index
    assert run(pixel_read.get_pixel_color(path, 0, 0, "art", 1)) == "#000000 (r=0, g=0, b=0, a=0, index=0)"
    assert json.loads(run(pixel_read.get_pixels_rect(path, 2, 1, 1, 1, "art", 1)))[0]["index"] == 2


def test_errors(sprite):
    fails(run(pixel_read.get_pixel_color(sprite, 0, 0, "nope", 1)), "layer not found")
    fails(run(pixel_read.get_pixel_color(sprite, 0, 0, "body", 7)), "frame index out of range")
    ok(run(canvas.add_frame(sprite)))
    fails(run(pixel_read.get_pixel_color(sprite, 0, 0, "body", 2)), "no cel")
    fails(run(pixel_read.get_composite_pixel(sprite + ".missing", 0, 0, 1)), "not found")
