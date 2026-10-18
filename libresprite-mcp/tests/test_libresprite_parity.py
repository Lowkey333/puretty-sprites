"""Checks against the real LibreSprite (skipped when it isn't installed).

These are the tests that matter most for the port: files written here
must load in LibreSprite exactly as intended, and the renderer must match
LibreSprite's own output pixel for pixel.
"""
import os
import subprocess

import numpy as np
import pytest
from PIL import Image

from conftest import ok, run
from libresprite_mcp.core import ase, libresprite, render
from libresprite_mcp.tools import animation, canvas, drawing, layers
from libresprite_mcp.tools import libresprite as ls_tools


def _random_sprite(rng, mode, w=12, h=10):
    ch = ase.CHANNELS[mode]
    spr = ase.Sprite(w, h, mode, palette=[(0, 0, 0, 255)], durations=[100, 140])
    for li in range(4):
        layer = ase.Layer(f"l{li}", blend_mode=int(rng.integers(0, 16)), opacity=int(rng.integers(0, 256)))
        for f in range(2):
            img = rng.integers(0, 256, (int(rng.integers(1, h + 3)), int(rng.integers(1, w + 3)), ch), dtype=np.uint8)
            a = img[..., -1]
            a[rng.random(a.shape) < 0.3] = 0
            a[rng.random(a.shape) < 0.3] = 255
            layer.cels[f] = ase.Cel(img, int(rng.integers(-3, w)), int(rng.integers(-3, h)), int(rng.integers(0, 256)))
        spr.layers.append(layer)
    return spr


@pytest.mark.parametrize("mode", [ase.RGB, ase.GRAY])
def test_render_matches_libresprite(libresprite_exe, tmp_path, mode):
    rng = np.random.default_rng(42 if mode == ase.RGB else 43)
    for trial in range(6):
        spr = _random_sprite(rng, mode)
        path, sheet = str(tmp_path / f"r{trial}.ase"), str(tmp_path / f"r{trial}.png")
        ase.write(spr, path)
        subprocess.run([libresprite_exe, "-b", path, "--sheet", sheet], capture_output=True, timeout=60)
        ref = np.array(Image.open(sheet).convert("RGBA")).astype(int)
        for f in range(2):
            ours = render.composite(spr, f).astype(int)
            theirs = ref[:, f * spr.width:(f + 1) * spr.width]
            clear = (ours[..., 3] == 0) & (theirs[..., 3] == 0)
            assert not (np.any(ours != theirs, axis=-1) & ~clear).any(), f"trial {trial} frame {f}"


def test_tool_made_sprite_is_compatible(libresprite_exe, tmp_path):
    path = str(tmp_path / "hero.ase")
    ok(run(canvas.create_canvas(16, 16, path)))
    ok(run(canvas.add_layer(path, "body")))
    ok(run(drawing.draw_ellipse_at(path, "body", 1, 8, 8, 5, 6, "#AC3232", True)))
    ok(run(animation.add_frames(path, 2, 120, copy_previous=True)))
    ok(run(animation.tween_cel_positions(path, "body", 1, 3, 0, 0, 0, -2)))
    ok(run(layers.set_layer_blend_mode(path, "body", "overlay")))
    ok(run(animation.set_tag(path, "bob", 1, 3, "pingpong")))
    report = run(ls_tools.check_libresprite_compat(path))
    assert report.startswith("OK"), report
    tags = libresprite.run(["--batch", "--list-tags", path])[1]
    assert "bob" in tags


def test_run_libresprite_script(libresprite_exe, tmp_path):
    path = str(tmp_path / "s.ase")
    ok(run(canvas.create_canvas(4, 4, path)))
    out = run(ls_tools.run_libresprite_script(
        'var s = app.activeSprite; console.log("W=" + s.width + " L=" + s.layer(0).name);', path))
    assert "W=4 L=Layer 1" in out
    assert run(ls_tools.run_libresprite_script("null.x();")).startswith("Script failed")


def test_run_libresprite_cli_exports_sheet(libresprite_exe, tmp_path):
    path = str(tmp_path / "c.ase")
    ok(run(canvas.create_canvas(4, 4, path)))
    ok(run(animation.add_frames(path, 1)))
    sheet = str(tmp_path / "c.png")
    ok(run(ls_tools.run_libresprite_cli([path, "--sheet", sheet])))
    assert Image.open(sheet).size == (8, 4)
