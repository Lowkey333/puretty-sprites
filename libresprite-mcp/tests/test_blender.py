"""Blender sprite rendering against the example character (skipped without Blender)."""
import json
import os
import subprocess

import numpy as np
import pytest
from PIL import Image

from conftest import ok, run
from libresprite_mcp.tools import blender

EXAMPLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "integrations", "blender",
                                       "examples", "test_character.blend"))
MATERIAL_COLORS = {"#F1C27D", "#3B5DC9", "#333C57", "#5D3A1A", "#B13E53", "#1A1C2C"}

pytestmark = pytest.mark.skipif(blender.find_blender() is None, reason="Blender not found (set BLENDER_PATH)")


def alpha_of(path):
    return np.array(Image.open(path).convert("RGBA"))[..., 3]


@pytest.fixture(scope="module")
def walk(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("blender") / "walk")
    text = ok(run(blender.blender_render_sprites(EXAMPLE, out, directions=8, frame_size=32,
                                                 animation="Walk", frame_step=6, supersample=2)))
    with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
        return text, json.load(f)


def test_status():
    text = run(blender.blender_status())
    assert "Version: 5." in text or "Version: 4." in text, text


def test_list_animations():
    data = json.loads(run(blender.blender_list_animations(EXAMPLE)))
    actions = {a["name"]: a for a in data["animations"]}
    assert {"Idle", "Walk"} <= set(actions)
    assert actions["Walk"]["cyclic"] and actions["Walk"]["active"]
    assert data["armature"] == "HeroRig"
    assert 1.7 < data["dimensions"][2] < 1.9


def test_walk_manifest(walk):
    text, m = walk
    assert m["directions"] == ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
    anim = m["animations"][0]
    assert anim["name"] == "Walk"
    assert anim["frames"] == [1, 7, 13, 19]  # the cyclic action's frame 25 repeats frame 1
    assert m["frame_count"] == 32 == len(m["sequence"])
    assert m["duration_ms"] == 500  # 12 fps, every 6th frame
    assert m["tags"][0] == {"name": "Walk_S", "from": 1, "to": 4, "direction": "forward"}
    assert m["tags"][-1] == {"name": "Walk_SE", "from": 29, "to": 32, "direction": "forward"}
    assert m["render_size"] == 64 and m["frame_size"] == 32
    assert "manifest.json" in text and "Walk_S 1-4" in text


def test_walk_frames_line_up(walk):
    _, m = walk
    pivot_row = m["pivot"]["y"] * m["supersample"]
    tops, bottoms = [], []
    for path in m["sequence"]:
        alpha = alpha_of(path)
        assert alpha.shape == (64, 64)
        assert alpha[0, 0] == 0 and alpha[-1, -1] == 0  # transparent background
        ys, xs = np.nonzero(alpha)
        assert len(ys) > 150  # the character is in the frame
        assert ys.min() > 0 and xs.min() > 0 and ys.max() < 63 and xs.max() < 63  # margin kept
    for d in m["directions"]:
        ys, _ = np.nonzero(alpha_of(m["animations"][0]["files"][d][0]))
        tops.append(ys.min())
        bottoms.append(ys.max())
    # One shared scale and ground line: the head and feet stay put across directions.
    assert max(tops) - min(tops) <= 6, tops
    assert all(pivot_row - 4 <= b <= pivot_row + 6 for b in bottoms), (pivot_row, bottoms)


def test_exact_size_flat_render(tmp_path):
    out = str(tmp_path / "flat")
    ok(run(blender.blender_render_sprites(EXAMPLE, out, directions=8, direction_names="S,E",
                                          frame_size=24, supersample=1, shading="flat",
                                          animation="static")))
    m = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert m["directions"] == ["S", "E"] and m["frame_count"] == 2
    for path in m["sequence"]:
        rgba = np.array(Image.open(path).convert("RGBA")).reshape(-1, 4)
        assert set(np.unique(rgba[:, 3]).tolist()) <= {0, 255}  # hard edges
        opaque = {"#%02X%02X%02X" % tuple(c[:3]) for c in np.unique(rgba[rgba[:, 3] > 0], axis=0)}
        assert opaque <= MATERIAL_COLORS, opaque  # unlit base colours, no dither noise


def test_gltf_import(tmp_path):
    glb = str(tmp_path / "hero.glb")
    subprocess.run([blender.find_blender(), "-b", "--factory-startup", EXAMPLE, "--python-expr",
                    f"import bpy; bpy.ops.export_scene.gltf(filepath=r'{glb}', export_animations=True)"],
                   capture_output=True, timeout=300)
    assert os.path.isfile(glb)
    data = json.loads(run(blender.blender_list_animations(glb)))
    assert {"Idle", "Walk"} <= {a["name"] for a in data["animations"]}
    assert data["dimensions"][2] < 2.0  # the importer's hidden bone-shape mesh is not framed


def test_errors(tmp_path):
    assert "not found" in run(blender.blender_render_sprites(str(tmp_path / "nope.glb"), str(tmp_path)))
    assert "directions must be" in run(blender.blender_render_sprites(EXAMPLE, str(tmp_path), directions=3))
    assert "Unsupported model format" in run(blender.blender_list_animations(__file__))
    text = run(blender.blender_render_sprites(EXAMPLE, str(tmp_path / "x"), animation="Dance"))
    assert text.startswith("Failed to render sprites") and "Dance" in text and "Walk" in text
