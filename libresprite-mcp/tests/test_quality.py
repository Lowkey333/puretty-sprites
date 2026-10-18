"""Animation consistency tools (quality.py)."""
import json
import os

import numpy as np

from conftest import fails, ok, run
from libresprite_mcp.core import ase
from libresprite_mcp.tools import quality


def _scene(base_dir, name="quality.ase"):
    """4 frames, 16x16: a full-canvas 'bg', a 'ball' moving right, a 'hand' on frames 1-2."""
    path = os.path.join(base_dir, name)
    spr = ase.Sprite(16, 16, ase.RGB, palette=[(0, 0, 0, 255)], durations=[100] * 4)
    bg = ase.Layer("bg")
    ball = ase.Layer("ball")
    hand = ase.Layer("hand")
    for f in range(4):
        bg.cels[f] = ase.Cel(np.full((16, 16, 4), (20, 20, 40, 255), np.uint8))
        img = np.zeros((16, 16, 4), np.uint8)  # canvas-sized, like the drawing tools leave it
        img[6:9, 2 + 3 * f:5 + 3 * f] = (255, 255, 255, 255)
        ball.cels[f] = ase.Cel(img)
    for f in (0, 1):
        img = np.zeros((16, 16, 4), np.uint8)
        img[6:9, 6:9] = (255, 200, 150, 255)
        hand.cels[f] = ase.Cel(img)
    spr.layers = [bg, ball, hand]
    ase.write(spr, path)
    return path


def test_ensure_layers_present_and_validate(base_dir):
    path = _scene(base_dir, "ensure.ase")
    report = json.loads(ok(run(quality.validate_scene(path, ["hand", "cape"]))))
    assert report == {"frames": 4, "range": {"start": 1, "end": 4}, "missing_layers": ["cape"],
                      "missing_cels": [{"layer": "hand", "frame": 3}, {"layer": "hand", "frame": 4}]}
    msg = ok(run(quality.ensure_layers_present(path, ["hand"], 2, 3)))
    assert "(1 created)" in msg
    hand = next(l for l in ase.read(path).layers if l.name == "hand")
    assert sorted(hand.cels) == [0, 1, 2]
    assert hand.cels[2].image.shape == (16, 16, 4) and not hand.cels[2].image.any()
    report = json.loads(run(quality.validate_scene(path, ["hand"], 1, 3)))
    assert report["missing_cels"] == []


def test_ensure_and_validate_errors(base_dir):
    path = _scene(base_dir, "ensure_err.ase")
    fails(run(quality.ensure_layers_present(path, ["nope"])), "no layers found")
    fails(run(quality.ensure_layers_present(path, ["hand"], 3, 9)), "out of bounds")
    assert run(quality.validate_scene(path, [])) == "Required layers list cannot be empty"


def test_audit_reports_painted_overlaps_and_out_of_range(base_dir):
    path = _scene(base_dir, "audit.ase")
    report = json.loads(ok(run(quality.audit_animation(
        path, overlap_pairs=["ball,hand"], layer_frame_ranges=["hand:1-1"], report_bounds=True))))
    # The ball covers x 2-4 on frame 1 and 5-7 on frame 2; the hand covers x 6-8.
    assert report["overlaps"] == [{"frame": 2, "a": "ball", "b": "hand",
                                   "a_bounds": [5, 6, 3, 3], "b_bounds": [6, 6, 3, 3]}]
    assert report["out_of_range"] == [{"frame": 2, "layer": "hand"}]
    assert report["summary"] == {"total_layers": 3, "layers_checked": 3, "total_cels": 10, "overlaps": 1,
                                 "overlaps_total": 1, "overlaps_truncated": False, "out_of_range": 1}
    assert report["frames"] == {"start": 1, "end": 4}
    assert "cels" not in report


def test_audit_cel_report_and_limits(base_dir):
    path = _scene(base_dir, "audit2.ase")
    report = json.loads(ok(run(quality.audit_animation(
        path, 1, 2, layer_names=["ball", "missing"], report_cels=True, report_bounds=True))))
    assert report["summary"]["layers_checked"] == 1
    assert report["cels"] == [{"frame": 1, "cels": [{"layer": "ball", "x": 2, "y": 6, "w": 3, "h": 3}]},
                              {"frame": 2, "cels": [{"layer": "ball", "x": 5, "y": 6, "w": 3, "h": 3}]}]
    report = json.loads(ok(run(quality.audit_animation(path, overlap_pairs=["bg:ball"], max_overlaps=1))))
    assert len(report["overlaps"]) == 1 and report["summary"]["overlaps_total"] == 4
    assert report["summary"]["overlaps_truncated"] is True
    assert run(quality.audit_animation(path, 3, 2)) == "End frame must be >= start frame"
    assert run(quality.audit_animation(path, max_overlaps=-1)) == "Max limits must be >= 0"


def test_sanitize_report_only_changes_nothing(base_dir):
    path = _scene(base_dir, "sanitize_ro.ase")
    before = open(path, "rb").read()
    report = json.loads(ok(run(quality.animation_sanitize(
        path, layer_order=["hand", "ball"], layer_frame_ranges=["hand:1-1"], ensure_layers=["hand"],
        report_only=True))))
    assert open(path, "rb").read() == before
    # ensured cels are only counted, not created, so they are not out of range either
    assert report["reordered"] is True and report["ensured"] == 2 and report["out_of_range"] == 1
    assert report["opacity_set"] == 0 and report["deleted"] == 0


def test_sanitize_applies_fixes(base_dir):
    path = _scene(base_dir, "sanitize.ase")
    report = json.loads(ok(run(quality.animation_sanitize(
        path, layer_order=["hand", "ball"], layer_frame_ranges=["hand:1-1"],
        overlap_pairs=["ball,hand", "bg,ball"], report_bounds=True))))
    spr = ase.read(path)
    assert [l.name for l in spr.layers] == ["hand", "ball", "bg"]
    hand = spr.layers[0]
    assert hand.cels[0].opacity == 255 and hand.cels[1].opacity == 0
    assert report["out_of_range"] == 1 and report["opacity_set"] == 1
    assert report["alerts"] == ["cels_out_of_range", "full_canvas_cels:bg"]
    # bg covers the whole canvas, so its overlaps are skipped by default
    assert report["overlap_samples"] == [{"frame": 2, "a": "ball", "b": "hand",
                                          "a_bounds": [5, 6, 3, 3], "b_bounds": [6, 6, 3, 3]}]
    assert report["layer_stats"]["ball"] == {"frames_active": 4, "cel_count": 4, "full_canvas_cels": 0,
                                             "bounds": [2, 6, 14, 9]}
    assert report["analysis"]["inactive_layers"] == [] and report["analysis"]["empty_frames"] == 0


def test_sanitize_deletes_and_unlinks(base_dir):
    path = _scene(base_dir, "sanitize_del.ase")
    spr = ase.read(path)
    hand = next(l for l in spr.layers if l.name == "hand")
    hand.cels[1] = hand.cels[0]  # frames 1-2 share one linked cel
    ase.write(spr, path)
    ok(run(quality.animation_sanitize(path, layer_frame_ranges=["hand:1-1"])))
    hand = next(l for l in ase.read(path).layers if l.name == "hand")
    assert hand.cels[0].opacity == 255, "hiding frame 2 must not hide its linked frame 1"
    assert hand.cels[1].opacity == 0
    ok(run(quality.animation_sanitize(path, layer_frame_ranges=["hand:1-1"], out_of_range_action="delete_cels")))
    assert sorted(next(l for l in ase.read(path).layers if l.name == "hand").cels) == [0]


def test_sanitize_keeps_background_at_bottom_and_validates(base_dir):
    path = _scene(base_dir, "sanitize_bg.ase")
    spr = ase.read(path)
    spr.layers[0].is_background = True
    ase.write(spr, path)
    ok(run(quality.animation_sanitize(path, layer_order=["hand", "bg"])))
    assert [l.name for l in ase.read(path).layers] == ["bg", "hand", "ball"]
    assert run(quality.animation_sanitize(path, out_of_range_action="explode")) == "Unsupported out_of_range_action"
    assert run(quality.animation_sanitize(path, out_of_range_opacity=300)) == "out_of_range_opacity must be 0-255"
    fails(run(quality.animation_sanitize(path, 1, 40)), "out of bounds")
