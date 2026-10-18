"""auto_rig.py on synthetic A-pose figures (skipped without Blender).

One Blender run builds a plain figure and one in a robe out of primitives,
rigs them (and the plain one again with pose keypoints), and measures the
posed meshes: feet on the ground, steps alternating, the jump leaving the
floor, the death ending low and compact.
"""
import json
import os
import subprocess

import pytest
from mcp.server.mcpserver import Image

from conftest import ok, run
from libresprite_mcp.tools import blender

AUTO_RIG_DIR = os.path.dirname(blender.auto_rig_script())

pytestmark = pytest.mark.skipif(blender.find_blender() is None, reason="Blender not found (set BLENDER_PATH)")

SCRIPT = r'''
import json, math, os, sys
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0, sys.argv[sys.argv.index("--") + 1])
import auto_rig

out = sys.argv[sys.argv.index("--") + 2]


def cylinder(a, b, r):
    a, b = Vector(a), Vector(b)
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=(b - a).length, location=(a + b) / 2, vertices=16)
    ob = bpy.context.active_object
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(b - a)


def box(location, size):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    bpy.context.active_object.scale = size


def sphere(location, r):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=location, segments=24, ring_count=12)


def figure(path, robe, hand=(0.36, 0.56)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for s in (1, -1):
        cylinder((0.08 * s, 0, 0.05), (0.07 * s, 0, 0.5), 0.04)    # leg
        box((0.08 * s, -0.04, 0.025), (0.06, 0.14, 0.05))           # foot, toes forward (-Y)
        cylinder((0.13 * s, 0, 0.78), (hand[0] * s, 0, hand[1]), 0.03)  # arm, 45 degrees down by default
        sphere(((hand[0] + 0.02) * s, 0, hand[1] - 0.02 * (hand[1] < 0.75)), 0.04)  # hand
    box((0, 0, 0.65), (0.26, 0.14, 0.32))                           # torso, z 0.49-0.81
    cylinder((0, 0, 0.8), (0, 0, 0.86), 0.035)                      # neck
    sphere((0, 0, 0.92), 0.08)                                      # head, top at 1.0
    if robe:
        bpy.ops.mesh.primitive_cone_add(radius1=0.2, radius2=0.12, depth=0.4, location=(0, 0, 0.32), vertices=24)
    # One dense surface, like image-to-3D output (primitives only have
    # vertices at their ends).
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    remesh = bpy.context.active_object.modifiers.new("Remesh", "REMESH")
    remesh.mode, remesh.voxel_size = "VOXEL", 0.008
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB")


def posed(action, frame):
    scene = bpy.context.scene
    rig, body = bpy.data.objects["CharacterRig"], bpy.data.objects["Character"]
    ad = rig.animation_data
    ad.action = bpy.data.actions[action]
    if getattr(ad, "action_slot", None) is None and len(getattr(ad, "action_suitable_slots", [])):
        ad.action_slot = ad.action_suitable_slots[0]
    scene.frame_set(frame)
    ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = ev.to_mesh()
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    m = np.array(body.matrix_world)
    return co @ m[:3, :3].T + m[:3, 3]


def toe_y(co, side):
    feet = co[(co[:, 2] < 0.08) & (side * co[:, 0] > 0.02)]
    return float(feet[:, 1].min())


def inspect(result):
    rig = bpy.data.objects["CharacterRig"]
    body = bpy.data.objects["Character"]
    info = {"result": result, "bones": sorted(b.name for b in rig.data.bones)}
    groups = {g.index: g.name for g in body.vertex_groups}
    owned = {}
    for v in body.data.vertices:
        for g in v.groups:
            if g.weight > 0.5:
                owned[groups[g.group]] = owned.get(groups[g.group], 0) + 1
    info["owned"] = owned
    info["actions"] = {a.name: {"range": list(a.frame_range), "cyclic": a.use_cyclic} for a in bpy.data.actions}
    info["walk_lowest"] = [float(posed("Walk", f)[:, 2].min()) for f in (1, 5, 9, 13)]
    first, half = posed("Walk", 1), posed("Walk", 9)
    info["walk_toes"] = [[toe_y(first, 1), toe_y(first, -1)], [toe_y(half, 1), toe_y(half, -1)]]
    info["jump_apex_lowest"] = float(posed("Jump", 9)[:, 2].min())
    rest_top = float(posed("Idle", 1)[:, 2].max())
    dead = posed("Death", 18)
    info["death"] = {"top": float(dead[:, 2].max()), "lowest": float(dead[:, 2].min()), "rest_top": rest_top,
                     "reach": float(np.abs(dead[:, :2]).max())}
    return info


report = {}
plain, robed = os.path.join(out, "plain.glb"), os.path.join(out, "robed.glb")
figure(plain, robe=False)
figure(robed, robe=True)
report["plain"] = inspect(auto_rig.rig_character(plain, os.path.join(out, "plain.blend")))
report["robed"] = {"result": auto_rig.rig_character(robed, os.path.join(out, "robed.blend"), animations=("Idle",))}
tpose = os.path.join(out, "tpose.glb")
figure(tpose, robe=False, hand=(0.45, 0.78))
report["tpose"] = {"result": auto_rig.rig_character(tpose, os.path.join(out, "tpose.blend"), animations=("Idle",))}

# Pose keypoints, written the way a DWPose JSON for a front render arrives.
mapping = {"center": [0.0, 0.5], "pixel": 0.002, "size": [600, 600]}
truth = {"nose": (0, 0.92), "neck": (0, 0.79), "Rsho": (-0.13, 0.78), "Lsho": (0.13, 0.78),
         "Relb": (-0.24, 0.675), "Lelb": (0.24, 0.675), "Rwri": (-0.33, 0.59), "Lwri": (0.33, 0.59),
         "Rhip": (-0.075, 0.5), "Lhip": (0.075, 0.5), "Rknee": (-0.075, 0.29), "Lknee": (0.075, 0.29),
         "Rank": (-0.08, 0.06), "Lank": (0.08, 0.06)}
flat = []
for name in auto_rig.POSE_NAMES:
    if name in truth:
        x, z = truth[name]
        flat += [300 + x / 0.002, 300 - (z - 0.5) / 0.002, 1.0]
    else:
        flat += [0.0, 0.0, 0.0]
pose = [{"people": [{"pose_keypoints_2d": flat}], "canvas_width": 600, "canvas_height": 600}]
points = auto_rig.keypoints_from_pose(pose, mapping)
report["keypoints"] = {k: [round(v, 4) for v in p] for k, p in points.items()}
report["pose_rig"] = {"result": auto_rig.rig_character(plain, os.path.join(out, "posed.blend"), keypoints=points,
                                                       animations=())}
print("TEST_RESULT " + json.dumps(report))
'''


@pytest.fixture(scope="module")
def rigs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("auto_rig")
    script = folder / "build_and_rig.py"
    script.write_text(SCRIPT, encoding="utf-8")
    proc = subprocess.run([blender.find_blender(), "-b", "--factory-startup", "--python-exit-code", "1",
                           "-P", str(script), "--", AUTO_RIG_DIR, str(folder)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("TEST_RESULT ")]
    assert lines, proc.stdout[-3000:] + proc.stderr[-2000:]
    return folder, json.loads(lines[-1][len("TEST_RESULT "):])


def test_joints_found_on_the_silhouette(rigs):
    _, report = rigs
    r = report["plain"]["result"]
    m = r["landmarks"]
    assert r["sources"] == {"legs": "silhouette", "arms": "silhouette", "spine": "silhouette"}
    assert not r["skirt"]
    assert 0.45 < m["hips"] < 0.6 and 0.2 < m["knee"] < 0.36 and 0.02 < m["ankle"] < 0.1
    assert 0.76 < m["neck"] < 0.88 and 0.7 < m["shoulder"] < 0.82
    assert 0.34 < m["hand_x"] < 0.44 and 0.48 < m["hand"] < 0.62
    assert 0.05 < m["leg_x"] < 0.11 and 0.08 < m["shoulder_x"] < 0.15
    toe = r["joints"]["toe.L"]
    assert toe[1] < r["joints"]["ankle.L"][1] - 0.03  # the foot points forward (-Y)


def test_skeleton_and_weights(rigs):
    folder, report = rigs
    info = report["plain"]
    expected = {"root", "hips", "chest", "head"} | {f"{p}.{s}" for s in "LR" for p in
                                                    ("thigh", "shin", "foot", "upper_arm", "forearm", "hand")}
    assert set(info["bones"]) == expected
    assert info["result"]["weights"].startswith("heat"), info["result"]["weights"]
    assert set(info["owned"]) == expected - {"root"}  # every deforming bone moves part of the mesh
    assert os.path.isfile(folder / "plain_skeleton.png")
    assert os.path.isfile(folder / "plain.blend")


def test_actions(rigs):
    _, report = rigs
    actions = report["plain"]["actions"]
    assert set(actions) == {"Idle", "Walk", "Run", "Attack", "Cast", "Hit", "Death", "Jump"}
    assert actions["Walk"] == {"range": [1.0, 17.0], "cyclic": True}  # frame 17 repeats frame 1
    assert actions["Idle"]["cyclic"] and actions["Run"]["cyclic"]
    assert actions["Attack"] == {"range": [1.0, 12.0], "cyclic": False}
    assert report["plain"]["result"]["loops"] == ["Idle", "Walk", "Run"]


def test_walk_keeps_feet_on_the_ground_and_alternates(rigs):
    _, report = rigs
    info = report["plain"]
    assert all(abs(z) < 0.012 for z in info["walk_lowest"]), info["walk_lowest"]
    (left_a, right_a), (left_b, right_b) = info["walk_toes"]
    assert left_a < right_a - 0.05  # frame 1: left foot forward
    assert right_b < left_b - 0.05  # half a cycle later: right foot forward


def test_jump_and_death(rigs):
    _, report = rigs
    info = report["plain"]
    assert info["jump_apex_lowest"] > 0.04
    death = info["death"]
    assert death["top"] < 0.75 * death["rest_top"]  # collapsed
    assert abs(death["lowest"]) < 0.02  # resting on the floor, not floating or sunk
    assert death["reach"] < 0.6  # stays inside the sprite frame


def test_robe_detected(rigs):
    _, report = rigs
    r = report["robed"]["result"]
    assert r["skirt"] and r["actions"] == ["Idle"]
    assert any("robe or skirt" in n for n in r["notes"])
    assert 0.12 < r["landmarks"]["hips"] < 0.6


def test_t_pose_shoulders_follow_the_arm(rigs):
    _, report = rigs
    m = report["tpose"]["result"]["landmarks"]
    assert m["shoulder"] == pytest.approx(0.78, abs=0.025)  # on the arm's centre line, not above it
    assert m["hand"] == pytest.approx(0.78, abs=0.03) and 0.44 < m["hand_x"] < 0.54


def test_pose_keypoints_used(rigs):
    _, report = rigs
    kp = report["keypoints"]
    assert kp["Lsho"][:2] == pytest.approx([0.13, 0.78], abs=1e-3)
    assert kp["Rank"][:2] == pytest.approx([-0.08, 0.06], abs=1e-3)
    assert "nose" in kp and "Reye" not in kp
    r = report["pose_rig"]["result"]
    assert r["sources"]["legs"] == "pose detector" and r["sources"]["arms"] == "pose detector"
    m = r["landmarks"]
    assert m["hips"] == pytest.approx(0.5, abs=0.02) and m["knee"] == pytest.approx(0.29, abs=0.02)
    assert m["shoulder"] == pytest.approx(0.78, abs=0.02) and m["shoulder_x"] == pytest.approx(0.13, abs=0.02)
    assert r["actions"] == []


def test_tool_without_pose_detection(rigs, tmp_path):
    folder, _ = rigs
    out = tmp_path / "hero.blend"
    result = run(blender.blender_auto_rig(str(folder / "plain.glb"), str(out), animations="Idle,walk",
                                          pose_detection="off", target_faces=0))
    text = ok(result[0])
    assert text.startswith("Rigged plain.glb"), text
    assert "weights: heat" in text and "loops Idle, Walk" in text and "one-shots" not in text
    assert "legs from the silhouette" in text
    assert isinstance(result[1], Image)
    assert out.is_file() and (tmp_path / "hero_rig" / "skeleton.png").is_file()


def test_tool_rejects_bad_arguments(tmp_path):
    model = tmp_path / "m.glb"
    model.write_bytes(b"glTF")
    assert "unknown animation Dance" in run(blender.blender_auto_rig(str(model), str(tmp_path / "o.blend"),
                                                                    animations="Idle,Dance"))[0]
    assert "pose_detection must be" in run(blender.blender_auto_rig(str(model), str(tmp_path / "o.blend"),
                                                                   pose_detection="maybe"))[0]
    assert "not found" in run(blender.blender_auto_rig(str(tmp_path / "none.glb"), str(tmp_path / "o.blend")))[0]
