"""Mocap retargeting onto the auto-rig skeleton, with synthetic BVH clips (skipped without Blender).

The clips are written here with known motion: thighs swinging in opposite
phase with a 0.75 s period, a root that travels along a heading 40 degrees
off the -Y the rig faces, and a vertical bob. The tests retarget them with
blender_add_animation and read the posed bones back from Blender.
"""
import json
import math
import os
import shutil
import subprocess
import time

import numpy as np
import pytest

from conftest import fails, ok, run
from libresprite_mcp.tools import blender, blender_anim

pytestmark = pytest.mark.skipif(blender.find_blender() is None, reason="Blender not found (set BLENDER_PATH)")

PERIOD, HEADING, SPEED, BOB = 0.75, 40.0, 1.6, 0.02
TEST_CHARACTER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "integrations", "blender",
                                              "examples", "test_character.blend"))

# name -> (parent, offset); Y up, facing +Z like CMU and Mixamo BVH files.
SKELETON = {
    "hips": (None, (0, 0, 0)),
    "upleg.L": ("hips", (0.09, -0.05, 0)), "leg.L": ("upleg.L", (0, -0.42, 0)),
    "foot.L": ("leg.L", (0, -0.42, 0)), "toe.L": ("foot.L", (0, -0.07, 0.12)),
    "upleg.R": ("hips", (-0.09, -0.05, 0)), "leg.R": ("upleg.R", (0, -0.42, 0)),
    "foot.R": ("leg.R", (0, -0.42, 0)), "toe.R": ("foot.R", (0, -0.07, 0.12)),
    "spine": ("hips", (0, 0.1, 0)), "spine1": ("spine", (0, 0.12, 0)), "spine2": ("spine1", (0, 0.12, 0)),
    "neck": ("spine2", (0, 0.14, 0)), "head": ("neck", (0, 0.1, 0)),
    "clav.L": ("spine2", (0.04, 0.1, 0)), "arm.L": ("clav.L", (0.12, 0, 0)),
    "forearm.L": ("arm.L", (0.27, 0, 0)), "hand.L": ("forearm.L", (0.24, 0, 0)),
    "clav.R": ("spine2", (-0.04, 0.1, 0)), "arm.R": ("clav.R", (-0.12, 0, 0)),
    "forearm.R": ("arm.R", (-0.27, 0, 0)), "hand.R": ("forearm.R", (-0.24, 0, 0)),
}
# The CMU "cgspeed" BVH layout: zero-length hip and lower-back joints, two neck joints.
CMU_SKELETON = {
    **{k: v for k, v in SKELETON.items() if k not in ("spine1", "spine2", "neck", "clav.L", "clav.R")},
    "hipjoint.L": ("hips", (0, 0, 0)), "upleg.L": ("hipjoint.L", (0.09, -0.05, 0)),
    "hipjoint.R": ("hips", (0, 0, 0)), "upleg.R": ("hipjoint.R", (-0.09, -0.05, 0)),
    "lowerback": ("hips", (0, 0, 0)), "spine": ("lowerback", (0, 0.1, 0)), "spine1": ("spine", (0, 0.24, 0)),
    "neck": ("spine1", (0, 0, 0)), "neck1": ("neck", (0, 0.14, 0)), "head": ("neck1", (0, 0.1, 0)),
    "clav.L": ("spine1", (0, 0, 0)), "arm.L": ("clav.L", (0.16, 0.1, 0)),
    "clav.R": ("spine1", (0, 0, 0)), "arm.R": ("clav.R", (-0.16, 0.1, 0)),
}
END_SITES = {"toe.L": (0, 0, 0.06), "toe.R": (0, 0, 0.06), "head": (0, 0.22, 0),
             "hand.L": (0.08, 0, 0), "hand.R": (-0.08, 0, 0)}
LIMBS = (("upleg", "UpLeg"), ("leg", "Leg"), ("foot", "Foot"), ("toe", "ToeBase"), ("clav", "Shoulder"),
         ("arm", "Arm"), ("forearm", "ForeArm"), ("hand", "Hand"))
NAMES = {
    "mixamo": {"hips": "Hips", "spine": "Spine", "spine1": "Spine1", "spine2": "Spine2", "neck": "Neck",
               "head": "Head", **{f"{k}.{s}": f"{side}{v}" for s, side in (("L", "Left"), ("R", "Right"))
                                  for k, v in LIMBS}},
    "cmu_bvh": {"hips": "Hips", "lowerback": "LowerBack", "spine": "Spine", "spine1": "Spine1", "neck": "Neck",
                "neck1": "Neck1", "head": "Head", "hipjoint.L": "LHipJoint", "hipjoint.R": "RHipJoint",
                **{f"{k}.{s}": f"{side}{v}" for s, side in (("L", "Left"), ("R", "Right")) for k, v in LIMBS}},
    "cmu_asf": {"hips": "root", "spine": "lowerback", "spine1": "upperback", "spine2": "thorax",
                "neck": "lowerneck", "head": "head", **{f"{k}.{s}": f"{s.lower()}{v}" for s in "LR"
                                                        for k, v in (("upleg", "femur"), ("leg", "tibia"),
                                                                     ("foot", "foot"), ("toe", "toes"),
                                                                     ("clav", "clavicle"), ("arm", "humerus"),
                                                                     ("forearm", "radius"), ("hand", "wrist"))}},
    "unreal": {"hips": "pelvis", "spine": "spine_01", "spine1": "spine_02", "spine2": "spine_03",
               "neck": "neck_01", "head": "head", **{f"{k}.{s}": f"{v}_{s.lower()}" for s in "LR"
                                                     for k, v in (("upleg", "thigh"), ("leg", "calf"),
                                                                  ("foot", "foot"), ("toe", "ball"),
                                                                  ("clav", "clavicle"), ("arm", "upperarm"),
                                                                  ("forearm", "lowerarm"), ("hand", "hand"))}},
}


def write_bvh(path, naming="mixamo", prefix="", seconds=3.0, rate=60, tpose_first=False):
    """A walk-like clip: thighs +/-30 deg in opposite phase, root travelling at SPEED along HEADING.

    With tpose_first the clip starts with an extra T-pose frame, like the CMU BVH conversion.
    """
    skeleton = CMU_SKELETON if naming == "cmu_bvh" else SKELETON
    names = {k: prefix + v for k, v in NAMES[naming].items()}
    children = {}
    for joint, (parent, _) in skeleton.items():
        children.setdefault(parent, []).append(joint)
    order, lines = [], ["HIERARCHY"]

    def emit(joint, depth):
        pad = "  " * depth
        order.append(joint)
        kind = "ROOT" if skeleton[joint][0] is None else "JOINT"
        channels = ("6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" if kind == "ROOT"
                    else "3 Zrotation Xrotation Yrotation")
        lines.extend([f"{pad}{kind} {names[joint]}", pad + "{",
                      f"{pad}  OFFSET {' '.join(map(str, skeleton[joint][1]))}", f"{pad}  CHANNELS {channels}"])
        for child in children.get(joint, []):
            emit(child, depth + 1)
        if joint in END_SITES:
            lines.extend([f"{pad}  End Site", pad + "  {", f"{pad}    OFFSET {' '.join(map(str, END_SITES[joint]))}",
                          pad + "  }"])
        lines.append(pad + "}")

    emit("hips", 0)
    count = int(seconds * rate) + 1
    frames = []
    heading = math.radians(HEADING)
    height = 0.05 + 0.42 + 0.42 + 0.07
    for k in range(-1 if tpose_first else 0, count):
        t = max(k, 0) / rate
        phase = 2 * math.pi * t / PERIOD
        swing = 30 * math.sin(phase)
        values = {j: [0.0, 0.0, 0.0] for j in order}  # Z, X, Y rotations in degrees
        if k >= 0:
            values["upleg.L"][1], values["upleg.R"][1] = -swing, swing
            values["leg.L"][1] = 20 * (1 + math.cos(phase))
            values["leg.R"][1] = 20 * (1 - math.cos(phase))
            values["arm.L"][0], values["arm.R"][0] = -75 + 8 * math.sin(phase), 75 + 8 * math.sin(phase)
            values["hips"][2] = HEADING
        travel = SPEED * t
        root = [math.sin(heading) * travel, height + BOB * math.cos(2 * phase), math.cos(heading) * travel]
        row = root + values["hips"]
        for j in order[1:]:
            row += values[j]
        frames.append(" ".join(f"{v:.5f}" for v in row))
    lines += ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {1 / rate:.7f}", *frames]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


RIG_SCRIPT = r'''
import bpy, sys
from mathutils import Vector
out, feet, mode = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
data = bpy.data.armatures.new("CharacterRig")
rig = bpy.data.objects.new("CharacterRig", data)
bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = data.edit_bones
def bone(name, head, tail, parent=None, connect=False):
    b = eb.new(name); b.head, b.tail = Vector(head), Vector(tail); b.align_roll(Vector((0, -1, 0)))
    if parent: b.parent = eb[parent]; b.use_connect = connect
bone("root", (0, 0, 0), (0, 0.1, 0)); eb["root"].align_roll(Vector((0, 0, 1)))
bone("hips", (0, 0, 0.22), (0, 0, 0.28), "root")
bone("chest", (0, 0, 0.28), (0, 0, 0.43), "hips", True)
bone("head", (0, 0, 0.43), (0, 0, 1.0), "chest", True)
for s, x in (("L", 1), ("R", -1)):
    bone(f"thigh.{s}", (0.06 * x, 0, 0.22), (0.06 * x, 0, 0.11), "hips")
    bone(f"shin.{s}", (0.06 * x, 0, 0.11), (0.06 * x, 0, 0.03), f"thigh.{s}", True)
    if feet == "1":
        bone(f"foot.{s}", (0.06 * x, 0, 0.03), (0.06 * x, -0.07, 0.0), f"shin.{s}", True)
    bone(f"upper_arm.{s}", (0.09 * x, 0, 0.40), (0.165 * x, 0, 0.355), "chest")
    bone(f"forearm.{s}", (0.165 * x, 0, 0.355), (0.24 * x, 0, 0.31), f"upper_arm.{s}", True)
bpy.ops.object.mode_set(mode="OBJECT")
for pb in rig.pose.bones:
    pb.rotation_mode = mode
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
body = bpy.context.object
body.name = "Character"
body.scale = (0.3, 0.2, 1.0)
bpy.ops.object.transform_apply(scale=True)
body.vertex_groups.new(name="hips").add(range(len(body.data.vertices)), 1.0, "REPLACE")
body.parent = rig
body.modifiers.new("Armature", "ARMATURE").object = rig
bpy.context.scene.render.fps = 12
bpy.ops.wm.save_as_mainfile(filepath=out)
'''

INSPECT_SCRIPT = r'''
import bpy, sys, json
name = sys.argv[sys.argv.index("--") + 1]
rig = bpy.data.objects["CharacterRig"]
action = bpy.data.actions[name]
ad = rig.animation_data or rig.animation_data_create()
ad.action = action
if getattr(ad, "action_slot", None) is None and getattr(ad, "action_suitable_slots", None):
    ad.action_slot = ad.action_suitable_slots[0]
first, last = (int(round(v)) for v in action.frame_range)
curves = [fc for layer in getattr(action, "layers", []) for strip in layer.strips
          for bag in strip.channelbags for fc in bag.fcurves] or list(getattr(action, "fcurves", []))
out = {"range": [first, last], "cyclic": all(any(m.type == "CYCLES" for m in fc.modifiers) for fc in curves),
       "use_cyclic": action.use_cyclic, "fake_user": action.use_fake_user,
       "paths": sorted({fc.data_path.split(".")[-1] for fc in curves}),
       "frames": []}
for f in range(first, last + 1):
    bpy.context.scene.frame_set(f)
    pb = rig.pose.bones
    out["frames"].append({b: [list(pb[b].head), list(pb[b].tail)] for b in pb.keys()})
print("INSPECT " + json.dumps(out))
'''


def blender_script(script, *args):
    proc = subprocess.run([blender.find_blender(), "-b", "--factory-startup", "--python-exit-code", "1",
                           *args], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.stdout


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    path = tmp_path_factory.mktemp("retarget")
    for name, code in (("make_rig.py", RIG_SCRIPT), ("inspect.py", INSPECT_SCRIPT)):
        (path / name).write_text(code, encoding="utf-8")
    for rig, feet, mode in (("rig_old.blend", "0", "XYZ"), ("rig_feet.blend", "1", "QUATERNION")):
        blender_script("make_rig", "-P", str(path / "make_rig.py"), "--", str(path / rig), feet, mode)
    return path


def inspect(workdir, blend, action):
    out = blender_script("inspect", blend, "-P", str(workdir / "inspect.py"), "--", action)
    line = next(l for l in out.splitlines() if l.startswith("INSPECT "))
    data = json.loads(line[len("INSPECT "):])
    pos = {b: np.array([fr[b] for fr in data["frames"]]) for b in data["frames"][0]}  # (frames, head/tail, xyz)
    return data, pos


def add(workdir, blend, clip, name, **kwargs):
    started = time.perf_counter()
    text = ok(run(blender_anim.blender_add_animation(str(blend), str(clip), name, **kwargs)))
    elapsed = time.perf_counter() - started
    result = json.loads(text.split("Result: ", 1)[1])
    return text, result, elapsed


@pytest.mark.parametrize("naming,prefix,detected", [("mixamo", "mixamorig:", "mixamo"),
                                                    ("cmu_asf", "", "cmu_asf"), ("unreal", "", "fuzzy")])
def test_loop_in_place(workdir, naming, prefix, detected):
    clip = write_bvh(workdir / f"walk_{naming}.bvh", naming, prefix)
    blend = workdir / f"loop_{naming}.blend"
    shutil.copy(workdir / "rig_old.blend", blend)
    text, result, elapsed = add(workdir, blend, clip, "Walk", loop=True, in_place=True, fps=12)
    assert elapsed < 60, f"retargeting took {elapsed:.1f} s"
    assert result["source"]["naming"] == detected and result["source"]["prefix"] == prefix
    # The planted 0.75 s cycle at 12 fps: 9 frames, keyed 1-10 with the first pose repeated.
    assert result["loop"]["frames"] == 9 and abs(result["loop"]["seconds"] - PERIOD) < 0.02, result["loop"]
    assert result["frames"] == 9 and result["loop"]["error"] < 0.01
    assert result["missing_bones"] == ["foot.L", "foot.R"] and "Not in this rig (skipped): foot.L, foot.R" in text
    assert result["rotation_keys"] == "euler"  # the old rig's bones are XYZ Euler
    data, pos = inspect(workdir, blend, "Walk")
    assert data["range"] == [1, 10] and data["use_cyclic"] and data["cyclic"] and data["fake_user"]
    for bone in pos:  # the closing key repeats the first pose
        assert np.abs(pos[bone][-1] - pos[bone][0]).max() < 1e-4, bone
    # Thighs swing in opposite phase (knee forward/back in pose space).
    knee_l, knee_r = pos["thigh.L"][:, 1, 1], pos["thigh.R"][:, 1, 1]
    assert np.ptp(knee_l) > 0.05 and np.corrcoef(knee_l, knee_r)[0, 1] < -0.9
    # Heading normalised: the hip line points +X, so the character faces -Y.
    hip_line = pos["thigh.L"][:, 0] - pos["thigh.R"][:, 0]
    assert np.degrees(np.abs(np.arctan2(hip_line[:, 1], hip_line[:, 0]))).max() < 3
    # In place: no horizontal drift, but the vertical bob survives (scaled to the short legs).
    hips = pos["hips"][:, 0]
    assert np.abs(hips[:, :2]).max() < 0.005
    ratio = result["scale"]
    assert np.ptp(hips[:, 2]) > BOB * ratio
    # The legs reach the floor: the lowest ankle sits at its rest height.
    assert abs(min(pos["shin.L"][:, 1, 2].min(), pos["shin.R"][:, 1, 2].min()) - 0.03) < 0.005


def test_one_shot_with_root_motion(workdir):
    clip = write_bvh(workdir / "walk_cmu.bvh", "cmu_bvh", tpose_first=True)
    blend = workdir / "oneshot.blend"
    shutil.copy(workdir / "rig_feet.blend", blend)
    text, result, elapsed = add(workdir, blend, clip, "Stride", start_seconds=0.5, end_seconds=2.0,
                                loop=False, in_place=False, fps=12)
    assert result["source"]["naming"] == "cmu_bvh" and result["source"]["bind_pose_dropped"]
    assert result["frames"] == 19 and result["loop"] is None  # 1.5 s at 12 fps, both ends included
    assert result["missing_bones"] == [] and {"foot.L", "foot.R"} <= set(result["mapped_bones"])
    assert result["rotation_keys"] == "quaternion"
    assert result["heading_from"] == "average" and abs(abs(result["heading_turned_deg"]) - HEADING) < 3
    data, pos = inspect(workdir, blend, "Stride")
    assert data["range"] == [1, 19] and not data["cyclic"] and not data["use_cyclic"]
    travel = pos["hips"][-1, 0] - pos["hips"][0, 0]
    # Root motion kept, scaled by the leg-length ratio, along -Y after normalising the heading.
    assert abs(math.degrees(math.atan2(travel[0], -travel[1]))) < 3
    assert abs(np.hypot(*travel[:2]) - SPEED * 1.5 * result["scale"]) < 0.1 * SPEED * 1.5 * result["scale"]
    # Feet follow the source feet: flat at rest, so they stay near horizontal on the planted leg.
    foot = pos["foot.L"][:, 1] - pos["foot.L"][:, 0]
    assert np.degrees(np.arctan2(-foot[:, 2], np.hypot(foot[:, 0], foot[:, 1]))).min() < 30


def test_updates_blend_in_place_and_lists(workdir):
    clip = write_bvh(workdir / "walk_inplace.bvh", "mixamo", "mixamorig1:")
    blend = workdir / "inplace" / "hero.blend"
    blend.parent.mkdir()
    shutil.copy(workdir / "rig_old.blend", blend)
    ok(run(blender_anim.blender_add_animation(str(blend), str(clip), "Run")))
    text, result, _ = add(workdir, blend, clip, "Run", loop=True)  # same name again: replaced
    assert result["replaced"] and result["blend"] == str(blend)
    assert sorted(os.listdir(blend.parent)) == ["hero.blend"]  # no temp or .blend1 files left behind
    listing = json.loads(run(blender.blender_list_animations(str(blend))))
    runs = [a for a in listing["animations"] if a["name"] == "Run"]
    # Cyclic with the closing key in range, as sprite_render expects (it renders frames 1-9).
    assert len(runs) == 1 and runs[0]["frame_start"] == 1 and runs[0]["frame_end"] == 10 and runs[0]["cyclic"], listing


def test_rejects_unknown_rig_and_bad_input(workdir):
    clip = write_bvh(workdir / "walk_bad.bvh", "mixamo")
    other = workdir / "other.blend"
    shutil.copy(TEST_CHARACTER, other)
    before = os.path.getmtime(other)
    fails(run(blender_anim.blender_add_animation(str(other), str(clip), "Run")), "Target rig not recognised")
    assert os.path.getmtime(other) == before
    fails(run(blender_anim.blender_add_animation(str(other), str(workdir / "missing.bvh"), "Run")), "not found")
    fails(run(blender_anim.blender_add_animation(str(other), str(other), "Run")), ".bvh")
    fails(run(blender_anim.blender_add_animation(str(other), str(clip), "Run", start_seconds=2, end_seconds=1)),
          "start_seconds")
    junk = workdir / "junk.bvh"
    junk.write_text("HIERARCHY\nROOT Bone\n{\nOFFSET 0 0 0\nCHANNELS 3 Zrotation Xrotation Yrotation\n"
                    "End Site\n{\nOFFSET 0 1 0\n}\n}\nMOTION\nFrames: 2\nFrame Time: 0.1\n0 0 0\n0 0 0\n",
                    encoding="utf-8")
    shutil.copy(workdir / "rig_old.blend", workdir / "junk_target.blend")
    fails(run(blender_anim.blender_add_animation(str(workdir / "junk_target.blend"), str(junk), "Run")),
          "Source skeleton not recognised")
