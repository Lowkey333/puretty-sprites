"""Quick humanoid rig and procedural animations for a generated character.

Image-to-3D models (TRELLIS.2, Hunyuan3D, ...) come out as one dense,
unrigged mesh. For sprites at 32-128 px a rough rig is enough, so this
script:

1. imports the model, joins it into one mesh, rests it on z=0 facing -Y
   and decimates it to a rig-friendly size (UVs and texture are kept),
2. places a humanoid skeleton from height fractions ("landmarks"), with
   the feet and hands found on the mesh when possible,
3. skins it with distance-based weights limited by anatomy (Blender's
   heat weights fail on most generated meshes), keeping everything above
   the neck (head, hair, hat) on the head bone,
4. adds in-place "Idle" and "Walk" actions, then saves a .blend that
   sprite_render.py renders like any rigged character.

Defaults suit chibi/stylized characters standing in an A- or T-pose. When a
part ends up on the wrong bone, look at the model and pass landmarks:
fractions of the height (hips, chest, neck, shoulder, knee) and of the
width (leg_x, shoulder_x).

    blender -b -P auto_rig.py -- --model hero.glb --out hero_rigged.blend
    blender -b -P auto_rig.py -- --model hero.glb --out hero.blend --landmarks hips=0.25,neck=0.5

For a production rig use Mixamo (upload, place markers, download FBX) or
Rigify; this is the five-minute version.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector

DEFAULT_LANDMARKS = {
    # Heights as fractions of the model height (0 = feet, 1 = top).
    "ankle": 0.03, "knee": 0.11, "hips": 0.22, "chest": 0.34, "neck": 0.43,
    "head_top": 1.0, "shoulder": 0.40, "hand": 0.31,
    # Widths as fractions of the model height, from the centre line.
    "leg_x": 0.06, "shoulder_x": 0.09, "hand_x": 0.24,
}


def _import(path: str) -> list:
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path) as (src, dst):
            dst.objects = src.objects
        for ob in dst.objects:
            if ob is not None:
                bpy.context.scene.collection.objects.link(ob)
    else:
        raise ValueError(f"unsupported model type {ext}")
    return [ob for ob in bpy.data.objects if ob not in before]


def _single_mesh(objects: list):
    meshes = [ob for ob in objects if ob.type == "MESH"]
    if not meshes:
        raise ValueError("the model has no mesh")
    for ob in objects:
        if ob.type != "MESH":
            bpy.data.objects.remove(ob, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for ob in meshes:
        ob.select_set(True)
        if ob.parent is not None:
            matrix = ob.matrix_world.copy()
            ob.parent = None
            ob.matrix_world = matrix
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    body = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    body.name = "Character"
    return body


def _vertices(body) -> np.ndarray:
    co = np.empty(len(body.data.vertices) * 3)
    body.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _normalize(body, height: float) -> None:
    """Feet on z=0, centred on x/y, scaled to the given height."""
    co = _vertices(body)
    lo, hi = co.min(axis=0), co.max(axis=0)
    scale = height / (hi[2] - lo[2])
    offset = Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    body.location = offset * scale
    body.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _decimate(body, target_faces: int) -> None:
    faces = len(body.data.polygons)
    if target_faces and faces > target_faces:
        mod = body.modifiers.new("Decimate", "DECIMATE")
        mod.ratio = target_faces / faces
        bpy.ops.object.modifier_apply(modifier=mod.name)
    # Generated meshes are often split per triangle; welding gives the
    # automatic weights a connected surface to work on.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0005)
    bpy.ops.object.mode_set(mode="OBJECT")


def _measure(co: np.ndarray, height: float, marks: dict) -> dict:
    """Refine leg and hand positions from the mesh when the geometry allows."""
    found = dict(marks)
    feet = co[co[:, 2] < height * marks["ankle"] * 2]
    if len(feet) > 50:
        left, right = feet[feet[:, 0] < 0], feet[feet[:, 0] >= 0]
        if len(left) > 20 and len(right) > 20:
            found["leg_x"] = float((np.median(right[:, 0]) - np.median(left[:, 0])) / 2 / height)
    band = co[np.abs(co[:, 2] - height * marks["hand"]) < height * 0.03]
    if len(band) > 50:
        found["hand_x"] = float(np.percentile(np.abs(band[:, 0]), 99) / height * 0.92)
    return found


def _make_armature(height: float, m: dict):
    data = bpy.data.armatures.new("CharacterRig")
    rig = bpy.data.objects.new("CharacterRig", data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = data.edit_bones
    h = height

    def bone(name, head, tail, parent=None, connect=False):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        b.align_roll(Vector((0, -1, 0)))  # local X = world X: rotation X swings forward/back
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        return b

    bone("root", (0, 0, 0), (0, 0.1 * h, 0))
    eb["root"].align_roll(Vector((0, 0, 1)))
    bone("hips", (0, 0, m["hips"] * h), (0, 0, (m["hips"] + m["chest"]) / 2 * h), "root")
    bone("chest", eb["hips"].tail, (0, 0, m["neck"] * h), "hips", True)
    bone("head", (0, 0, m["neck"] * h), (0, 0, m["head_top"] * h), "chest", True)
    for side, sx in (("L", 1), ("R", -1)):
        lx = m["leg_x"] * h * sx
        bone(f"thigh.{side}", (lx, 0, m["hips"] * h), (lx, 0, m["knee"] * h), "hips")
        bone(f"shin.{side}", (lx, 0, m["knee"] * h), (lx, 0, m["ankle"] * h), f"thigh.{side}", True)
        shoulder = Vector((m["shoulder_x"] * h * sx, 0, m["shoulder"] * h))
        hand = Vector((m["hand_x"] * h * sx, 0, m["hand"] * h))
        elbow = shoulder.lerp(hand, 0.5)
        bone(f"upper_arm.{side}", shoulder, elbow, "chest")
        bone(f"forearm.{side}", elbow, hand, f"upper_arm.{side}", True)
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def _segment_distance(points: np.ndarray, a: Vector, b: Vector) -> np.ndarray:
    a, b = np.array(a), np.array(b)
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _skin(body, rig, height: float, m: dict, sigma: float = 0.045, smooth_passes: int = 6) -> str:
    """Distance-based weights with anatomical limits, smoothed over the mesh.

    Blender's automatic (heat) weights fail on most generated meshes, which
    are full of tiny disconnected pieces. Here each vertex weighs the bones
    by distance to the bone segment, limited to where that bone can be:
    legs below the hips on their own side, arms out from the torso in the
    arm band (and not behind the body, which is hair), the head above the
    neck. A few smoothing passes over the mesh edges soften the joints.
    """
    co = _vertices(body)
    h = height
    bones = [b for b in rig.data.bones if b.name != "root"]
    D = np.stack([_segment_distance(co, b.head_local, b.tail_local) for b in bones], axis=1)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    allowed = np.ones_like(D, dtype=bool)
    neck_z, hips_z = m["neck"] * h, m["hips"] * h
    for i, b in enumerate(bones):
        side = 1 if b.name.endswith(".L") else -1 if b.name.endswith(".R") else 0
        if b.name.startswith(("thigh", "shin")):
            allowed[:, i] = (z < hips_z + 0.02 * h) & (x * side > -0.01 * h)
        elif b.name.startswith(("upper_arm", "forearm")):
            arm_band = (z < m["shoulder"] * h + 0.06 * h) & (z > m["hand"] * h - 0.08 * h)
            allowed[:, i] = arm_band & (x * side > m["shoulder_x"] * h * 0.75) & (np.abs(y) < 0.09 * h)
        elif b.name == "head":
            allowed[:, i] = z > neck_z - 0.03 * h
        else:  # hips, chest
            allowed[:, i] = z < neck_z + 0.02 * h
    W = np.exp(-(D / (sigma * h)) ** 2) * allowed
    W[W.sum(axis=1) == 0, [i for i, b in enumerate(bones) if b.name == "hips"][0]] = 1.0
    W /= W.sum(axis=1, keepdims=True)

    edges = np.empty(len(body.data.edges) * 2, dtype=np.int64)
    body.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=len(co)).astype(float)
    for _ in range(smooth_passes):
        acc = np.zeros_like(W)
        np.add.at(acc, edges[:, 0], W[edges[:, 1]])
        np.add.at(acc, edges[:, 1], W[edges[:, 0]])
        neighbour = np.divide(acc, degree[:, None], out=W.copy(), where=degree[:, None] > 0)
        W = (0.5 * W + 0.5 * neighbour) * allowed
        W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
    head_only = z > neck_z + 0.03 * h
    W[head_only] = 0
    W[head_only, [i for i, b in enumerate(bones) if b.name == "head"][0]] = 1.0

    for g in list(body.vertex_groups):
        body.vertex_groups.remove(g)
    for i, b in enumerate(bones):
        group = body.vertex_groups.new(name=b.name)
        for weight in np.unique(np.round(W[:, i], 2)):
            if weight <= 0:
                continue
            idx = np.nonzero(np.round(W[:, i], 2) == weight)[0].tolist()
            group.add(idx, float(weight), "REPLACE")
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    body.parent = rig
    counts = {b.name: int((W[:, i] > 0.5).sum()) for i, b in enumerate(bones)}
    weak = [n for n, c in counts.items() if c == 0]
    return "geometric" + (f" (bones owning no vertices: {', '.join(weak)})" if weak else "")


def _key(pb, frame, rot=None, loc=None):
    if rot is not None:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = Euler([math.radians(a) for a in rot])
        pb.keyframe_insert("rotation_euler", frame=frame)
    if loc is not None:
        pb.location = Vector(loc)
        pb.keyframe_insert("location", frame=frame)


def _new_action(rig, name: str):
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    rig.animation_data_create()
    rig.animation_data.action = action
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)
    return action


def _cyclic(action) -> None:
    curves = []
    if hasattr(action, "fcurves") and action.fcurves:
        curves = list(action.fcurves)
    else:  # Blender 5 layered actions
        for layer in getattr(action, "layers", []):
            for strip in layer.strips:
                for bag in strip.channelbags:
                    curves += list(bag.fcurves)
    for fc in curves:
        if not any(m.type == "CYCLES" for m in fc.modifiers):
            fc.modifiers.new("CYCLES")
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"


def _walk(rig, height: float, frames: int = 16, stride: float = 26.0, arm_swing: float = 22.0) -> None:
    action = _new_action(rig, "Walk")
    pose = rig.pose.bones
    for i in range(frames + 1):
        f = 1 + i
        t = i / frames * 2 * math.pi
        s = math.sin(t)
        lift_l = max(0.0, -math.cos(t)) * 28  # knee bends while the leg swings back
        lift_r = max(0.0, math.cos(t)) * 28
        _key(pose["thigh.L"], f, rot=(stride * s, 0, 0))
        _key(pose["thigh.R"], f, rot=(-stride * s, 0, 0))
        _key(pose["shin.L"], f, rot=(-lift_l, 0, 0))
        _key(pose["shin.R"], f, rot=(-lift_r, 0, 0))
        _key(pose["upper_arm.L"], f, rot=(-arm_swing * s, 0, 0))
        _key(pose["upper_arm.R"], f, rot=(arm_swing * s, 0, 0))
        _key(pose["forearm.L"], f, rot=(-6 - 6 * max(0.0, -s), 0, 0))
        _key(pose["forearm.R"], f, rot=(-6 - 6 * max(0.0, s), 0, 0))
        bob = -abs(math.cos(t)) * 0.025 * height  # lowest when the feet pass each other
        _key(pose["root"], f, loc=(0, 0, 0))
        _key(pose["hips"], f, rot=(0, 0, 5 * s), loc=(0, bob, 0))
        _key(pose["chest"], f, rot=(3, 0, -7 * s))
        _key(pose["head"], f, rot=(-2, 0, 2 * s))
    action.frame_range = (1, frames)
    _cyclic(action)


def _idle(rig, height: float, frames: int = 24) -> None:
    action = _new_action(rig, "Idle")
    pose = rig.pose.bones
    for i in range(frames + 1):
        f = 1 + i
        s = math.sin(i / frames * 2 * math.pi)
        _key(pose["hips"], f, loc=(0, -0.006 * height * (1 + s), 0))
        _key(pose["chest"], f, rot=(2 * s, 0, 0))
        _key(pose["head"], f, rot=(-2 * s, 0, 1.5 * s))
        for side, sign in (("L", 1), ("R", -1)):
            _key(pose[f"upper_arm.{side}"], f, rot=(3 * s, 0, sign * 2 * s))
            _key(pose[f"forearm.{side}"], f, rot=(-4 - 2 * s, 0, 0))
    action.frame_range = (1, frames)
    _cyclic(action)


def rig_character(model: str, out: str, landmarks: dict | None = None, target_faces: int = 40000,
                  height: float = 1.0) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = _single_mesh(_import(model))
    _normalize(body, height)
    faces_before = len(body.data.polygons)
    _decimate(body, target_faces)
    marks = dict(DEFAULT_LANDMARKS, **(landmarks or {}))
    marks = _measure(_vertices(body), height, marks) if not landmarks else marks
    rig = _make_armature(height, marks)
    method = _skin(body, rig, height, marks)
    _idle(rig, height)
    _walk(rig, height)
    rig.animation_data.action = bpy.data.actions["Idle"]
    scene = bpy.context.scene
    scene.render.fps = 12
    scene.frame_start, scene.frame_end = 1, 24
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
    return {"blend": os.path.abspath(out), "faces_before": faces_before, "faces": len(body.data.polygons),
            "weights": method, "landmarks": {k: round(v, 3) for k, v in marks.items()},
            "actions": ["Idle", "Walk"], "fps": 12}


def _parse_landmarks(text: str) -> dict:
    out = {}
    for part in filter(None, (text or "").split(",")):
        key, _, value = part.partition("=")
        if key.strip() not in DEFAULT_LANDMARKS:
            raise SystemExit(f"unknown landmark {key!r}; use {', '.join(DEFAULT_LANDMARKS)}")
        out[key.strip()] = float(value)
    return out


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Rig and animate a generated humanoid character.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--landmarks", default="", help="e.g. hips=0.25,neck=0.5,leg_x=0.07")
    parser.add_argument("--faces", type=int, default=40000, help="decimate to about this many faces (0 = keep)")
    args = parser.parse_args(argv)
    result = rig_character(args.model, args.out, _parse_landmarks(args.landmarks), args.faces)
    print("AUTO_RIG_RESULT " + json.dumps(result))
