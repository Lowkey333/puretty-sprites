"""Retarget motion capture onto the character rig made by auto_rig.py.

Adds one action to a rigged .blend from a BVH file (CMU, Mixamo-style and
most mocap exports) or a Mixamo FBX downloaded "Without Skin". Each target
bone is aimed along the matching segment of the source skeleton (thigh =
hip -> knee, shin = knee -> ankle, upper arm = shoulder -> elbow, ...), so
the two skeletons can differ in proportions and rest pose (a T-pose clip on
an A-pose chibi); twist comes from the source bone's own rotation. Then:

- the clip is turned to face -Y by the average heading of its hip line, or
  by the stance it starts from when the hips swing or turn (a roundhouse
  kick, a spin, a fall with a half turn),
- root motion is scaled by the leg-length ratio and the feet are put on the
  floor; frames where the torso or head would sink into it are lifted,
- with in_place the horizontal travel is removed (the vertical bob stays;
  loops keep their side-to-side sway, one-shots keep the hips on the spot),
- it is resampled to the scene's frame rate (or --fps), and with loop the
  best seamless cycle of 0.4-1.6 s is cut out and keyed to repeat; a CMU
  T-pose glued in front of the motion is dropped.

The target is the rig contract of auto_rig.py: an armature "CharacterRig"
facing -Y, Z up, with root, hips, chest, head, thigh/shin/foot.L/R and
upper_arm/forearm.L/R (.L on +X). Bones the rig lacks (older rigs have no
feet) are skipped and reported; other bones (hand.L/R) are keyed at rest,
following their parent. Keys follow each bone's rotation mode: quaternions
on the current rig, XYZ Euler on rigs made before it (switching the mode
would break their Idle/Walk). Loops are cyclic actions with frame range
(1, N+1), the last key repeating the first, like auto_rig.py's loops.

    blender -b hero.blend -P retarget.py -- --source run.bvh --name Run --loop --in-place
    blender -b hero.blend -P retarget.py -- --source punch.fbx --name Punch --start 0.4 --end 1.6 --out hero2.blend

Inside a live Blender (for example through blender-mcp):

    import sys; sys.path.append("C:/AI/Puretty-sprites/integrations/blender")
    import retarget
    retarget.add_animation("C:/mocap/run.bvh", "Run", loop=True, in_place=True)

Prints RETARGET_RESULT {json} (or RETARGET_ERROR message) for the MCP server.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import bpy
import numpy as np
from mathutils import Matrix

RIG_NAME = "CharacterRig"
LOOP_MIN, LOOP_MAX = 0.4, 1.6  # seconds
LOOP_WARN = 0.1  # loop error above which the cycle visibly pops

# Target bone: (from landmark, to landmark, source bone giving the twist, twist reference,
# rest-calibrated). Limbs copy the source directions exactly. The spine, head and feet are
# corrected by the difference between the two rest poses, because those landmarks differ
# between skeletons (where the root joint sits, ball or tip of the foot).
SEGMENTS = {
    "hips": ("pelvis", "spine", "pelvis", "forward", True),
    "chest": ("spine", "neck", "chest", "forward", True),
    "head": ("neck", "head_top", "head", "forward", True),
}
for _s in "LR":
    SEGMENTS.update({
        f"thigh.{_s}": (f"hip.{_s}", f"knee.{_s}", f"upleg.{_s}", "forward", False),
        f"shin.{_s}": (f"knee.{_s}", f"ankle.{_s}", f"leg.{_s}", "forward", False),
        f"foot.{_s}": (f"ankle.{_s}", f"toe.{_s}", f"foot.{_s}", "left", True),
        f"upper_arm.{_s}": (f"shoulder.{_s}", f"elbow.{_s}", f"arm.{_s}", "forward", False),
        f"forearm.{_s}": (f"elbow.{_s}", f"wrist.{_s}", f"forearm.{_s}", "forward", False),
    })
TARGET_BONES = ["root", *SEGMENTS]
REQUIRED = ("hips", "thigh.L", "thigh.R", "shin.L", "shin.R")
CLAMP_BONES = ("hips", "chest", "head")  # their mesh is kept above the floor
REFS = {"forward": np.array([0.0, -1.0, 0.0]), "left": np.array([1.0, 0.0, 0.0])}

# Source naming. Mixamo and the CMU "cgspeed" BVH conversion share these names (any prefix,
# e.g. "mixamorig:"); the CMU ASF names come from conversions of the original AMC files.
LIMBS_MIXAMO = {"upleg": "{side}UpLeg", "leg": "{side}Leg", "foot": "{side}Foot", "toe": "{side}ToeBase",
                "arm": "{side}Arm", "forearm": "{side}ForeArm", "hand": "{side}Hand"}
LIMBS_ASF = {"upleg": "{s}femur", "leg": "{s}tibia", "foot": "{s}foot", "toe": "{s}toes",
             "arm": "{s}humerus", "forearm": "{s}radius", "hand": "{s}wrist"}
_SIDE_WORDS = {"left": "L", "l": "L", "right": "R", "r": "R"}
_PART_WORDS = (  # first match wins: forearm before arm, upleg before leg, toe before foot
    ("toe", ("toe", "ball")),
    ("forearm", ("forearm", "lowerarm", "elbow", "radius", "ulna")),
    ("hand", ("hand", "wrist")),
    ("upleg", ("upleg", "upperleg", "thigh", "femur")),
    ("leg", ("lowerleg", "shin", "calf", "knee", "tibia", "leg")),
    ("foot", ("foot", "ankle")),
    ("arm", ("upperarm", "humerus", "arm")),
    ("clavicle", ("shoulder", "clavicle", "collar")),
    ("hip", ("hip",)),
)
_DATA_KINDS = ("objects", "armatures", "meshes", "materials", "images", "textures", "actions",
               "collections", "cameras", "lights", "node_groups")


class RetargetError(Exception):
    pass


# ---------------------------------------------------------------- math helpers

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def _orthonormal(m: np.ndarray) -> np.ndarray:
    """Remove scale from rotation matrices (columns normalized)."""
    return m / np.linalg.norm(m, axis=-2, keepdims=True).clip(1e-12)


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction a to direction b."""
    a, b = _unit(a), _unit(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:
        axis = _unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _frames(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotations (n, 3, 3) whose Y axis is y and whose X axis leans toward ref."""
    y = _unit(np.atleast_2d(y))
    ref = np.broadcast_to(ref, y.shape)
    x = ref - np.sum(ref * y, axis=-1, keepdims=True) * y
    bad = np.linalg.norm(x, axis=-1) < 1e-6
    if bad.any():  # reference along the bone: any perpendicular will do
        alt = np.cross(y[bad], [0.0, 0.0, 1.0])
        weak = np.linalg.norm(alt, axis=-1) < 1e-6
        alt[weak] = np.cross(y[bad][weak], [1.0, 0.0, 0.0])
        x[bad] = alt
    x = _unit(x)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def _heading(left: np.ndarray) -> float:
    """Facing angle (atan2) of the summed horizontal facing of hip-line vectors (left side minus right)."""
    facing = np.stack([left[:, 1], -left[:, 0]], axis=1).sum(axis=0)  # left x up
    return math.atan2(facing[1], facing[0])


def _interp(values: np.ndarray, rate: float, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of uniformly sampled values (time 0 at index 0) at times t."""
    if len(values) == 1:
        return np.repeat(values, len(t), axis=0)
    x = np.clip(np.asarray(t) * rate, 0, len(values) - 1)
    i = np.minimum(np.floor(x).astype(int), len(values) - 2)
    f = (x - i).reshape(-1, *([1] * (values.ndim - 1)))
    return values[i] * (1 - f) + values[i + 1] * f


# ---------------------------------------------------------------- target rig

def find_rig(name: str = ""):
    """The auto-rig armature, or raise RetargetError naming what was found instead."""
    arms = [ob for ob in bpy.context.scene.objects if ob.type == "ARMATURE"]
    if name:
        rig = bpy.data.objects.get(name)
        if rig is None or rig.type != "ARMATURE":
            raise RetargetError(f"No armature named {name!r} (armatures: {', '.join(a.name for a in arms) or 'none'})")
        candidates = [rig]
    else:
        named = bpy.data.objects.get(RIG_NAME)
        candidates = [named] if named is not None and named.type == "ARMATURE" else arms
    for rig in candidates:
        if all(b in rig.data.bones for b in REQUIRED):
            return rig
    found = "; ".join(f"'{a.name}' with bones {', '.join(list(a.data.bones.keys())[:12])}"
                      for a in candidates) or "no armature"
    raise RetargetError(
        "Target rig not recognised: retargeting needs the skeleton made by auto_rig.py / blender_auto_rig "
        f"(an armature '{RIG_NAME}' with bones {', '.join(TARGET_BONES)}); found {found}")


class _TargetBone:
    """A rig bone's rest pose, in armature space and world space."""

    def __init__(self, bone, mw: np.ndarray):
        self.name = bone.name
        self.parent = bone.parent.name if bone.parent else None
        self.rest = np.array(bone.matrix_local)
        parent_rest = np.array(bone.parent.matrix_local) if bone.parent else np.eye(4)
        self.rest_rel = np.linalg.inv(parent_rest) @ self.rest
        self.length = bone.length
        self.rest_world = _orthonormal((mw @ self.rest)[:3, :3])
        self.tail_world = (mw @ np.append(np.array(bone.tail_local), 1.0))[:3]
        self.head_world = (mw @ np.append(np.array(bone.head_local), 1.0))[:3]


def _fk(order, bones, rot_arm, hips_head, n):
    """Armature-space pose and matrix_basis per bone (parents first).

    matrix_basis = (parent_pose @ rest_rel)^-1 @ desired, where desired keeps the
    position the parent gives the bone (except the hips) with the wanted rotation.
    """
    eye = np.broadcast_to(np.eye(4), (n, 4, 4))
    pose, basis = {}, {}
    for name in order:
        bone = bones[name]
        base = (pose[bone.parent] if bone.parent else eye) @ bone.rest_rel
        desired = base.copy()
        if name in rot_arm:
            desired[:, :3, :3] = rot_arm[name]
        if name == "hips":
            desired[:, :3, 3] = hips_head
        pose[name] = desired
        basis[name] = np.linalg.inv(base) @ desired
    return pose, basis


def _tail_world(pose, bone, mw):
    p = pose[bone.name] @ np.array([0.0, bone.length, 0.0, 1.0])
    return p @ mw.T


def _clamp_points(rig, mw_inv, above: float):
    """Mesh vertices that follow the torso or head, to keep above the floor.

    Returns their rest positions (armature space) per bone, and the floor height
    (the mesh's lowest vertex). Only vertices above the hip joint count: a skirt
    or a cape would make a crouch look like sinking into the floor.
    """
    points, floor = {}, None
    meshes = [ob for ob in bpy.data.objects if ob.type == "MESH" and any(
        m.type == "ARMATURE" and m.object == rig for m in ob.modifiers)]
    for ob in meshes:
        verts = ob.data.vertices
        if not len(verts):
            continue
        co = np.empty(len(verts) * 3)
        verts.foreach_get("co", co)
        co = np.c_[co.reshape(-1, 3), np.ones(len(verts))] @ np.array(ob.matrix_world).T
        floor = co[:, 2].min() if floor is None else min(floor, co[:, 2].min())
        group_bone = {vg.index: vg.name for vg in ob.vertex_groups if vg.name in rig.data.bones}
        step = max(1, len(verts) // 20000)
        picked = {}
        for i in range(0, len(verts), step):
            best, weight = None, 0.0
            for g in verts[i].groups:
                if g.weight > weight and g.group in group_bone:
                    best, weight = group_bone[g.group], g.weight
            if best in CLAMP_BONES and co[i, 2] >= above:
                picked.setdefault(best, []).append(i)
        for bone, idx in picked.items():
            pts = co[idx] @ mw_inv.T
            points[bone] = np.concatenate([points[bone], pts]) if bone in points else pts
    return points, floor


# ---------------------------------------------------------------- source skeleton

def _bvh_frame_time(path: str) -> float:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().lower().startswith("frame time"):
                return float(line.split(":", 1)[1])
    raise RetargetError(f"{os.path.basename(path)} has no 'Frame Time' line; is it a BVH file?")


def _snapshot() -> dict:
    return {kind: {d.as_pointer() for d in getattr(bpy.data, kind)} for kind in _DATA_KINDS}


def _remove_new(before: dict) -> None:
    ids = [d for kind, old in before.items() for d in getattr(bpy.data, kind) if d.as_pointer() not in old]
    if ids:
        bpy.data.batch_remove(ids)


def import_source(path: str) -> float:
    """Import a BVH or FBX into the scene; returns its frames per second."""
    ext = os.path.splitext(path)[1].lower()
    scene = bpy.context.scene
    fps = (scene.render.fps, scene.render.fps_base)
    try:
        if ext == ".bvh":
            rate = 1.0 / max(1e-6, _bvh_frame_time(path))
            bpy.ops.import_anim.bvh(filepath=path, target="ARMATURE", frame_start=1, use_fps_scale=False,
                                    update_scene_fps=False, update_scene_duration=False, rotate_mode="NATIVE",
                                    axis_forward="-Z", axis_up="Y")
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path, use_anim=True, ignore_leaf_bones=False,
                                     automatic_bone_orientation=False, use_image_search=False)
            rate = scene.render.fps / scene.render.fps_base  # the importer sets the file's rate
        else:
            raise RetargetError(f"Unsupported motion file {os.path.basename(path)}: use .bvh or .fbx")
    except RuntimeError as e:
        raise RetargetError(f"Could not import {os.path.basename(path)}: {e}") from e
    finally:
        scene.render.fps, scene.render.fps_base = fps
    return rate


def _words(name: str) -> list[str]:
    base = re.split(r"[:|]", name)[-1]
    base = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", base)
    return [w for w in re.split(r"[^a-z0-9]+", base.lower()) if w]


def _classify(name: str):
    """(side or None, limb part or None) from a bone name."""
    side, rest = None, []
    for w in _words(name):
        if side is None and w in _SIDE_WORDS:
            side = _SIDE_WORDS[w]
        else:
            rest.append(w)
    core = "".join(rest)
    if side is None and core[:1] in ("l", "r"):  # lfemur, rhand, LHipJoint
        for part, keys in _PART_WORDS:
            if any(core[1:].startswith(k) for k in keys):
                return core[0].upper(), part
    for part, keys in _PART_WORDS:
        if any(k in core for k in keys):
            return side, part
    return side, None


def _depth(bone) -> int:
    return len(bone.parent_recursive)


def _fuzzy_roles(bones) -> dict:
    roles = {}
    found = {}
    for bone in bones:
        side, part = _classify(bone.name)
        if side and part:
            found.setdefault(f"{part}.{side}", []).append(bone)
        core = "".join(_words(bone.name))
        if side is None and core in ("hips", "hip", "pelvis"):
            found.setdefault("pelvis", []).append(bone)
        if side is None and "head" in core:
            key = "head_end" if any(k in core for k in ("top", "end", "nub", "site")) else "head"
            found.setdefault(key, []).append(bone)
    for key, cands in found.items():
        roles[key] = min(cands, key=_depth).name
    by_name = {b.name: b for b in bones}
    for s in "LR":
        leg = by_name.get(roles.get(f"leg.{s}"))
        if leg is not None and f"upleg.{s}" not in roles:
            hip = roles.get(f"hip.{s}")
            if hip:
                roles[f"upleg.{s}"] = hip
            elif leg.parent is not None:
                roles[f"upleg.{s}"] = leg.parent.name
        if leg is not None and f"foot.{s}" not in roles and leg.children:
            roles[f"foot.{s}"] = leg.children[0].name
        foot = by_name.get(roles.get(f"foot.{s}"))
        if foot is not None and f"toe.{s}" not in roles and foot.children:
            roles[f"toe.{s}"] = foot.children[0].name
        if roles.get(f"toe.{s}") == roles.get(f"foot.{s}"):
            roles.pop(f"toe.{s}", None)
        fore = by_name.get(roles.get(f"forearm.{s}"))
        if fore is not None:
            if f"arm.{s}" not in roles and fore.parent is not None:
                roles[f"arm.{s}"] = fore.parent.name
            if f"hand.{s}" not in roles and fore.children:
                roles[f"hand.{s}"] = fore.children[0].name
        for part in ("hip", "clavicle"):
            roles.pop(f"{part}.{s}", None)
    return roles


def detect_skeleton(arm) -> tuple[str, str, dict]:
    """(naming, prefix, roles): roles maps pelvis/head/head_end and upleg.L ... hand.R to bone names."""
    bones = arm.data.bones
    names = set(bones.keys())
    naming, prefix, roles = "fuzzy", "", None
    for name in sorted(names):
        p = name[:-4]
        if name.endswith("Hips") and {p + "LeftUpLeg", p + "RightUpLeg", p + "LeftLeg", p + "RightLeg"} <= names:
            naming = "cmu_bvh" if (p + "LHipJoint" in names or p + "LowerBack" in names) else "mixamo"
            prefix = p
            roles = {"pelvis": name, "head": p + "Head", "head_end": p + "HeadTop_End"}
            for side in ("Left", "Right"):
                for part, pattern in LIMBS_MIXAMO.items():
                    roles[f"{part}.{side[0]}"] = p + pattern.format(side=side)
            break
    if roles is None:
        lower = {n.lower(): n for n in names}
        for low in sorted(lower):
            p = low[:-6]
            if low.endswith("lfemur") and p + "rfemur" in lower and p + "ltibia" in lower:
                naming, prefix = "cmu_asf", lower[low][:len(p)]
                roles = {"pelvis": lower.get(p + "root") or lower.get(p + "hips"), "head": lower.get(p + "head")}
                for s in "lr":
                    for part, pattern in LIMBS_ASF.items():
                        roles[f"{part}.{s.upper()}"] = lower.get(p + pattern.format(s=s))
                break
    if roles is None:
        roles = _fuzzy_roles(list(bones))
    roles = {k: v for k, v in roles.items() if v and v in names}
    if "pelvis" not in roles and "upleg.L" in roles and "upleg.R" in roles:
        left = [b.name for b in bones[roles["upleg.L"]].parent_recursive]
        common = [b.name for b in bones[roles["upleg.R"]].parent_recursive if b.name in left]
        if common:
            roles["pelvis"] = common[0]
    missing = [k for k in ("pelvis", "upleg.L", "upleg.R", "leg.L", "leg.R") if k not in roles]
    if missing:
        raise RetargetError(
            f"Source skeleton not recognised (no {', '.join(missing)}; naming tried: Mixamo, CMU BVH, CMU ASF, "
            f"generic). Bones: {', '.join(sorted(names)[:40])}")
    _structure(bones, roles)
    return naming, prefix, roles


def _structure(bones, roles: dict) -> None:
    """Add the chest (where the arms meet) and the spine and neck chains, from the hierarchy."""
    pelvis = bones[roles["pelvis"]]

    def chain_to(bone):  # bones from just below `bone` up to the pelvis, pelvis excluded, top first
        out = []
        while bone is not None and bone != pelvis:
            out.append(bone)
            bone = bone.parent
        return out if bone == pelvis else None

    chest = None
    if "arm.L" in roles and "arm.R" in roles:
        left = list(bones[roles["arm.L"]].parent_recursive)
        common = [b for b in bones[roles["arm.R"]].parent_recursive if b in left]
        if common and (common[0] == pelvis or chain_to(common[0]) is not None):
            chest = common[0]
    head = bones.get(roles.get("head", ""))
    if chest is None:
        chest = head.parent if head is not None and head.parent is not None else pelvis
    spine = list(reversed(chain_to(chest) or []))
    neck = []
    if head is not None:
        bone = head.parent
        while bone is not None and bone != chest and bone != pelvis:
            neck.append(bone)
            bone = bone.parent
        neck.reverse()
    roles["chest"] = chest.name
    roles["spine_chain"] = [b.name for b in spine]
    roles["neck_chain"] = [b.name for b in neck]


def _landmarks(bones, roles: dict, mw: np.ndarray) -> dict:
    """Landmark name -> [((bone, 'head'|'tail'), weight)], chosen on the rest skeleton."""
    def rest(key):
        b = bones[key[0]]
        v = b.head_local if key[1] == "head" else b.tail_local
        return (mw @ np.append(np.array(v), 1.0))[:3]

    def point(role, end="head"):
        return [((roles[role], end), 1.0)]

    lm = {"pelvis": point("pelvis")}
    for s in "LR":
        lm[f"hip.{s}"] = point(f"upleg.{s}")
        lm[f"knee.{s}"] = point(f"leg.{s}")
        lm[f"ankle.{s}"] = point(f"foot.{s}") if f"foot.{s}" in roles else point(f"leg.{s}", "tail")
        if f"toe.{s}" in roles:
            lm[f"toe.{s}"] = point(f"toe.{s}")
        elif f"foot.{s}" in roles:
            lm[f"toe.{s}"] = point(f"foot.{s}", "tail")
        if f"arm.{s}" in roles and f"forearm.{s}" in roles:
            lm[f"shoulder.{s}"] = point(f"arm.{s}")
            lm[f"elbow.{s}"] = point(f"forearm.{s}")
            lm[f"wrist.{s}"] = point(f"hand.{s}") if f"hand.{s}" in roles else point(f"forearm.{s}", "tail")
    pelvis = rest((roles["pelvis"], "head"))
    if "head" in roles:
        lm["head_top"] = point("head_end") if "head_end" in roles else point("head", "tail")
        # Neck base: the neck joint nearest the shoulders' height (CMU has two neck joints).
        up = _unit(rest(lm["head_top"][0][0]) - pelvis)
        cands = [(b, "head") for b in roles["neck_chain"]] + [(roles["head"], "head")]
        if "shoulder.L" in lm and "shoulder.R" in lm:
            level = ((rest(lm["shoulder.L"][0][0]) + rest(lm["shoulder.R"][0][0])) / 2 - pelvis) @ up
            cands.sort(key=lambda k: abs((rest(k) - pelvis) @ up - level))
        lm["neck"] = [(cands[0], 1.0)]
    elif roles["neck_chain"]:
        lm["neck"] = [((roles["neck_chain"][0], "head"), 1.0)]
    if "neck" in lm:
        # Mid-spine: the spine joint nearest 40% of the way up, where the target's hips bone ends.
        up = _unit(rest(lm["neck"][0][0]) - pelvis)
        height = (rest(lm["neck"][0][0]) - pelvis) @ up
        cands = [(b, "head") for b in roles["spine_chain"]
                 if 0.1 * height < (rest((b, "head")) - pelvis) @ up < 0.9 * height]
        if cands:
            lm["spine"] = [(min(cands, key=lambda k: abs((rest(k) - pelvis) @ up - 0.4 * height)), 1.0)]
        else:
            lm["spine"] = [((roles["pelvis"], "head"), 0.6), (lm["neck"][0][0], 0.4)]
    return lm


def _sample(src, points: set, twist: set, frames: list[int]):
    """World positions of (bone, end) points and world rotations of bones, per frame."""
    scene = bpy.context.scene
    n = len(frames)
    pos = {key: np.empty((n, 3)) for key in points}
    rot = {b: np.empty((n, 3, 3)) for b in twist}
    for i, f in enumerate(frames):
        scene.frame_set(f)
        ev = src.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mw = np.array(ev.matrix_world)
        pbs = ev.pose.bones
        for key in points:
            pb = pbs[key[0]]
            pos[key][i] = mw[:3, :3] @ np.array(pb.head if key[1] == "head" else pb.tail) + mw[:3, 3]
        for b in twist:
            rot[b][i] = mw[:3, :3] @ np.array(pbs[b].matrix.to_3x3())
    for b in rot:
        rot[b] = _orthonormal(rot[b])
    return pos, rot


# ---------------------------------------------------------------- loop search

def _find_loop(feat: np.ndarray, rate: float, lo: int, hi: int):
    """(start sample, length in samples, error) of the most seamless cycle.

    The error compares pose and velocity at both ends, relative to how much the
    pose varies inside the cycle, so standing still never counts as a loop.
    Among near-equal candidates the shortest wins (one cycle, not two).
    """
    n = len(feat)
    vel = np.gradient(feat, axis=0) * rate * 0.1
    c1 = np.vstack([np.zeros(feat.shape[1]), np.cumsum(feat, axis=0)])
    c2 = np.concatenate([[0.0], np.cumsum((feat ** 2).sum(axis=1))])
    eps = 1e-3 * float(feat.var(axis=0).sum()) + 1e-12
    results = []
    for length in range(lo, min(hi, n - 1) + 1):
        m = n - length
        d = ((feat[:m] - feat[length:]) ** 2).sum(axis=1) + ((vel[:m] - vel[length:]) ** 2).sum(axis=1)
        s1 = c1[length + 1:length + 1 + m] - c1[:m]
        s2 = c2[length + 1:length + 1 + m] - c2[:m]
        var = s2 / (length + 1) - (s1 ** 2).sum(axis=1) / (length + 1) ** 2
        cost = d / (var + eps)
        i = int(np.argmin(cost))
        results.append((float(cost[i]), length, i))
    if not results:
        return None
    best = min(r[0] for r in results)
    error, length, start = min((r for r in results if r[0] <= best * 1.25 + 0.005), key=lambda r: r[1])
    return start, length, error


# ---------------------------------------------------------------- keys

def _new_action(rig, name: str):
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    bag = None
    if hasattr(action, "slots"):  # Blender 4.4+ layered actions
        from bpy_extras import anim_utils
        slot = action.slots.new(id_type="OBJECT", name=rig.name)
        bag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
    return action, bag


def _write_curve(action, bag, path: str, index: int, group: str, values, cyclic: bool, offset: bool):
    fc = (bag.fcurves.new(path, index=index, group_name=group) if bag is not None
          else action.fcurves.new(path, index=index, action_group=group))
    fc.keyframe_points.add(len(values))
    co = np.empty(2 * len(values))
    co[0::2] = np.arange(1, len(values) + 1)
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points:
        kp.interpolation = "BEZIER"
        kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
    if cyclic:
        mod = fc.modifiers.new("CYCLES")
        if offset:
            mod.mode_before = mod.mode_after = "REPEAT_OFFSET"
    fc.update()


def _rotation_keys(pb, mats: np.ndarray):
    """(data path, [values per channel]) in the pose bone's rotation mode, continuous over time."""
    mode = pb.rotation_mode
    out, prev = [], None
    for m in mats:
        mat = Matrix(m.tolist())
        if mode == "QUATERNION":
            q = mat.to_quaternion()
            if prev is not None and q.dot(prev) < 0:
                q.negate()
            prev = q
            out.append(tuple(q))
        elif mode == "AXIS_ANGLE":
            axis, angle = mat.to_quaternion().to_axis_angle()
            out.append((angle, *axis))
        else:
            e = mat.to_euler(mode, prev) if prev is not None else mat.to_euler(mode)
            prev = e
            out.append(tuple(e))
    path = {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(mode, "rotation_euler")
    return path, [list(ch) for ch in zip(*out)]


# ---------------------------------------------------------------- the clip

FEATURES = ("knee.L", "knee.R", "ankle.L", "ankle.R", "toe.L", "toe.R", "elbow.L", "elbow.R",
            "wrist.L", "wrist.R", "neck", "head_top")


class _Clip:
    """A source clip sampled at its own frame rate (world space), with its rest pose."""

    def __init__(self, path, rate, naming, prefix, bone_count, lm, mapped, pos, rot, rest_pos, rest_rot):
        self.file, self.rate, self.naming, self.prefix, self.bone_count = path, rate, naming, prefix, bone_count
        self.lm, self.mapped, self.pos, self.rot = lm, mapped, pos, rot
        self.rest_pos, self.rest_rot = rest_pos, rest_rot
        self.first = 0  # samples to skip (a bind pose in front of the motion)
        # Up axis: the rest torso should point +Z (fixes Z-up files read as Y-up).
        mid = (self.rest("hip.L") + self.rest("hip.R")) / 2
        top = self.rest("neck") if "neck" in lm else self.rest("pelvis") + np.array([0, 0, 1.0])
        torso = _unit(top - mid)
        self.up_fix = _rotation_between(torso, np.array([0, 0, 1.0])) if torso[2] < 0.8 else np.eye(3)
        # The rest pose turned to face -Y: where the twist references and calibrations are defined.
        turn = _rz(-math.pi / 2 - _heading((self.up_fix @ (self.rest("hip.L") - self.rest("hip.R")))[None]))
        self.canon_rest = turn @ self.up_fix
        self.ref_world = {k: self.canon_rest.T @ v for k, v in REFS.items()}
        self.leg = float(np.mean([np.linalg.norm(self.rest(f"knee.{s}") - self.rest(f"hip.{s}")) +
                                  np.linalg.norm(self.rest(f"ankle.{s}") - self.rest(f"knee.{s}")) for s in "LR"]))

    @property
    def duration(self) -> float:
        return (len(self.pos[next(iter(self.pos))]) - 1 - self.first) / self.rate

    def rest(self, name: str) -> np.ndarray:
        return sum(self.rest_pos[key] * w for key, w in self.lm[name])

    def at(self, name: str, pos: dict | None = None) -> np.ndarray:
        return sum((self.pos if pos is None else pos)[key] * w for key, w in self.lm[name])

    def features(self, turn: np.ndarray) -> np.ndarray:
        """Per-frame pose for comparing frames: joints around the mid-hip, in leg lengths, plus its height."""
        mid = (self.at("hip.L") + self.at("hip.R")) / 2
        rows = [((self.at(k) - mid) @ turn.T) / self.leg for k in FEATURES if k in self.lm]
        height = (mid @ turn.T)[:, 2:3]
        return np.hstack(rows + [(height - height.mean()) / self.leg])

    def resample(self, t: np.ndarray):
        """Point positions and bone rotations at clip times t (seconds from the first motion frame)."""
        return ({k: _interp(v[self.first:], self.rate, t) for k, v in self.pos.items()},
                {k: _interp(v[self.first:], self.rate, t) for k, v in self.rot.items()})


def _read_clip(source: str, rig) -> _Clip:
    """Import the clip, sample its skeleton at every frame, and remove it again.

    The scene is left as it was: frame, selection, active object and the rig's action.
    """
    scene, view_layer = bpy.context.scene, bpy.context.view_layer
    frame = scene.frame_current
    active = view_layer.objects.active
    selected = [ob for ob in scene.objects if ob.select_get()]
    ad = rig.animation_data
    rig_action = ad.action if ad else None
    rig_slot = getattr(ad, "action_slot", None) if ad else None
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    if ad is not None:
        ad.action = None  # nothing else re-evaluates while the clip is sampled
    before = _snapshot()
    try:
        rate = import_source(source)
        animated = [ob for ob in bpy.data.objects if ob.type == "ARMATURE" and ob.as_pointer() not in
                    before["objects"] and ob.animation_data and ob.animation_data.action]
        if not animated:
            raise RetargetError(f"No animated skeleton in {os.path.basename(source)}")
        src = max(animated, key=lambda ob: len(ob.data.bones))
        f0, f1 = (int(round(v)) for v in src.animation_data.action.frame_range)
        naming, prefix, roles = detect_skeleton(src)
        mw = np.array(src.matrix_world)
        lm = _landmarks(src.data.bones, roles, mw)
        mapped = {tb: (a, b, roles[tw], kind, calib) for tb, (a, b, tw, kind, calib) in SEGMENTS.items()
                  if a in lm and b in lm and tw in roles}
        points = {key for spec in lm.values() for key, _ in spec}
        twist = {m[2] for m in mapped.values()}
        pos, rot = _sample(src, points, twist, list(range(f0, f1 + 1)))
        bones = src.data.bones
        rest_pos = {key: (mw @ np.append(np.array(bones[key[0]].head_local if key[1] == "head"
                                                  else bones[key[0]].tail_local), 1.0))[:3] for key in points}
        rest_rot = {b: _orthonormal(mw[:3, :3] @ np.array(bones[b].matrix_local.to_3x3())) for b in twist}
        return _Clip(source, rate, naming, prefix, len(bones), lm, mapped, pos, rot, rest_pos, rest_rot)
    finally:
        _remove_new(before)
        if ad is not None:
            ad.action = rig_action
            if rig_slot is not None and getattr(ad, "action_slot", None) != rig_slot:
                try:
                    ad.action_slot = rig_slot
                except (AttributeError, TypeError, RuntimeError):
                    pass
        for ob in scene.objects:
            ob.select_set(ob in selected)
        view_layer.objects.active = active
        scene.frame_set(frame)


def _timing(clip: _Clip, start: float, end: float, loop: bool, fps: float):
    """Clip times to sample, the frame count and the loop found (or None).

    Loops get one extra sample, the cycle's end, which closes it.
    """
    feat = clip.features(clip.up_fix)
    steps = np.linalg.norm(np.diff(feat, axis=0), axis=1)
    if len(steps) > 2 and steps[0] > max(8 * float(np.median(steps[1:])), 0.05 * math.sqrt(feat.shape[1])):
        clip.first = 1  # a bind pose in front of the motion (the CMU BVH files start with a T-pose)
    duration = clip.duration
    if duration <= 0:
        raise RetargetError(f"{os.path.basename(clip.file)} has no motion (one frame)")
    t0 = max(0.0, float(start))
    t1 = duration if not end or end <= 0 else min(float(end), duration)
    if t0 >= duration or t1 - t0 < (LOOP_MIN if loop else 0.0):
        raise RetargetError(f"Clip range {t0:g}-{t1:g} s is empty or too short (the clip lasts {duration:.2f} s"
                            + (f"; a loop needs at least {LOOP_MIN:g} s" if loop else "") + ")")
    if not loop:
        count = int(math.floor((t1 - t0) * fps + 1e-6)) + 1
        return t0 + np.arange(count) / fps, count, None
    i0 = clip.first + int(math.ceil(t0 * clip.rate - 1e-6))
    i1 = clip.first + int(math.floor(t1 * clip.rate + 1e-6))
    hip_line = (clip.at("hip.L") - clip.at("hip.R"))[i0:i1 + 1] @ clip.up_fix.T
    window = clip.features(_rz(-math.pi / 2 - _heading(hip_line)) @ clip.up_fix)[i0:i1 + 1]
    found = _find_loop(window, clip.rate, int(round(LOOP_MIN * clip.rate)), int(round(LOOP_MAX * clip.rate)))
    if found is None:
        raise RetargetError(f"No loop of {LOOP_MIN:g}-{LOOP_MAX:g} s fits in {t0:g}-{t1:g} s")
    s, length, error = found
    loop_start, loop_len = (i0 - clip.first + s) / clip.rate, length / clip.rate
    count = max(2, int(round(loop_len * fps)))  # the cycle is fitted to whole frames
    info = {"start": round(loop_start, 3), "end": round(loop_start + loop_len, 3), "seconds": round(loop_len, 3),
            "frames": count, "error": round(error, 4), "speed": round(loop_len / (count / fps), 3)}
    return loop_start + np.arange(count + 1) * loop_len / count, count, info


# ---------------------------------------------------------------- solving the pose

def _solve(clip: _Clip, rig, t: np.ndarray, count: int, loop: bool, in_place: bool, fps: float):
    """matrix_basis per rig bone at clip times t, the per-cycle travel, and a report."""
    P, R = clip.resample(t)
    n = len(t)

    # Face -Y by the average hip line. When the hips swing or turn (a roundhouse kick, a spin,
    # a fall with a half turn), face -Y with the stance the move starts from instead.
    left = (clip.at("hip.L", P) - clip.at("hip.R", P))[:count] @ clip.up_fix.T
    facing = np.stack([left[:, 1], -left[:, 0]], axis=1)
    heading_from = "average"
    if np.linalg.norm(facing.sum(axis=0)) < 0.95 * np.linalg.norm(facing, axis=1).sum():
        left, heading_from = left[:max(1, int(round(0.25 * fps)))], "start"
    turn = -math.pi / 2 - _heading(left)
    canon = _rz(turn) @ clip.up_fix
    C = {key: v @ canon.T for key, v in P.items()}

    mw = np.array(rig.matrix_world)
    mw_inv = np.linalg.inv(mw)
    rig_rot = _orthonormal(mw[:3, :3])
    rig_scale = float(np.linalg.norm(mw[:3, 0]))
    order = [b.name for b in sorted(rig.data.bones, key=_depth)]
    bones = {b.name: _TargetBone(b, mw) for b in rig.data.bones}
    leg = float(np.mean([bones[f"thigh.{s}"].length + bones[f"shin.{s}"].length for s in "LR"])) * rig_scale
    ratio = leg / clip.leg

    # Rotations: aim each bone along its source segment, twisted like the source bone. Calibrated
    # bones turn by the source segment's change from its rest pose instead (see SEGMENTS).
    rot_arm = {}
    for tb, (a, b, tw, kind, calib) in clip.mapped.items():
        if tb not in bones:
            continue
        bone = bones[tb]
        ref = np.einsum("nij,j->ni", R[tw], clip.rest_rot[tw].T @ clip.ref_world[kind]) @ canon.T
        f_src = _frames(clip.at(b, C) - clip.at(a, C), ref)
        if calib:
            f_rest = _frames(clip.canon_rest @ (clip.rest(b) - clip.rest(a)), REFS[kind])[0]
        else:
            f_rest = _frames(bone.rest_world[:, 1], REFS[kind])[0]
        rot_arm[tb] = rig_rot.T @ (f_src @ f_rest.T @ bone.rest_world)

    # Root motion: the mid-hip path, scaled by the leg-length ratio.
    mid = (clip.at("hip.L", C) + clip.at("hip.R", C)) / 2
    rel = mid - mid[0]
    travel = np.zeros(3)
    if loop:  # make the cycle close: remove its drift (all of it in place, else keep it as travel)
        drift = rel[count].copy()
        ramp = (np.arange(n) / count)[:, None]
        rel[:, 2:] -= ramp * drift[2:]
        if in_place:
            rel[:, :2] -= ramp * drift[:2]
            rel[:, :2] -= rel[:count, :2].mean(axis=0)
        else:
            travel[:2] = drift[:2] * ratio
    elif in_place:
        rel[:, :2] = 0.0
    hips_world = bones["hips"].head_world + rel * ratio

    def solve(lift):
        head = np.c_[hips_world + np.c_[np.zeros((n, 2)), lift], np.ones(n)] @ mw_inv.T
        return _fk(order, bones, rot_arm, head[:, :3], n)

    # Feet on the floor: the lowest ankle (or toe) sits at its rest height.
    pose, _ = solve(np.zeros(n))
    clearance = [_tail_world(pose, bones[bn], mw)[:, 2] - bones[bn].tail_world[2]
                 for s in "LR" for bn in (f"shin.{s}", f"foot.{s}") if bn in bones]
    lift = np.full(n, -float(np.percentile(np.min(clearance, axis=0)[:count], 5)))
    # The torso and head never sink below the floor (falls, lying down, big chibi heads).
    clamp_pts, floor = _clamp_points(rig, mw_inv, bones["hips"].head_world[2])
    lifted = 0.0
    if clamp_pts:
        pose, _ = solve(lift)
        lowest = np.full(n, np.inf)
        for bn, pts in clamp_pts.items():
            skin = np.einsum("ij,njk,kl->nil", mw, pose[bn], np.linalg.inv(bones[bn].rest))
            lowest = np.minimum(lowest, (skin[:, 2, :] @ pts.T).min(axis=1))
        extra = np.maximum(0.0, floor - lowest)
        if loop:
            extra[count] = extra[0]
        lifted = float(extra.max())
        lift = lift + extra
    _, basis = solve(lift)
    if loop and travel.any():  # per-cycle travel in the hips' own frame, for the closing key
        travel = bones["hips"].rest[:3, :3].T @ (rig_rot.T @ travel) / rig_scale
    report = {
        "mapped_bones": [b for b in SEGMENTS if b in clip.mapped and b in bones],
        "missing_bones": [b for b in TARGET_BONES if b not in bones],
        "unmapped_bones": [b for b in SEGMENTS if b in bones and b not in clip.mapped],
        "heading_turned_deg": round(math.degrees(math.atan2(math.sin(turn), math.cos(turn))), 1),
        "heading_from": heading_from, "scale": round(ratio, 5), "lifted_max": round(lifted, 4),
    }
    return order, basis, travel, report


def _key_action(rig, name: str, order: list, basis: dict, count: int, loop: bool, travel: np.ndarray):
    """Write the keys as a new action (replacing one with the same name); returns (action, replaced, keys)."""
    stale = bpy.data.actions.get(name)
    action, bag = _new_action(rig, name + ".retarget" if stale else name)
    paths = set()
    for bn in order:
        mats = basis[bn][:, :3, :3].copy()
        locs = basis[bn][:, :3, 3].copy()
        if loop:  # the closing key repeats the first pose exactly
            mats[count] = mats[0]
            locs[count] = locs[0] + (travel if bn == "hips" else 0)
        path, channels = _rotation_keys(rig.pose.bones[bn], mats)
        paths.add(path)
        esc = bpy.utils.escape_identifier(bn)
        for i, values in enumerate(channels):
            _write_curve(action, bag, f'pose.bones["{esc}"].{path}', i, bn, values, loop, False)
        if bn in ("root", "hips") or np.abs(locs).max() > 1e-6:
            offset = loop and bn == "hips" and bool(travel.any())
            for i in range(3):
                _write_curve(action, bag, f'pose.bones["{esc}"].location', i, bn, locs[:, i], loop, offset)
    # Loops follow auto_rig.py: cyclic, with the closing key inside the range (sprite_render drops it).
    action.frame_range = (1, count + 1) if loop else (1, count)
    action.use_cyclic = bool(loop)
    if stale:
        for ob in bpy.data.objects:
            if ob.animation_data and ob.animation_data.action == stale:
                ob.animation_data.action = action
        bpy.data.actions.remove(stale)
        action.name = name
    names = {"rotation_quaternion": "quaternion", "rotation_euler": "euler"}
    keys = names.get(paths.pop(), "mixed") if len(paths) == 1 else "mixed"
    return action, bool(stale), keys


# ---------------------------------------------------------------- main entry

def add_animation(source: str, name: str = "", start: float = 0.0, end: float = 0.0, loop: bool = False,
                  in_place: bool = False, fps: float = 0.0, rig_name: str = "") -> dict:
    """Retarget `source` onto the scene's character rig as action `name`; returns a summary dict."""
    source = os.path.abspath(source)
    if not os.path.isfile(source):
        raise RetargetError(f"Motion file {source} not found")
    name = (name or os.path.splitext(os.path.basename(source))[0]).strip()[:63]
    rig = find_rig(rig_name)
    scene = bpy.context.scene
    scene_fps = scene.render.fps / scene.render.fps_base
    fps = float(fps) if fps and fps > 0 else scene_fps
    warnings = []
    if abs(fps - scene_fps) > 1e-3:
        warnings.append(f"sampled at {fps:g} fps but the scene runs at {scene_fps:g} fps: sprites play at the "
                        f"scene rate, so use --fps {scene_fps:g} (or change the scene's frame rate)")

    clip = _read_clip(source, rig)
    t, count, loop_info = _timing(clip, start, end, loop, fps)
    if loop_info and loop_info["error"] > LOOP_WARN:
        warnings.append(f"no clean cycle in this clip (loop error {loop_info['error']:.2f}), so the loop will "
                        "pop; pick a cyclic clip or narrow start/end to the steady part")
    order, basis, travel, report = _solve(clip, rig, t, count, loop, in_place, fps)
    action, replaced, keys = _key_action(rig, name, order, basis, count, loop, travel)
    return {
        "action": action.name, "replaced": replaced, "frames": count, "fps": fps,
        "seconds": round(count / fps, 3), "loop": loop_info, "in_place": bool(in_place),
        "clip": None if loop else {"start": round(float(t[0]), 3), "end": round(float(t[-1]), 3)},
        "source": {"file": os.path.basename(source), "naming": clip.naming, "prefix": clip.prefix,
                   "fps": round(clip.rate, 3), "seconds": round(clip.duration, 3), "bones": clip.bone_count,
                   "bind_pose_dropped": bool(clip.first)},
        **report, "rotation_keys": keys, "rig": rig.name, "warnings": warnings,
    }


def _save(path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
    return path


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Retarget a BVH or Mixamo FBX clip onto the auto-rig skeleton.")
    parser.add_argument("--source", required=True, help=".bvh or .fbx motion file")
    parser.add_argument("--name", default="", help="action name (default: the file name)")
    parser.add_argument("--start", type=float, default=0.0, help="clip start in seconds")
    parser.add_argument("--end", type=float, default=0.0, help="clip end in seconds (0 = the end)")
    parser.add_argument("--loop", action="store_true", help="cut the most seamless 0.4-1.6 s cycle")
    parser.add_argument("--in-place", action="store_true", help="remove horizontal travel")
    parser.add_argument("--fps", type=float, default=0.0, help="frame rate (default: the scene's)")
    parser.add_argument("--rig", default="", help=f"armature object (default: {RIG_NAME})")
    parser.add_argument("--out", default="", help="where to save the .blend (default: overwrite the opened file)")
    args = parser.parse_args(argv)
    try:
        result = add_animation(args.source, args.name, args.start, args.end, args.loop, args.in_place,
                               args.fps, args.rig)
        out = args.out or bpy.data.filepath
        if not out:
            raise RetargetError("No --out given and no .blend is open")
        result["blend"] = _save(out)
    except RetargetError as e:
        print("RETARGET_ERROR " + str(e), flush=True)
        sys.exit(1)
    print("RETARGET_RESULT " + json.dumps(result), flush=True)
