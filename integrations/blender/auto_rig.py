"""Humanoid auto-rig and animation library for generated characters.

Image-to-3D models (TRELLIS.2, Hunyuan3D, ...) come out as one dense,
unrigged mesh. This script gives one a skeleton and a set of in-place
animations ready for sprite_render.py:

1. imports the model, joins it into one mesh, turns it to face -Y, rests it
   on z=0 at height 1 and decimates it (UVs and texture are kept);
2. finds the joints on the model's own front silhouette and geometry: where
   the legs separate (the crotch, or the hem of a robe), the ankles and toes,
   the neck, the hands and the armpits. When a pose detector found a complete
   skeleton on a front render (--front-render, then --keypoints), its hips,
   knees, ankles and arm joints replace the estimates. Pose detectors are
   accurate on realistic proportions but usually fail on chibi ones, which
   is why the silhouette comes first;
3. builds the skeleton: root, hips, chest, head, thighs, shins, feet, upper
   arms, forearms and hands;
4. skins it with Blender's heat weights, computed on a watertight voxel copy
   of the mesh (generated meshes are full of holes and loose pieces that
   make heat weighting fail on the original) and transferred back, then
   limited by anatomy with soft edges so robes and skirts stretch instead of
   tearing. Distance weights are the fallback;
5. adds in-place animations posed with forward kinematics, with the lowest
   foot kept on the ground: Idle, Walk and Run (loops) and Attack, Cast, Hit,
   Death and Jump.

    blender -b -P auto_rig.py -- --model=hero.glb --out=hero.blend
    blender -b -P auto_rig.py -- --model=hero.glb --front-render=front.png
    blender -b -P auto_rig.py -- --model=hero.glb --out=hero.blend --keypoints=pose.json

The model should stand in an A- or T-pose, facing -Y (or pass --front-axis).
When a part follows the wrong bone, look at the skeleton overlay written next
to the .blend and pass --landmarks: heights as fractions of the model height
(hips, knee, ankle, chest, neck, shoulder, hand, head_top) and distances from
the centre line as fractions of the height (leg_x, shoulder_x, hand_x).
This script resets the open scene; run it headless.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

FPS = 12
ANIMATIONS = ("Idle", "Walk", "Run", "Attack", "Cast", "Hit", "Death", "Jump")
LOOPS = {"Idle", "Walk", "Run"}
LANDMARK_KEYS = ("ankle", "knee", "hips", "chest", "neck", "head_top", "shoulder", "hand",
                 "leg_x", "shoulder_x", "hand_x")
FRONT_AXES = {"-Y": 0.0, "+Y": 180.0, "+X": -90.0, "-X": 90.0}
POSE_NAMES = ("nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rknee",
              "Rank", "Lhip", "Lknee", "Lank", "Reye", "Leye", "Rear", "Lear")
SIDES = (("L", 1.0), ("R", -1.0))  # .L is the character's left, +X


class RigError(Exception):
    pass


# ---------------------------------------------------------------- model

def _import(path: str) -> list:
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path) as (src, dst):
            dst.objects = src.objects
        for ob in dst.objects:
            if ob is not None:
                bpy.context.scene.collection.objects.link(ob)
    else:
        raise RigError(f"unsupported model type {ext}")
    return [ob for ob in bpy.data.objects if ob not in before]


def _single_mesh(objects: list):
    meshes = [ob for ob in objects if ob.type == "MESH"]
    if not meshes:
        raise RigError("the model has no mesh")
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


def _vertices(ob) -> np.ndarray:
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
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


def load_character(model: str, front_axis: str = "-Y", height: float = 1.0):
    """Import a model into an empty scene as one mesh facing -Y, feet on z=0."""
    if front_axis not in FRONT_AXES:
        raise RigError(f"front axis must be one of {', '.join(FRONT_AXES)}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = _single_mesh(_import(model))
    if FRONT_AXES[front_axis]:
        body.rotation_euler = (0.0, 0.0, math.radians(FRONT_AXES[front_axis]))
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    _normalize(body, height)
    return body


def _decimate(body, target_faces: int) -> None:
    faces = len(body.data.polygons)
    if target_faces and faces > target_faces:
        mod = body.modifiers.new("Decimate", "DECIMATE")
        mod.ratio = target_faces / faces
        bpy.ops.object.modifier_apply(modifier=mod.name)
    # Generated meshes are often split per triangle; welding gives the
    # weights a connected surface to spread over.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0005)
    bpy.ops.object.mode_set(mode="OBJECT")


# ---------------------------------------------------------------- front renders

def _front_camera(scene, body, res_x: int, res_y: int, margin: float = 1.06):
    co = _vertices(body)
    lo, hi = co.min(axis=0), co.max(axis=0)
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    pixel = max((hi[0] - lo[0]) * margin / res_x, (hi[2] - lo[2]) * margin / res_y)
    data = bpy.data.cameras.new("AutoRigFront")
    data.type = "ORTHO"
    data.ortho_scale = pixel * max(res_x, res_y)
    data.clip_start, data.clip_end = 0.01, 100.0
    cam = bpy.data.objects.new("AutoRigFront", data)
    scene.collection.objects.link(cam)
    cam.location = (cx, lo[1] - 5.0, cz)
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = res_x, res_y
    scene.render.resolution_percentage = 100
    return cam, {"center": [float(cx), float(cz)], "pixel": float(pixel), "size": [res_x, res_y]}


def render_front(body, path: str, height_px: int = 1024, look: str = "eevee") -> dict:
    """Orthographic front view (3:4) of the character.

    look: "eevee" (lit, for pose detectors), "workbench" (textured, fast) or
    "mask" (white on transparent). Returns the mapping between pixels and
    world X/Z: the camera centre, the size of a pixel and the image size.
    """
    scene = bpy.context.scene
    res_y = int(height_px)
    res_x = int(round(res_y * 0.75))
    cam, mapping = _front_camera(scene, body, res_x, res_y)
    created = [cam]
    world_before = scene.world
    try:
        settings = scene.render.image_settings
        settings.file_format = "PNG"
        if look == "mask":
            scene.render.engine = "BLENDER_WORKBENCH"
            shading = scene.display.shading
            shading.light, shading.color_type = "FLAT", "SINGLE"
            shading.single_color = (1.0, 1.0, 1.0)
            scene.render.film_transparent = True
            settings.color_mode = "RGBA"
        else:
            world = bpy.data.worlds.new("AutoRigFront")
            created.append(world)
            world.color = (0.42, 0.42, 0.42)
            scene.world = world
            scene.render.film_transparent = False
            settings.color_mode = "RGB"
            scene.view_settings.view_transform = "Standard"
            if look == "eevee":
                scene.render.engine = "BLENDER_EEVEE"
                try:
                    world.use_nodes = True
                    background = world.node_tree.nodes.get("Background")
                    background.inputs[0].default_value = (0.42, 0.42, 0.42, 1.0)
                except (AttributeError, TypeError):
                    pass
                sun = bpy.data.objects.new("AutoRigSun", bpy.data.lights.new("AutoRigSun", "SUN"))
                created += [sun, sun.data]
                scene.collection.objects.link(sun)
                sun.data.energy = 3.0
                sun.rotation_euler = (math.radians(50), 0.0, math.radians(-30))
            else:
                scene.render.engine = "BLENDER_WORKBENCH"
                shading = scene.display.shading
                shading.light, shading.color_type = "STUDIO", "TEXTURE"
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
    finally:
        scene.world = world_before
        scene.camera = None
        cam_data = cam.data
        for block in created:
            if isinstance(block, bpy.types.Object):
                bpy.data.objects.remove(block, do_unlink=True)
            elif isinstance(block, bpy.types.World):
                bpy.data.worlds.remove(block)
            elif isinstance(block, bpy.types.Light):
                bpy.data.lights.remove(block)
        bpy.data.cameras.remove(cam_data)
        scene.render.film_transparent = False
    return dict(mapping, path=os.path.abspath(path), look=look)


def _read_image(path: str) -> np.ndarray:
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[::-1]  # row 0 at the top


# ---------------------------------------------------------------- silhouette

def _runs(row: np.ndarray) -> np.ndarray:
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1)


class Silhouette:
    """A front mask with helpers in world units (X across, Z up)."""

    def __init__(self, mask: np.ndarray, mapping: dict):
        self.mask = mask
        self.rows, self.cols = mask.shape
        self.cx, self.cz = mapping["center"]
        self.p = mapping["pixel"]
        self.z = self.cz + (self.rows / 2 - np.arange(self.rows) - 0.5) * self.p
        runs = [_runs(mask[r]) for r in range(self.rows)]
        # Ignore specks a pixel or two wide (laces, stray triangles).
        self.solid = [r[(r[:, 1] - r[:, 0]) >= 2] for r in runs]
        filled = np.flatnonzero(mask.any(axis=1))
        self.top, self.bottom = int(filled.min()), int(filled.max())

    def edge(self, c: float) -> float:
        """World x of the boundary on the left of column c."""
        return self.cx + (c - self.cols / 2) * self.p

    def row(self, z: float) -> int:
        return int(np.clip(round((self.cz - z) / self.p + self.rows / 2 - 0.5), 0, self.rows - 1))

    def col(self, x: float) -> int:
        return int(np.clip(round((x - self.cx) / self.p + self.cols / 2 - 0.5), 0, self.cols - 1))

    def inside(self, x: float, z: float, tol: float) -> bool:
        r0, r1 = self.row(z + tol), self.row(z - tol)
        c0, c1 = self.col(x - tol), self.col(x + tol)
        return bool(self.mask[r0:r1 + 1, c0:c1 + 1].any())

    def central_run(self, r: int, x: float):
        runs = self.solid[r]
        if not len(runs):
            return None
        c = (x - self.cx) / self.p + self.cols / 2
        hit = [run for run in runs if run[0] <= c < run[1]]
        if hit:
            return hit[0]
        return min(runs, key=lambda run: min(abs(run[0] - c), abs(run[1] - c)))

    def extent(self, r: int):
        runs = self.solid[r]
        if not len(runs):
            return None
        return self.edge(runs[0][0]), self.edge(runs[-1][1])


def _bump(u: float, centre: float, width: float) -> float:
    """Gaussian bump on a looping 0-1 phase."""
    d = (u - centre + 0.5) % 1.0 - 0.5
    return math.exp(-(d / width) ** 2)


def _depth(co: np.ndarray, x: float, z: float, rx: float, rz: float, default: float = 0.0) -> float:
    """Middle of the mesh's front-to-back extent around (x, z)."""
    sel = (np.abs(co[:, 0] - x) < rx) & (np.abs(co[:, 2] - z) < rz)
    if sel.sum() < 8:
        return default
    ys = co[sel, 1]
    return float((np.percentile(ys, 5) + np.percentile(ys, 95)) / 2)


def _settle_depths(J: dict, h: float) -> None:
    """Both sides share one depth, kept near the body's middle: robes, hair
    and capes make the measured front-to-back centre wander."""
    spine_y = J["spine"].y
    for name in ("hip", "knee", "ankle", "toe", "heel_ground", "toe_ground",
                 "shoulder", "elbow", "wrist", "hand"):
        y = (J[f"{name}.L"].y + J[f"{name}.R"].y) / 2
        J[f"{name}.L"].y = J[f"{name}.R"].y = y
    limits = (("hip", J["hips"].y, 0.03), ("knee", None, 0.04), ("shoulder", spine_y, 0.04))
    for name, ref, tol in limits:
        if ref is None:
            ref = J["hip.L"].y
        for side, _ in SIDES:
            J[f"{name}.{side}"].y = float(np.clip(J[f"{name}.{side}"].y, ref - tol * h, ref + tol * h))


def _arm_line(sil: Silhouette, x_c: float, s: float, hand_off: float, hand_z: float, shoulder_off: float,
              h: float):
    """Height of an arm's centre line against distance from the body centre,
    traced column by column from the hand inwards, as polyfit coefficients;
    None when the trace is short or crooked (the arm merges into something)."""
    c = sil.col(x_c + s * (hand_off - 3 * sil.p))
    pts, extents, prev = [], [], None
    while True:
        off = s * (sil.edge(c + 0.5) - x_c)
        if off <= shoulder_off or not 0 <= c < sil.cols:
            break
        runs = _runs(sil.mask[:, c])
        if prev is None:
            target = sil.row(hand_z)
            cand = [r for r in runs if r[0] - 3 <= target < r[1] + 3]
        else:
            cand = [r for r in runs if r[0] < prev[1] and r[1] > prev[0]]
        if not cand:
            break
        run = max(cand, key=lambda r: min(r[1], prev[1]) - max(r[0], prev[0]) if prev is not None else r[1] - r[0])
        extent = run[1] - run[0]
        if len(extents) >= 4 and extent > 1.8 * np.median(extents):
            break  # the arm ran into the body, hair or a sleeve
        extents.append(extent)
        pts.append((off, (sil.z[run[0]] + sil.z[run[1] - 1]) / 2))
        prev = run
        c -= int(s)
    if len(pts) < 8 or pts[0][0] - pts[-1][0] < 0.35 * hand_off:
        return None
    offs, zs = np.array(pts).T
    line = np.polyfit(offs, zs, 1)
    if np.sqrt(np.mean((np.polyval(line, offs) - zs) ** 2)) > 0.015 * h:
        return None
    return line


def measure(sil: Silhouette, co: np.ndarray, h: float) -> tuple[dict, dict, list]:
    """Joint estimates from the silhouette and the vertices.

    Returns (joints, info, notes): joints in world space, info with the
    measurements the weights and animations need.
    """
    notes = []

    # Legs: the gap between the feet, followed upwards until it closes at
    # the crotch (or the hem of a robe).
    gap_rows, prev = [], None
    wide = 0.02 * h / sil.p
    for r in range(sil.bottom, sil.top - 1, -1):
        z, runs = sil.z[r], sil.solid[r]
        gaps = [(runs[i][1], runs[i + 1][0], i) for i in range(len(runs) - 1)]
        if prev is None:
            if z > 0.12 * h:
                break
            # A gap between two feet, not between bits of one sole.
            cand = [g for g in gaps if abs(sil.edge((g[0] + g[1]) / 2)) < 0.1 * h
                    and runs[:g[2] + 1, 1].max() - runs[:g[2] + 1, 0].min() >= wide
                    and runs[g[2] + 1:, 1].max() - runs[g[2] + 1:, 0].min() >= wide]
            if not cand:
                continue
            centre = sil.cols / 2
        else:
            cand = [g for g in gaps if g[0] <= prev[1] and g[1] >= prev[0]]
            if not cand:
                if len(gap_rows) < 6 and z < 0.12 * h:  # a false start: look again higher up
                    gap_rows, prev = [], None
                    continue
                break
            centre = (prev[0] + prev[1]) / 2
        g = min(cand, key=lambda g: abs((g[0] + g[1]) / 2 - centre))
        gap_rows.append((float(z), runs[g[2]], runs[g[2] + 1], g))
        prev = g
    legs_found = len(gap_rows) >= 4
    if legs_found:
        gap_top = gap_rows[-1][0]
        lower = [row for row in gap_rows if row[0] <= gap_top * 0.7 + 0.01 * h] or gap_rows
        x_c = float(np.median([(sil.edge(g[0]) + sil.edge(g[1])) / 2 for _, _, _, g in lower]))
    else:
        gap_top, x_c = 0.0, 0.0
        notes.append("the feet aren't apart in the front view; leg positions are guesses")

    # Feet: the ankle is where the foot's front-to-back depth drops to the
    # leg's; the toe is the most forward point near the floor.
    feet = {}
    for side, s in SIDES:
        leg_rows = [(z, (right if s > 0 else left)) for z, left, right, _ in gap_rows]
        if legs_found:
            low = [run for z, run in leg_rows if z < min(0.15 * h, gap_top)] or [run for _, run in leg_rows]
            leg_x = float(np.median([(sil.edge(a) + sil.edge(b)) / 2 for a, b in low]))
            leg_w = float(np.median([sil.edge(b) - sil.edge(a) for a, b in low]))
        else:
            leg_x, leg_w = x_c + s * 0.065 * h, 0.06 * h
        zmax = min(0.2 * h, gap_top) if legs_found and gap_top > 0.06 * h else 0.12 * h
        sel = (co[:, 2] < zmax) & (s * (co[:, 0] - x_c) > 0) & (np.abs(co[:, 0] - leg_x) < max(leg_w, 0.03 * h))
        pts = co[sel]
        ankle_z, toe, heel_y, ankle_y = 0.045 * h, None, None, None
        if len(pts) > 60:
            step = 0.005 * h
            bins = np.arange(0.0, zmax, step)
            depth = np.full(len(bins), np.nan)
            for i, z0 in enumerate(bins):
                ys = pts[(pts[:, 2] >= z0) & (pts[:, 2] < z0 + step), 1]
                if len(ys) >= 6:
                    depth[i] = np.percentile(ys, 97) - np.percentile(ys, 3)
            foot_d = np.nanmedian(depth[:3]) if np.isfinite(depth[:3]).any() else np.nan
            top = bins >= zmax - 0.03 * h
            leg_d = np.nanmedian(depth[top]) if np.isfinite(depth[top]).any() else np.nan
            if np.isfinite(foot_d) and np.isfinite(leg_d) and foot_d > 1.2 * leg_d:
                limit = leg_d + 0.4 * (foot_d - leg_d)
                for i in range(3, len(bins)):
                    if np.isfinite(depth[i]) and depth[i] < limit:
                        ankle_z = float(bins[i] + step / 2)
                        break
            ankle_z = float(np.clip(ankle_z, 0.025 * h, min(0.12 * h, max(0.6 * gap_top, 0.04 * h))))
            near_floor = pts[pts[:, 2] < ankle_z]
            if len(near_floor) > 20:
                toe_y = float(np.percentile(near_floor[:, 1], 2))
                heel_y = float(np.percentile(near_floor[:, 1], 98))
                toe_x = float(np.median(near_floor[near_floor[:, 1] < toe_y + 0.015 * h, 0]))
                toe = (toe_x, toe_y)
            ankle_y = _depth(pts, leg_x, ankle_z, max(leg_w, 0.03 * h), 0.01 * h, default=0.0)
        feet[side] = {"x": leg_x, "w": leg_w, "ankle_z": ankle_z, "toe": toe, "heel_y": heel_y,
                      "ankle_y": ankle_y}
    ankle_z = (feet["L"]["ankle_z"] + feet["R"]["ankle_z"]) / 2

    # Leg lines through the middle of each leg, above the ankles.
    lines = {}
    for side, s in SIDES:
        pts = []
        for z, left, right, g in gap_rows:
            if ankle_z + 0.03 * h < z < gap_top - 0.015 * h:
                run = right if s > 0 else left
                pts.append((z, (sil.edge(run[0]) + sil.edge(run[1])) / 2))
        if len(pts) >= 8 and pts[-1][0] - pts[0][0] > 0.08 * h:
            zs, xs = np.array(pts).T
            lines[side] = np.polyfit(zs, xs, 1)
        else:
            lines[side] = np.array([0.0, feet[side]["x"]])

    # Robe or skirt: the silhouette widens sharply where the legs join.
    skirt = False
    if legs_found:
        below, above = [], []
        for z, left, right, _ in gap_rows:
            if gap_top - 0.04 * h <= z <= gap_top - 0.005 * h:
                below.append(sil.edge(right[1]) - sil.edge(left[0]))
        for r in range(sil.row(gap_top + 0.04 * h), sil.row(gap_top + 0.005 * h) + 1):
            run = sil.central_run(r, x_c)
            if run is not None:
                above.append(sil.edge(run[1]) - sil.edge(run[0]))
        if below and above and np.median(above) > 1.3 * np.median(below):
            skirt = True
            notes.append("the legs disappear into a robe or skirt at "
                         f"{gap_top / h:.2f} of the height; hips placed by proportion")

    # Neck: the first strong narrowing below the widest part of the head,
    # scanning down from the top.
    width = np.zeros(sil.rows)
    for r in range(sil.top, sil.bottom + 1):
        ext = sil.extent(r)
        if ext:
            width[r] = ext[1] - ext[0]
    smooth = width.copy()
    for r in range(sil.top + 2, sil.bottom - 1):
        smooth[r] = np.median(width[r - 2:r + 3])
    floor_z = (gap_top if legs_found else 0.2 * h) + 0.1 * h
    neck_z, widest, r = None, 0.0, sil.top
    while r <= sil.row(floor_z):
        widest = max(widest, smooth[r])
        if 0 < smooth[r] < 0.6 * widest:
            best, lowest = r, smooth[r]
            while r + 1 <= sil.row(floor_z) and smooth[r + 1] <= lowest * 1.15:
                r += 1
                if smooth[r] < lowest:
                    best, lowest = r, smooth[r]
            if r + 1 <= sil.bottom and smooth[min(r + 1, sil.bottom)] > lowest:
                neck_z = float(sil.z[best])
            break
        r += 1
    if neck_z is None:
        notes.append("no neck found in the silhouette; head placed by proportion")

    # Hands: the farthest point on each side between the legs and the neck.
    lo_z = (gap_top if legs_found else 0.2 * h) + 0.02 * h
    hi_z = (neck_z if neck_z is not None else 0.92 * h) - 0.02 * h
    hands = {}
    for side, s in SIDES:
        best = (0.0, None)
        for r in range(sil.row(hi_z), sil.row(lo_z) + 1):
            ext = sil.extent(r)
            if not ext:
                continue
            reach = (ext[1] - x_c) if s > 0 else (x_c - ext[0])
            if reach > best[0]:
                best = (reach, float(sil.z[r]))
        hands[side] = best
    hand_off = (hands["L"][0] + hands["R"][0]) / 2
    if hands["L"][1] is None or hands["R"][1] is None:
        raise RigError("could not find the arms in the front view; is the model a humanoid facing -Y?")
    hand_z = (hands["L"][1] + hands["R"][1]) / 2

    # Torso width between the legs and the hands.
    z0, z1 = lo_z, hand_z - 0.01 * h
    if z1 - z0 < 0.03 * h:
        z0, z1 = lo_z, lo_z + 0.1 * h
    halves = []
    for r in range(sil.row(z1), sil.row(z0) + 1):
        run = sil.central_run(r, x_c)
        if run is not None:
            halves.append((sil.edge(run[1]) - sil.edge(run[0])) / 2)
    waist_half = float(np.percentile(halves, 25)) if halves else 0.1 * h

    # Armpits: the top of the gap between each arm and the body.
    armpits = {}
    for side, s in SIDES:
        def separated(r):
            run = sil.central_run(r, x_c)
            if run is None:
                return False
            others = sil.solid[r]
            return any((o[0] >= run[1]) if s > 0 else (o[1] <= run[0]) for o in others)
        start = None
        r_hand = sil.row(hands[side][1])
        for d in range(0, int(0.04 * h / sil.p) + 1):
            for r in (r_hand - d, r_hand + d):
                if 0 <= r < sil.rows and separated(r):
                    start = r
                    break
            if start is not None:
                break
        if start is None:
            armpits[side] = None
            continue
        r, miss = start, 0
        top_r = start
        while r > sil.top and miss <= 2:
            r -= 1
            if separated(r):
                top_r, miss = r, 0
            else:
                miss += 1
        armpits[side] = float(sil.z[top_r])
    pits = [z for z in armpits.values() if z is not None]
    armpit_z = float(np.mean(pits)) if pits else None

    # Shoulders: where the arm's centre line, traced in from the hand, meets
    # the body. When the arm can't be traced cleanly (sleeves merging with
    # hair, arms touching the body), by proportion from the neck and hands.
    shoulder_off = float(np.clip(0.85 * waist_half, 0.04 * h, 0.6 * hand_off))
    shoulder_z = None
    lines_found = []
    for side, s in SIDES:
        line = _arm_line(sil, x_c, s, hands[side][0], hands[side][1], shoulder_off, h)
        if line is not None:
            lines_found.append(float(np.polyval(line, shoulder_off)))
    if len(lines_found) == 2 and neck_z is not None:
        candidate = float(np.mean(lines_found))
        if hand_z - 0.02 * h <= candidate <= neck_z - 0.005 * h:
            shoulder_z = candidate
    if shoulder_z is None:
        if neck_z is not None:
            shoulder_z = neck_z - 0.18 * (neck_z - hand_z)
        elif armpit_z is not None:
            shoulder_z = armpit_z + 0.05 * h
        else:
            shoulder_z = hand_z + 0.08 * h
        if armpit_z is not None:
            shoulder_z = max(shoulder_z, armpit_z + 0.015 * h)
    if neck_z is None:
        neck_z = shoulder_z + 0.04 * h

    if skirt or not legs_found:
        hip_z = float(np.clip(0.6 * shoulder_z, gap_top + 0.04 * h, shoulder_z - 0.1 * h))
    else:
        hip_z = gap_top + 0.12 * (gap_top - ankle_z)
    spine_z = hip_z + 0.42 * (neck_z - hip_z)

    J = {}
    y_hips = _depth(co, x_c, hip_z, 0.04 * h, 0.03 * h)
    y_spine = _depth(co, x_c, spine_z, 0.04 * h, 0.03 * h, default=y_hips)
    y_neck = float(np.clip(_depth(co, x_c, neck_z, 0.03 * h, 0.02 * h, default=y_spine),
                           y_spine - 0.03 * h, y_spine + 0.03 * h))
    top_z = float(co[:, 2].max())
    J["hips"] = Vector((x_c, y_hips, hip_z))
    J["spine"] = Vector((x_c, y_spine, spine_z))
    J["neck"] = Vector((x_c, y_neck, neck_z))
    J["top"] = Vector((x_c, y_neck, top_z * 0.985))
    leg_offsets = []
    for side, s in SIDES:
        line = lines[side]
        at = lambda z: float(np.polyval(line, z))
        f = feet[side]
        if skirt or not legs_found:
            hip_x = x_c + 0.9 * (f["x"] - x_c)
            knee_x = (hip_x + f["x"]) / 2
            ankle_x = f["x"]
        else:
            hip_x = at(gap_top - 0.02 * h)
            ankle_x = at(f["ankle_z"]) if abs(line[0]) > 0 else f["x"]
            knee_x = at((hip_z + f["ankle_z"]) / 2)
        leg_offsets.append((abs(hip_x - x_c), abs(knee_x - x_c), abs(ankle_x - x_c)))
    hip_off, knee_off, ankle_off = np.mean(leg_offsets, axis=0)
    knee_z = ankle_z + 0.5 * (hip_z - ankle_z)
    for side, s in SIDES:
        f = feet[side]
        hip = Vector((x_c + s * hip_off, 0.0, hip_z))
        knee = Vector((x_c + s * knee_off, 0.0, knee_z))
        ankle = Vector((x_c + s * ankle_off, 0.0, ankle_z))
        r = max(f["w"] * 0.6, 0.03 * h)
        hip.y = _depth(co, hip.x, hip.z, r, 0.02 * h, default=y_hips)
        knee.y = _depth(co, knee.x, knee.z, r, 0.02 * h, default=hip.y)
        ankle.y = f["ankle_y"] if f["ankle_y"] is not None else knee.y
        if f["toe"] is not None:
            toe_x, toe_y = f["toe"]
            heel_y = f["heel_y"]
        else:
            toe_x, toe_y, heel_y = ankle.x, ankle.y - 0.08 * h, ankle.y + 0.02 * h
        foot_len = max(heel_y - toe_y, 0.03 * h)
        J[f"hip.{side}"], J[f"knee.{side}"], J[f"ankle.{side}"] = hip, knee, ankle
        J[f"toe.{side}"] = Vector((toe_x, toe_y + 0.18 * foot_len, max(0.25 * ankle_z, 0.008 * h)))
        J[f"heel_ground.{side}"] = Vector((ankle.x, heel_y, 0.0))
        J[f"toe_ground.{side}"] = Vector((toe_x, toe_y, 0.0))
    for side, s in SIDES:
        shoulder = Vector((x_c + s * shoulder_off, 0.0, shoulder_z))
        tip = Vector((x_c + s * hand_off, 0.0, hand_z))
        shoulder.y = _depth(co, shoulder.x, shoulder.z, 0.03 * h, 0.03 * h, default=y_spine)
        tip.y = _depth(co, tip.x, tip.z, 0.04 * h, 0.04 * h, default=shoulder.y)
        J[f"shoulder.{side}"] = shoulder
        J[f"elbow.{side}"] = shoulder.lerp(tip, 0.42)
        J[f"wrist.{side}"] = shoulder.lerp(tip, 0.8)
        J[f"hand.{side}"] = shoulder.lerp(tip, 0.96)
    for name in ("elbow", "wrist"):
        for side, _ in SIDES:
            v = J[f"{name}.{side}"]
            v.y = _depth(co, v.x, v.z, 0.03 * h, 0.03 * h, default=v.y)
    _settle_depths(J, h)
    # The head's outermost points (a hat brim, hair) keep it above the floor
    # when a pose brings it down.
    head_pts = co[co[:, 2] > neck_z + 0.03 * h]
    if len(head_pts):
        for name, i in (("head_front", np.argmin(head_pts[:, 1])), ("head_back", np.argmax(head_pts[:, 1])),
                        ("head_left", np.argmax(head_pts[:, 0])), ("head_right", np.argmin(head_pts[:, 0]))):
            J[name] = Vector(head_pts[i])

    info = {"x_c": x_c, "gap_top": gap_top, "skirt": skirt, "legs_found": legs_found,
            "armpit_z": armpit_z, "waist_half": waist_half, "leg_w": (feet["L"]["w"] + feet["R"]["w"]) / 2,
            "sources": {"legs": "silhouette" if legs_found else "guess",
                        "arms": "silhouette", "spine": "silhouette"}}
    return J, info, notes


def apply_keypoints(J: dict, info: dict, kp: dict, sil: Silhouette, co: np.ndarray, h: float,
                    notes: list) -> None:
    """Use a pose detector's skeleton where it is complete and plausible.

    kp maps OpenPose-18 names to [x, z, confidence] in world space.
    """
    def point(name):
        v = kp.get(name)
        if not v or len(v) < 3 or v[2] < 0.3:
            return None
        x, z = float(v[0]), float(v[1])
        return Vector((x, 0.0, z)) if sil.inside(x, z, 0.02 * h) else None

    core = ("neck", "Rsho", "Lsho", "Rhip", "Lhip", "Rknee", "Lknee", "Rank", "Lank")
    missing = [n for n in core if point(n) is None]
    if missing:
        notes.append("the pose detector's skeleton is incomplete or off the model "
                     f"(no {', '.join(missing)}); joints come from the silhouette")
        return
    x_c = info["x_c"]
    used = []

    legs = {}
    for side, prefix in (("L", "L"), ("R", "R")):
        hip, knee, ankle = point(prefix + "hip"), point(prefix + "knee"), point(prefix + "ank")
        if hip.z > knee.z > ankle.z:
            ratio = (hip.z - knee.z) / (hip.z - ankle.z)
            if 0.3 < ratio < 0.7:
                legs[side] = (hip, knee, ankle)
    if len(legs) == 2:
        off = lambda i: float(np.mean([abs(legs[s][i].x - x_c) for s in legs]))
        zz = lambda i: float(np.mean([legs[s][i].z for s in legs]))
        hip_z, knee_z = zz(0), zz(1)
        ankle_z = zz(2) if abs(zz(2) - J["ankle.L"].z) < 0.04 * h else J["ankle.L"].z
        for side, s in SIDES:
            for name, i, z in (("hip", 0, hip_z), ("knee", 1, knee_z), ("ankle", 2, ankle_z)):
                v = J[f"{name}.{side}"]
                v.x, v.z = x_c + s * off(i), z
                if name != "ankle":
                    v.y = _depth(co, v.x, v.z, 0.035 * h, 0.02 * h, default=v.y)
        J["hips"].z = hip_z
        used.append("legs")

    arms = {}
    for side, prefix in (("L", "L"), ("R", "R")):
        sho, elb, wri = point(prefix + "sho"), point(prefix + "elb"), point(prefix + "wri")
        if not (sho and elb and wri):
            continue
        o = lambda v: abs(v.x - x_c)
        geo_tip = J[f"hand.{side}"]
        if (o(wri) > o(elb) > o(sho) and elb.z <= sho.z + 0.05 * h
                and abs(sho.z - J[f"shoulder.{side}"].z) < 0.08 * h
                and (wri - Vector((geo_tip.x, 0, geo_tip.z))).length < 0.15 * h):
            arms[side] = (sho, elb, wri)
    if arms:
        names = ("shoulder", "elbow", "wrist")
        off = lambda i: float(np.mean([abs(arms[s][i].x - x_c) for s in arms]))
        zz = lambda i: float(np.mean([arms[s][i].z for s in arms]))
        for side, s in SIDES:
            for i, name in enumerate(names):
                v = J[f"{name}.{side}"]
                v.x, v.z = x_c + s * off(i), zz(i)
                v.y = _depth(co, v.x, v.z, 0.03 * h, 0.03 * h, default=v.y)
            wrist, elbow, tip = J[f"wrist.{side}"], J[f"elbow.{side}"], J[f"hand.{side}"]
            if abs(tip.x - x_c) < abs(wrist.x - x_c) + 0.02 * h:
                J[f"hand.{side}"] = wrist + (wrist - elbow) * 0.35
        used.append("arms" if len(arms) == 2 else "arms (one side, mirrored)")

    nose = point("nose")
    sho_z = float(np.mean([J["shoulder.L"].z, J["shoulder.R"].z]))
    if nose is not None and not (sho_z < J["neck"].z < nose.z):
        J["neck"].z = sho_z + 0.45 * (nose.z - sho_z)
    if "legs" in used:
        J["spine"].z = J["hips"].z + 0.42 * (J["neck"].z - J["hips"].z)
    _settle_depths(J, h)
    if used:
        for part in ("legs", "arms"):
            if any(u.startswith(part) for u in used):
                info["sources"][part] = "pose detector"
        notes.append("used the pose detector's " + " and ".join(used))
    else:
        notes.append("the pose detector's limbs didn't match the model; joints come from the silhouette")


def apply_landmarks(J: dict, info: dict, marks: dict, h: float) -> None:
    """Manual overrides, as fractions of the height."""
    x_c = info["x_c"]
    for side, s in SIDES:
        if "leg_x" in marks:
            for name in ("hip", "knee", "ankle"):
                J[f"{name}.{side}"].x = x_c + s * marks["leg_x"] * h
        for key, name in (("hips", "hip"), ("knee", "knee"), ("ankle", "ankle")):
            if key in marks:
                J[f"{name}.{side}"].z = marks[key] * h
        if "shoulder" in marks:
            J[f"shoulder.{side}"].z = marks["shoulder"] * h
        if "shoulder_x" in marks:
            J[f"shoulder.{side}"].x = x_c + s * marks["shoulder_x"] * h
        if "hand" in marks:
            J[f"hand.{side}"].z = marks["hand"] * h
        if "hand_x" in marks:
            J[f"hand.{side}"].x = x_c + s * marks["hand_x"] * h
        if {"shoulder", "shoulder_x", "hand", "hand_x"} & set(marks):
            # The hand bone ends at 96% of the way to the fingertips.
            sho, hand = J[f"shoulder.{side}"], J[f"hand.{side}"]
            J[f"elbow.{side}"] = sho.lerp(hand, 0.42 / 0.96)
            J[f"wrist.{side}"] = sho.lerp(hand, 0.8 / 0.96)
    if "hips" in marks:
        J["hips"].z = marks["hips"] * h
    if "neck" in marks:
        J["neck"].z = marks["neck"] * h
    if "head_top" in marks:
        J["top"].z = marks["head_top"] * h
    if "chest" in marks:
        J["spine"].z = marks["chest"] * h
    elif {"hips", "neck"} & set(marks):
        J["spine"].z = J["hips"].z + 0.42 * (J["neck"].z - J["hips"].z)
    if marks:
        for part, keys in (("legs", ("hips", "knee", "ankle", "leg_x")),
                           ("arms", ("shoulder", "shoulder_x", "hand", "hand_x")),
                           ("spine", ("neck", "chest", "head_top", "hips"))):
            if set(keys) & set(marks):
                info["sources"][part] += " + landmarks"


def landmark_summary(J: dict, info: dict, h: float) -> dict:
    x_c = info["x_c"]
    f = lambda v: round(float(v) / h, 3)
    return {"ankle": f(J["ankle.L"].z), "knee": f(J["knee.L"].z), "hips": f(J["hips"].z),
            "chest": f(J["spine"].z), "neck": f(J["neck"].z), "head_top": f(J["top"].z),
            "shoulder": f(J["shoulder.L"].z), "hand": f(J["hand.L"].z),
            "leg_x": f(abs(J["hip.L"].x - x_c)), "shoulder_x": f(abs(J["shoulder.L"].x - x_c)),
            "hand_x": f(abs(J["hand.L"].x - x_c))}


# ---------------------------------------------------------------- skeleton

def build_armature(J: dict, h: float):
    data = bpy.data.armatures.new("CharacterRig")
    rig = bpy.data.objects.new("CharacterRig", data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = data.edit_bones

    def bone(name, head, tail, parent=None, connect=False, deform=True):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        d = b.tail - b.head
        if abs(d.z) >= max(abs(d.x), abs(d.y)):  # spine and legs: local X = +X
            b.align_roll(Vector((0, -1, 0)) if d.z > 0 else Vector((0, 1, 0)))
        elif abs(d.y) > abs(d.x):  # feet point forward: local X = +X
            b.align_roll(Vector((0, 0, -1)) if d.y < 0 else Vector((0, 0, 1)))
        else:  # arms: local Z forward
            b.align_roll(Vector((0, -1, 0)))
        b.use_deform = deform
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        return b

    root = bone("root", (0, 0, 0), (0, 0.1 * h, 0), deform=False)
    root.align_roll(Vector((0, 0, 1)))
    bone("hips", J["hips"], J["spine"], "root")
    bone("chest", J["spine"], J["neck"], "hips", True)
    bone("head", J["neck"], J["top"], "chest", True)
    for side, _ in SIDES:
        bone(f"thigh.{side}", J[f"hip.{side}"], J[f"knee.{side}"], "hips")
        bone(f"shin.{side}", J[f"knee.{side}"], J[f"ankle.{side}"], f"thigh.{side}", True)
        bone(f"foot.{side}", J[f"ankle.{side}"], J[f"toe.{side}"], f"shin.{side}", True)
        bone(f"upper_arm.{side}", J[f"shoulder.{side}"], J[f"elbow.{side}"], "chest")
        bone(f"forearm.{side}", J[f"elbow.{side}"], J[f"wrist.{side}"], f"upper_arm.{side}", True)
        bone(f"hand.{side}", J[f"wrist.{side}"], J[f"hand.{side}"], f"forearm.{side}", True)
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


# ---------------------------------------------------------------- weights

def _segment_distance(points: np.ndarray, a, b) -> np.ndarray:
    a, b = np.array(a), np.array(b)
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _ramp(v, a: float, b: float):
    """0 at a, 1 at b (either direction), linear in between."""
    return np.clip((v - a) / (b - a), 0.0, 1.0)


def _heat_weights(body, rig, names: list, voxel: float):
    """Blender's bone heat on a voxel-remeshed copy, transferred back.

    Returns (weights per body vertex, bones the heat solve left empty).
    """
    mod = body.modifiers.new("AutoRigRemesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    mod.adaptivity = 0.0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(body.evaluated_get(depsgraph))
    body.modifiers.remove(mod)
    proxy = bpy.data.objects.new("AutoRigProxy", mesh)
    bpy.context.scene.collection.objects.link(proxy)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        proxy.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        P = _vertices(proxy)
        Wp = np.zeros((len(P), len(names)))
        index = {proxy.vertex_groups[n].index: i for i, n in enumerate(names) if n in proxy.vertex_groups}
        for v in proxy.data.vertices:
            for g in v.groups:
                i = index.get(g.group)
                if i is not None:
                    Wp[v.index, i] = g.weight
        tree = KDTree(len(P))
        for i, p in enumerate(P):
            tree.insert(p, i)
        tree.balance()
        B = _vertices(body)
        W = np.zeros((len(B), len(names)))
        for i, p in enumerate(B):
            found = tree.find_n(p, 4)
            inv = np.array([1.0 / (d + voxel * 0.25) for _, _, d in found])
            W[i] = (inv[:, None] * Wp[[j for _, j, _ in found]]).sum(axis=0) / inv.sum()
        empty = [n for i, n in enumerate(names) if Wp[:, i].max() < 0.05]
        unassigned = float((Wp.sum(axis=1) < 1e-6).mean())
        return W, empty, unassigned, len(P)
    finally:
        bpy.data.objects.remove(proxy, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def skin(body, rig, J: dict, info: dict, h: float, method: str = "auto") -> str:
    """Weight the mesh to the deforming bones; returns a description."""
    co = _vertices(body)
    bones = [b for b in rig.data.bones if b.use_deform]
    names = [b.name for b in bones]
    col = {n: i for i, n in enumerate(names)}
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    x_c = info["x_c"]
    neck_z, hip_z = J["neck"].z, J["hips"].z
    shoulder_z = (J["shoulder.L"].z + J["shoulder.R"].z) / 2
    shoulder_off = abs(J["shoulder.L"].x - x_c)
    ankle_z = J["ankle.L"].z
    knee_z = J["knee.L"].z

    # Where each bone may have influence, with soft edges.
    M = np.ones((len(co), len(names)))
    spine_mask = _ramp(z, neck_z + 0.05 * h, neck_z + 0.02 * h)
    for n in ("hips", "chest"):
        M[:, col[n]] = spine_mask
    M[:, col["head"]] = _ramp(z, neck_z - 0.04 * h, neck_z - 0.01 * h)
    for side, s in SIDES:
        lateral = _ramp(s * (x - x_c), -0.02 * h, 0.02 * h)
        below_hips = _ramp(z, hip_z + 0.08 * h, hip_z + 0.03 * h)
        M[:, col[f"thigh.{side}"]] = lateral * below_hips
        M[:, col[f"shin.{side}"]] = lateral * _ramp(z, knee_z + 0.1 * h, knee_z + 0.04 * h)
        M[:, col[f"foot.{side}"]] = lateral * _ramp(z, ankle_z + 0.06 * h, ankle_z + 0.025 * h)
        arm_lateral = _ramp(s * (x - x_c), 0.55 * shoulder_off, 0.9 * shoulder_off)
        arm_height = _ramp(z, shoulder_z + 0.1 * h, shoulder_z + 0.05 * h)
        behind = _ramp(y - J[f"shoulder.{side}"].y, 0.1 * h, 0.06 * h)  # hair hanging behind the arms
        for part in ("upper_arm", "forearm", "hand"):
            M[:, col[f"{part}.{side}"]] = arm_lateral * arm_height * behind

    D = np.stack([_segment_distance(co, b.head_local, b.tail_local) for b in bones], axis=1)
    distance_W = np.exp(-(D / (0.04 * h)) ** 2) * M + 1e-12 * M

    description = "distance"
    W = None
    if method in ("auto", "heat"):
        try:
            heat, empty, unassigned, proxy_verts = _heat_weights(body, rig, names, voxel=h / 110)
        except RuntimeError as e:
            heat, empty, unassigned, proxy_verts = None, names, 1.0, 0
            description = f"distance (heat weighting failed: {e})"
        if heat is not None and len(empty) <= len(names) // 3 and unassigned < 0.2:
            W = heat
            for n in empty:  # the heat solve gave up on these bones
                W[:, col[n]] = distance_W[:, col[n]]
            description = f"heat (voxel proxy, {proxy_verts} vertices)"
            if empty:
                description += f"; distance for {', '.join(empty)}"
        elif heat is not None:
            description = (f"distance (heat weighting left {len(empty)} bones empty"
                           f"{', '.join([''] + empty[:4])})")
    if W is None:
        if method == "heat":
            raise RigError("heat weighting failed on this model: " + description)
        W = distance_W.copy()

    W = W * M
    orphan = W.sum(axis=1) < 1e-6
    W[orphan] = distance_W[orphan]
    W[W.sum(axis=1) < 1e-9, col["hips"]] = 1.0
    W /= W.sum(axis=1, keepdims=True)

    if info["skirt"]:
        # The robe below the hips follows the legs only partly, so it
        # stretches less between them.
        robe = (z > info["gap_top"]) & (z < hip_z + 0.02 * h)
        for side, _ in SIDES:
            share = W[robe, col[f"thigh.{side}"]] * 0.3
            W[robe, col[f"thigh.{side}"]] -= share
            W[robe, col["hips"]] += share

    edges = np.empty(len(body.data.edges) * 2, dtype=np.int64)
    body.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=len(co)).astype(float)
    for _ in range(3):
        acc = np.zeros_like(W)
        np.add.at(acc, edges[:, 0], W[edges[:, 1]])
        np.add.at(acc, edges[:, 1], W[edges[:, 0]])
        neighbour = np.divide(acc, degree[:, None], out=W.copy(), where=degree[:, None] > 0)
        W = (0.5 * W + 0.5 * neighbour) * M
        W[W.sum(axis=1) < 1e-9, col["hips"]] = 1.0
        W /= W.sum(axis=1, keepdims=True)

    # Everything clearly above the neck (face, hair, hats) is the head.
    head_only = (z > neck_z + 0.03 * h) & ((np.abs(x - x_c) < 0.9 * shoulder_off) | (z > shoulder_z + 0.1 * h))
    W[head_only] = 0.0
    W[head_only, col["head"]] = 1.0

    for g in list(body.vertex_groups):
        body.vertex_groups.remove(g)
    for i, n in enumerate(names):
        group = body.vertex_groups.new(name=n)
        rounded = np.round(W[:, i], 2)
        for weight in np.unique(rounded):
            if weight > 0:
                group.add(np.flatnonzero(rounded == weight).tolist(), float(weight), "REPLACE")
    for mod in list(body.modifiers):
        body.modifiers.remove(mod)
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    body.parent = rig
    owned = {n: int((W[:, i] > 0.5).sum()) for i, n in enumerate(names)}
    weak = [n for n, c in owned.items() if c == 0]
    if weak:
        description += f" (bones owning no vertices: {', '.join(weak)})"
    return description


# ---------------------------------------------------------------- posing

def _rot_between(a: Vector, b: Vector) -> Matrix:
    return a.rotation_difference(b).to_matrix()


def _spine_rot(pitch: float, roll: float, yaw: float) -> Matrix:
    """World-axis rotation: pitch leans forward (-Y), roll tips toward the
    character's left (+X), yaw turns toward the character's left."""
    return (Matrix.Rotation(math.radians(yaw), 3, "Z") @ Matrix.Rotation(math.radians(pitch), 3, "X")
            @ Matrix.Rotation(math.radians(roll), 3, "Y"))


class Poser:
    """Turns readable poses into bone rotations, with forward kinematics.

    A pose is a flat dict: "hips.pitch/roll/yaw", "chest.*", "head.*",
    "legL.flex" (thigh forward of vertical, world), "legL.spread" (added
    to the rest stance), "legL.knee", "legL.foot" (toes up from flat),
    "armL.abd" (raised sideways from hanging down), "armL.flex" (forward),
    "armL.elbow", "armL.wrist", plus "air" (0 on the ground, 1 airborne),
    "lift" (height when airborne) and "bounce" (extra height on the
    ground), both in model heights.
    """

    FORWARD = Vector((0.0, -1.0, 0.0))

    def __init__(self, rig, J: dict, h: float):
        self.rig, self.h = rig, h
        bones = rig.data.bones
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in bones}
        self.order, done = [], set()
        while len(self.order) < len(bones):
            for b in bones:
                if b.name not in done and (b.parent is None or b.parent.name in done):
                    self.order.append(b.name)
                    done.add(b.name)
        self.head = {b.name: b.head_local.copy() for b in bones}
        self.dir = {b.name: (b.tail_local - b.head_local).normalized() for b in bones}
        self.basis = {b.name: b.matrix_local.to_3x3() for b in bones}
        self.hinge = {}
        for b in bones:
            if b.name.startswith(("shin", "forearm", "hand")):
                self.hinge[b.name] = self.FORWARD.cross(self.dir[b.name]).normalized()
        self.leg_rest, self.arm_rest, self.arm_idle = {}, {}, {}
        for side, s in SIDES:
            d = self.dir[f"thigh.{side}"]
            self.leg_rest[side] = (math.degrees(math.atan2(s * d.x, -d.z)), math.degrees(math.asin(-d.y)))
            d = self.dir[f"upper_arm.{side}"]
            abd = math.degrees(math.atan2(s * d.x, -d.z))
            self.arm_rest[side] = abd
            self.arm_idle[side] = abd - 0.75 * max(0.0, abd - 15.0)
        # A big head (chibi proportions, a large hat) turns small tilts of
        # the spine into big swings, so they are toned down.
        head_share = (J["top"].z - J["neck"].z) / h
        self.tilt = float(np.clip(0.2 / max(head_share, 1e-3), 0.35, 1.0))
        leg_r = 0.35 * max(0.03 * h, abs(J["hip.L"].x - J["hip.R"].x) * 0.5)
        self.contacts = {"feet": [], "knees": [], "body": [],
                         "head": [("head", J[k]) for k in ("head_front", "head_back", "head_left", "head_right")
                                  if k in J]}
        for side, _ in SIDES:
            self.contacts["feet"] += [(f"foot.{side}", J[f"heel_ground.{side}"]),
                                      (f"foot.{side}", J[f"toe_ground.{side}"])]
            knee = J[f"knee.{side}"]
            self.contacts["knees"].append((f"shin.{side}", knee + Vector((0, -leg_r, 0))))
            self.contacts["body"].append((f"hand.{side}", J[f"hand.{side}"]))
        self.contacts["body"] += [("head", J["top"]), ("head", J["neck"].lerp(J["top"], 0.5) +
                                                         Vector((0, -0.08 * h, 0)))]

    @staticmethod
    def limb_dir(s: float, abd: float, flex: float) -> Vector:
        a, f = math.radians(abd), math.radians(flex)
        return Vector((s * math.sin(a) * math.cos(f), -math.sin(f), -math.cos(a) * math.cos(f)))

    def rotations(self, P: dict) -> dict:
        g = lambda k, d=0.0: P.get(k, d)
        R = {}
        tilt, turn = self.tilt, math.sqrt(self.tilt)
        for n in ("hips", "chest", "head"):
            R[n] = _spine_rot(tilt * g(n + ".pitch"), tilt * g(n + ".roll"), turn * g(n + ".yaw"))
        Rh = R["hips"]
        for side, s in SIDES:
            abd0, flex0 = self.leg_rest[side]
            target = self.limb_dir(s, abd0 + g(f"leg{side}.spread"), flex0 + g(f"leg{side}.flex"))
            thigh, shin, foot = f"thigh.{side}", f"shin.{side}", f"foot.{side}"
            R[thigh] = _rot_between(self.dir[thigh], Rh.transposed() @ target)
            R[shin] = Matrix.Rotation(math.radians(g(f"leg{side}.knee")), 3, self.hinge[shin])
            above = Rh @ R[thigh] @ R[shin]
            want = Matrix.Rotation(math.radians(-g(f"leg{side}.foot")), 3, "X") @ self.dir[foot]
            R[foot] = _rot_between(self.dir[foot], above.transposed() @ want)
            upper, fore, hand = f"upper_arm.{side}", f"forearm.{side}", f"hand.{side}"
            target = self.limb_dir(s, g(f"arm{side}.abd", self.arm_idle[side]), g(f"arm{side}.flex"))
            R[upper] = _rot_between(self.dir[upper], target)
            R[fore] = Matrix.Rotation(math.radians(-g(f"arm{side}.elbow")), 3, self.hinge[fore])
            R[hand] = Matrix.Rotation(math.radians(-g(f"arm{side}.wrist")), 3, self.hinge[hand])
        return R

    def fk(self, R: dict) -> tuple[dict, dict]:
        world, head = {}, {}
        ident = Matrix.Identity(3)
        for n in self.order:
            p = self.parent[n]
            if p is None:
                world[n], head[n] = R.get(n, ident), self.head[n].copy()
            else:
                world[n] = world[p] @ R.get(n, ident)
                head[n] = head[p] + world[p] @ (self.head[n] - self.head[p])
        return world, head

    def lowest(self, R: dict, groups) -> float:
        world, head = self.fk(R)
        zs = [(head[b] + world[b] @ (p - self.head[b])).z for g in groups for b, p in self.contacts[g]]
        return min(zs)

    def local(self, name: str, R: Matrix):
        B = self.basis[name]
        return (B.inverted() @ R @ B).to_quaternion()


def _blend(keys: list, t: float) -> dict:
    """Interpolate key poses [(t, pose), ...] with ease-in-out."""
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if t <= t1 + 1e-9:
            u = 0.0 if t1 <= t0 else min(1.0, max(0.0, (t - t0) / (t1 - t0)))
            u = u * u * (3 - 2 * u)
            return {k: p0.get(k, 0.0) + (p1.get(k, 0.0) - p0.get(k, 0.0)) * u for k in set(p0) | set(p1)}
    return dict(keys[-1][1])


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


def _bake(rig, poser: Poser, name: str, frames: int, pose_fn, loop: bool, contacts=("feet",)) -> None:
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    rig.animation_data_create()
    rig.animation_data.action = action
    bones = rig.pose.bones
    for pb in bones:
        pb.rotation_mode = "QUATERNION"
        pb.location = (0, 0, 0)
    previous = {}
    keys = frames + 1 if loop else frames
    for i in range(keys):
        t = i / frames if loop else i / max(1, frames - 1)
        P = pose_fn(t)
        R = poser.rotations(P)
        air = P.get("air", 0.0)
        ground = -poser.lowest(R, contacts)
        height = (1 - air) * ground + (air * P.get("lift", 0.0) + P.get("bounce", 0.0)) * poser.h
        for pb in bones:
            q = poser.local(pb.name, R[pb.name]) if pb.name in R else Matrix.Identity(3).to_quaternion()
            if pb.name in previous and q.dot(previous[pb.name]) < 0:
                q.negate()
            previous[pb.name] = q
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=1 + i)
        bones["root"].location = (0.0, 0.0, height)
        bones["root"].keyframe_insert("location", frame=1 + i)
    action.frame_range = (1, keys)
    if loop:
        action.use_cyclic = True
        _cyclic(action)


def _base(poser: Poser) -> dict:
    P = {"armL.elbow": 10.0, "armR.elbow": 10.0}
    for side, _ in SIDES:
        P[f"arm{side}.abd"] = poser.arm_idle[side]
    return P


def _key(base: dict, **values) -> dict:
    """A key pose: the base pose plus values written as legL_knee=... or air=..."""
    pose = dict(base)
    for k, v in values.items():
        part, _, field = k.partition("_")
        pose[f"{part}.{field}" if field else part] = v
    return pose


def _both(base: dict, part: str, **values) -> dict:
    pose = dict(base)
    for k, v in values.items():
        for side in "LR":
            pose[f"{part}{side}.{k}"] = v
    return pose


def add_animations(rig, J: dict, h: float, names=ANIMATIONS) -> list:
    poser = Poser(rig, J, h)
    base = _base(poser)
    idle_abd = dict((side, poser.arm_idle[side]) for side, _ in SIDES)

    def idle(t):
        s, c = math.sin(2 * math.pi * t), math.cos(2 * math.pi * t)
        P = dict(base)
        bend = 4.0 + 2.5 * (1 - c) / 2
        for side, sign in SIDES:
            P[f"leg{side}.flex"], P[f"leg{side}.knee"] = bend / 2, bend
            P[f"arm{side}.abd"] = idle_abd[side] + 1.5 * s
            P[f"arm{side}.flex"] = 3.0 + 1.0 * s
            P[f"arm{side}.elbow"] = 12.0 + 3.0 * s
        P["chest.pitch"], P["head.pitch"] = 1.5 * s, -1.2 * s
        P["head.yaw"] = 1.5 * math.sin(2 * math.pi * t + 1.0)
        return P

    def gait(t, stride, arm_swing, knee_swing, knee_load, toe_off, lean, elbow, twist, flight=0.0):
        P = dict(base)
        for side, offset in (("L", 0.0), ("R", 0.5)):
            u = (t + offset) % 1.0
            c = math.cos(2 * math.pi * u)
            P[f"leg{side}.flex"] = stride * c + lean * 0.3
            P[f"leg{side}.knee"] = 5 + knee_load * _bump(u, 0.12, 0.08) + knee_swing * _bump(u, 0.7, 0.14)
            P[f"leg{side}.foot"] = (10 * _bump(u, 0.0, 0.07) - toe_off * _bump(u, 0.55, 0.08)
                                    + 4 * _bump(u, 0.85, 0.1))
            P[f"arm{side}.flex"] = -arm_swing * c + lean * 0.4
            P[f"arm{side}.elbow"] = elbow + 14 * max(0.0, -c)
        c0, s0 = math.cos(2 * math.pi * t), math.sin(2 * math.pi * t)
        P["hips.yaw"], P["chest.yaw"], P["head.yaw"] = -twist * c0, 1.5 * twist * c0, -0.5 * twist * c0
        P["hips.roll"], P["chest.roll"] = -2.0 * s0, 1.5 * s0
        P["chest.pitch"], P["head.pitch"] = lean, -0.7 * lean
        if flight:  # both feet off the ground just after each push-off
            P["bounce"] = flight * (_bump(t, 0.47, 0.07) + _bump(t, 0.97, 0.07))
        return P

    def walk(t):
        return gait(t, stride=26, arm_swing=22, knee_swing=52, knee_load=12, toe_off=22, lean=4, elbow=12, twist=6)

    def run(t):
        P = gait(t, stride=36, arm_swing=34, knee_swing=90, knee_load=25, toe_off=32, lean=12, elbow=78,
                 twist=8, flight=0.02)
        for side, _ in SIDES:
            P[f"arm{side}.abd"] = idle_abd[side] + 6
        return P

    attack = [
        (0.0, base),
        (0.3, _key(base, armR_abd=150, armR_flex=20, armR_elbow=70, chest_yaw=-25, chest_pitch=-6,
                   hips_yaw=-10, armL_flex=25, armL_abd=30, armL_elbow=40, legL_flex=12, legL_knee=12,
                   legR_flex=-10, legR_knee=10)),
        (0.45, _key(base, armR_abd=40, armR_flex=75, armR_elbow=5, chest_yaw=22, chest_pitch=14, hips_yaw=10,
                    head_pitch=6, armL_flex=-20, armL_abd=25, legL_flex=22, legL_knee=25, legR_flex=-18,
                    legR_knee=8)),
        (0.6, _key(base, armR_abd=25, armR_flex=50, armR_elbow=15, chest_yaw=28, chest_pitch=18, hips_yaw=10,
                   head_pitch=6, armL_flex=-15, armL_abd=25, legL_flex=22, legL_knee=25, legR_flex=-18,
                   legR_knee=8)),
        (1.0, base),
    ]
    cast = [
        (0.0, base),
        (0.25, _key(_both(base, "arm", abd=35, flex=35, elbow=95), chest_pitch=-7, head_pitch=-4,
                    legL_knee=8, legR_knee=8, legL_flex=4, legR_flex=4)),
        (0.45, _key(_both(base, "arm", abd=18, flex=88, elbow=8), chest_pitch=10, head_pitch=4,
                    legL_flex=12, legL_knee=10, legR_flex=-6, legR_knee=4)),
        (0.75, _key(_both(base, "arm", abd=22, flex=82, elbow=12), chest_pitch=8, head_pitch=3,
                    legL_flex=12, legL_knee=10, legR_flex=-6, legR_knee=4)),
        (1.0, base),
    ]
    flung = {f"arm{side}.abd": idle_abd[side] + 25 for side, _ in SIDES}
    hit = [
        (0.0, base),
        (0.25, _key(_both(dict(base, **flung), "arm", flex=-15, elbow=35), chest_pitch=-16, head_pitch=-14,
                    hips_pitch=-4, legL_knee=10, legR_knee=10, legL_flex=5, legR_flex=5)),
        (0.55, _key(_both(base, "arm", elbow=20), chest_pitch=-6, head_pitch=-4)),
        (1.0, base),
    ]
    death = [
        (0.0, base),
        (0.15, _key(_both(dict(base, **flung), "arm", flex=-10, elbow=30), chest_pitch=-14, head_pitch=-16,
                    legL_knee=8, legR_knee=8, legL_flex=4, legR_flex=4)),
        (0.45, _key(_both(_both(base, "arm", flex=15, elbow=20), "leg", flex=40, knee=85, foot=0),
                    hips_pitch=10, chest_pitch=18, head_pitch=15)),
        (0.7, _key(_both(_both(base, "arm", abd=15, flex=25, elbow=20), "leg", flex=12, knee=105, foot=-70),
                   hips_pitch=15, chest_pitch=35, head_pitch=25)),
        (1.0, _key(_both(_both(base, "arm", abd=10, flex=40, elbow=15), "leg", flex=15, knee=110, foot=-80),
                   hips_pitch=30, chest_pitch=55, head_pitch=30)),
    ]
    jump = [
        (0.0, base),
        (0.18, _key(_both(_both(base, "arm", abd=25, flex=-35, elbow=15), "leg", flex=32, knee=62, foot=0),
                    hips_pitch=18, chest_pitch=12, head_pitch=-10)),
        (0.32, _key(_both(_both(base, "arm", abd=35, flex=120, elbow=15), "leg", flex=-4, knee=4, foot=-38),
                    chest_pitch=-4)),
        (0.5, _key(_both(_both(base, "arm", abd=55, flex=95, elbow=25), "leg", flex=40, knee=75, foot=-5),
                   chest_pitch=6, air=1.0, lift=0.08)),
        (0.68, _key(_both(_both(base, "arm", abd=45, flex=60, elbow=20), "leg", flex=18, knee=28, foot=-15),
                    air=1.0, lift=0.03)),
        (0.8, _key(_both(_both(base, "arm", abd=30, flex=25, elbow=20), "leg", flex=30, knee=55, foot=0),
                   hips_pitch=14, chest_pitch=12)),
        (1.0, base),
    ]
    # The jump only hops (8% of the height): sprite_render frames every
    # animation with one camera, so a full-height jump would shrink the
    # character in all of them. Games move the sprite up themselves.
    library = {
        "Idle": (24, idle, ("feet",)),
        "Walk": (16, walk, ("feet",)),
        "Run": (12, run, ("feet",)),
        "Attack": (12, lambda t: _blend(attack, t), ("feet",)),
        "Cast": (16, lambda t: _blend(cast, t), ("feet",)),
        "Hit": (8, lambda t: _blend(hit, t), ("feet",)),
        "Death": (18, lambda t: _blend(death, t), ("feet", "knees", "body", "head")),
        "Jump": (16, lambda t: _blend(jump, t), ("feet",)),
    }
    made = []
    for name in names:
        frames, fn, contacts = library[name]
        _bake(rig, poser, name, frames, fn, name in LOOPS, contacts)
        made.append(name)
    return made


# ---------------------------------------------------------------- overlay

def draw_skeleton(body, rig, path: str, height_px: int = 768) -> None:
    """Front render with the bones drawn on top, for checking the joints."""
    tmp = os.path.join(tempfile.mkdtemp(prefix="autorig_"), "front.png")
    rig.data.pose_position = "REST"
    try:
        mapping = render_front(body, tmp, height_px, look="workbench")
    finally:
        rig.data.pose_position = "POSE"
    img = _read_image(tmp)[..., :3].copy()
    os.remove(tmp)
    rows, cols = img.shape[:2]
    cx, cz = mapping["center"]
    p = mapping["pixel"]

    def px(v):
        return (v.x - cx) / p + cols / 2, rows / 2 - (v.z - cz) / p

    def disk(c, r, color, radius):
        r0, r1 = int(max(0, r - radius)), int(min(rows, r + radius + 1))
        c0, c1 = int(max(0, c - radius)), int(min(cols, c + radius + 1))
        yy, xx = np.mgrid[r0:r1, c0:c1]
        inside = (yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2
        img[r0:r1, c0:c1][inside] = color

    for b in rig.data.bones:
        if not b.use_deform:
            continue
        (c0, r0), (c1, r1) = px(b.head_local), px(b.tail_local)
        steps = int(max(abs(c1 - c0), abs(r1 - r0)) * 2) + 1
        for t in np.linspace(0, 1, steps):
            disk(c0 + (c1 - c0) * t, r0 + (r1 - r0) * t, (0.1, 1.0, 0.2), 2.2)
    for b in rig.data.bones:
        if b.use_deform:
            for v in (b.head_local, b.tail_local):
                disk(*px(v), (1.0, 0.85, 0.0), 3.5)
    out = bpy.data.images.new("AutoRigSkeleton", cols, rows, alpha=False)
    rgba = np.ones((rows, cols, 4), dtype=np.float32)
    rgba[..., :3] = img
    out.pixels.foreach_set(rgba[::-1].ravel())
    out.filepath_raw = path
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)


# ---------------------------------------------------------------- main

def rig_character(model: str, out: str, keypoints: dict | None = None, landmarks: dict | None = None,
                  target_faces: int = 40000, weights: str = "auto", animations=ANIMATIONS,
                  front_axis: str = "-Y", skeleton_png: str | None = None, height: float = 1.0) -> dict:
    body = load_character(model, front_axis, height)
    faces_before = len(body.data.polygons)
    _decimate(body, target_faces)
    tmp = os.path.join(tempfile.mkdtemp(prefix="autorig_"), "mask.png")
    mapping = render_front(body, tmp, 640, look="mask")
    mask = _read_image(tmp)[..., 3] > 0.5
    os.remove(tmp)
    sil = Silhouette(mask, mapping)
    co = _vertices(body)
    J, info, notes = measure(sil, co, height)
    if keypoints:
        apply_keypoints(J, info, keypoints, sil, co, height, notes)
    if landmarks:
        apply_landmarks(J, info, landmarks, height)
    rig = build_armature(J, height)
    method = skin(body, rig, J, info, height, weights)
    made = add_animations(rig, J, height, animations) if animations else []
    if made:
        rig.animation_data.action = bpy.data.actions[made[0]]
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if skeleton_png is None:
        skeleton_png = os.path.splitext(out)[0] + "_skeleton.png"
    if skeleton_png:
        draw_skeleton(body, rig, os.path.abspath(skeleton_png))
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, 24
    scene.render.engine = "BLENDER_EEVEE"
    # The render path is a fixed-size buffer saved whole: writing "//" alone
    # would leave the temp file name (with the user's name) after it.
    scene.render.filepath = " " * 1000
    scene.render.filepath = "//"
    bpy.ops.wm.save_as_mainfile(filepath=out)
    joints = {k: [round(c, 4) for c in v] for k, v in J.items() if "ground" not in k and not k.startswith("head_")}
    return {"blend": out, "faces_before": faces_before, "faces": len(body.data.polygons),
            "weights": method, "sources": info["sources"], "skirt": info["skirt"],
            "landmarks": landmark_summary(J, info, height), "joints": joints,
            "actions": made, "loops": [a for a in made if a in LOOPS], "fps": FPS,
            "skeleton_png": os.path.abspath(skeleton_png) if skeleton_png else None, "notes": notes}


def front_render(model: str, path: str, size: int = 1024, look: str = "eevee", front_axis: str = "-Y") -> dict:
    body = load_character(model, front_axis)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return render_front(body, os.path.abspath(path), size, look)


def keypoints_from_pose(pose: dict, mapping: dict) -> dict:
    """DWPose/OpenPose JSON (pixels or 0-1) for a front render -> world X/Z keypoints."""
    frames = pose if isinstance(pose, list) else [pose]
    frame = frames[0]
    people = frame.get("people") or []
    if not people:
        return {}
    width, height = mapping["size"]
    cw, ch = frame.get("canvas_width", width), frame.get("canvas_height", height)
    flat = people[0].get("pose_keypoints_2d") or []
    values = [v for i, v in enumerate(flat) if i % 3 != 2]
    normalized = bool(values) and max(values) <= 1.0
    cx, cz = mapping["center"]
    p = mapping["pixel"]
    out = {}
    for i, name in enumerate(POSE_NAMES):
        if 3 * i + 2 >= len(flat):
            break
        u, v, conf = flat[3 * i:3 * i + 3]
        if conf <= 0:
            continue
        if normalized:
            u, v = u * cw, v * ch
        u, v = u * width / cw, v * height / ch
        out[name] = [cx + (u - width / 2) * p, cz + (height / 2 - v) * p, float(conf)]
    return out


def _parse_landmarks(text: str) -> dict:
    out = {}
    for part in filter(None, (text or "").split(",")):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in LANDMARK_KEYS:
            raise RigError(f"unknown landmark {key!r}; use {', '.join(LANDMARK_KEYS)}")
        out[key] = float(value)
    return out


def _parse_animations(text: str) -> tuple:
    text = (text or "all").strip()
    if text.lower() == "all":
        return ANIMATIONS
    if text.lower() == "none":
        return ()
    by_lower = {a.lower(): a for a in ANIMATIONS}
    names = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        if part.lower() not in by_lower:
            raise RigError(f"unknown animation {part!r}; use {', '.join(ANIMATIONS)}, all or none")
        names.append(by_lower[part.lower()])
    return tuple(names)


def main(argv: list) -> None:
    parser = argparse.ArgumentParser(description="Rig and animate a generated humanoid character.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", help="rigged .blend to write")
    parser.add_argument("--front-render", help="only render the front view for a pose detector")
    parser.add_argument("--front-size", type=int, default=1024)
    parser.add_argument("--front-look", default="eevee", choices=("eevee", "workbench"))
    parser.add_argument("--front-axis", default="-Y", choices=tuple(FRONT_AXES))
    parser.add_argument("--keypoints", default="",
                        help='JSON: {"pose": <DWPose/OpenPose output>, "mapping": <AUTO_RIG_FRONT result>} '
                             'or {"points": {"neck": [x, z, confidence], ...}} in world units')
    parser.add_argument("--landmarks", default="", help="e.g. hips=0.25,neck=0.5,leg_x=0.07")
    parser.add_argument("--faces", type=int, default=40000, help="decimate to about this many faces (0 = keep)")
    parser.add_argument("--weights", default="auto", choices=("auto", "heat", "distance"))
    parser.add_argument("--animations", default="all", help="all, none or e.g. Idle,Walk,Run")
    parser.add_argument("--skeleton-png", default=None, help="skeleton overlay path ('' to skip)")
    args = parser.parse_args(argv)
    try:
        if args.front_render:
            result = front_render(args.model, args.front_render, args.front_size, args.front_look,
                                  args.front_axis)
            print("AUTO_RIG_FRONT " + json.dumps(result))
            return
        if not args.out:
            raise RigError("pass --out (or --front-render)")
        keypoints = None
        if args.keypoints:
            with open(args.keypoints, encoding="utf-8") as f:
                data = json.load(f)
            if "pose" in data and "mapping" in data:
                keypoints = keypoints_from_pose(data["pose"], data["mapping"])
            else:
                keypoints = data.get("points", data)
        result = rig_character(args.model, args.out, keypoints, _parse_landmarks(args.landmarks), args.faces,
                               args.weights, _parse_animations(args.animations), args.front_axis,
                               args.skeleton_png)
        print("AUTO_RIG_RESULT " + json.dumps(result))
    except RigError as e:
        print("AUTO_RIG_ERROR " + str(e))
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
