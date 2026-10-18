"""Render pixel-art-ready sprite frames of a 3D character with Blender.

An orthographic camera orbits the character on a rig that also carries the
lights, so every direction is lit the same way relative to the screen. The
framing is computed once from every rendered frame and direction, so the
character keeps the same scale and ground line throughout; pixelization
happens afterwards (LibreSprite MCP pixelize / frames_to_sprite).

Headless (write options as --key=value: "--front -Y" would read -Y as an option):
    blender -b --factory-startup -P sprite_render.py -- --model=hero.glb --out=renders/hero
    blender -b --factory-startup hero.blend -P sprite_render.py -- --out=renders/hero --animation=all
    blender -b --factory-startup -P sprite_render.py -- --model=hero.fbx --list

Inside a running Blender (for example from blender-mcp's execute_blender_code):
    import sys; sys.path.append(r"C:\\AI\\Puretty-sprites\\integrations\\blender")
    import sprite_render
    manifest = sprite_render.render(out_dir=r"C:\\renders\\hero", directions=8, size=64)

The scene is left as it was found: materials, render settings, action
assignments and visibility are restored, and the temporary camera rig is
deleted. A --model imported into a live session stays in the scene.

Output: <out>/<animation>/<direction>/<frame>.png plus <out>/manifest.json.
Directions are named by where the character faces on screen, starting
with S (facing the viewer) and going clockwise: S, SW, W, NW, N, NE, E, SE.
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
from mathutils import Euler, Vector

DIRECTION_NAMES = {
    1: ["S"],
    2: ["S", "N"],
    4: ["S", "W", "N", "E"],
    8: ["S", "SW", "W", "NW", "N", "NE", "E", "SE"],
    16: ["S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
         "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE"],
}

# Rig rotation that brings a model facing this axis toward the S camera
# (the S camera looks along +Y from -Y, so a -Y-facing model faces it).
FRONT_OFFSETS = {"-Y": 0.0, "+X": 90.0, "+Y": 180.0, "-X": 270.0}

RENDERABLE_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES"}
IMPORTERS = {
    ".glb": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".gltf": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".fbx": lambda p: _import_fbx(p),
    ".obj": lambda p: bpy.ops.wm.obj_import(filepath=p),
    ".ply": lambda p: bpy.ops.wm.ply_import(filepath=p),
    ".stl": lambda p: bpy.ops.wm.stl_import(filepath=p),
    ".usd": lambda p: bpy.ops.wm.usd_import(filepath=p),
    ".usda": lambda p: bpy.ops.wm.usd_import(filepath=p),
    ".usdc": lambda p: bpy.ops.wm.usd_import(filepath=p),
    ".usdz": lambda p: bpy.ops.wm.usd_import(filepath=p),
    ".abc": lambda p: bpy.ops.wm.alembic_import(filepath=p),
}

MAX_POINTS_PER_OBJECT = 4000
MAX_BOUNDS_FRAMES = 240


class SpriteRenderError(Exception):
    pass


def _import_fbx(path):
    try:
        return bpy.ops.import_scene.fbx(filepath=path)
    except (AttributeError, RuntimeError):
        return bpy.ops.wm.fbx_import(filepath=path)


# ---------------------------------------------------------------- scene

def load_model(model: str | None) -> list:
    """Load or import the model; returns the newly imported objects (or [])."""
    if not model:
        return []
    path = os.path.abspath(model)
    if not os.path.isfile(path):
        raise SpriteRenderError(f"Model not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".blend":
        if os.path.normcase(bpy.data.filepath) == os.path.normcase(path):
            return []
        if not bpy.app.background:
            raise SpriteRenderError("Open the .blend in Blender first, then render without a model")
        bpy.ops.wm.open_mainfile(filepath=path)
        return []
    if ext not in IMPORTERS:
        raise SpriteRenderError(f"Unsupported model format '{ext}' (use {', '.join(sorted(IMPORTERS))} or .blend)")
    if bpy.app.background and not bpy.data.filepath:
        # The startup scene's cube, camera and light would end up in the shot.
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
    before = set(bpy.data.objects)
    IMPORTERS[ext](path)
    imported = [ob for ob in bpy.data.objects if ob not in before]
    if not imported:
        raise SpriteRenderError(f"Importing {path} produced no objects")
    return imported


def character_objects(scene, objects: str = "", collection: str = "", imported=None) -> list:
    if objects:
        names = [n.strip() for n in objects.split(",") if n.strip()]
        missing = [n for n in names if n not in bpy.data.objects]
        if missing:
            raise SpriteRenderError(f"Objects not found: {', '.join(missing)}")
        chosen = [bpy.data.objects[n] for n in names]
    elif collection:
        if collection not in bpy.data.collections:
            raise SpriteRenderError(f"Collection not found: {collection}")
        chosen = list(bpy.data.collections[collection].all_objects)
    elif imported:
        # Importers add hidden helpers (e.g. glTF's bone-shape Icosphere).
        chosen = [ob for ob in imported if not ob.hide_render and ob.visible_get()]
    else:
        chosen = [ob for ob in scene.objects if not ob.hide_render and ob.visible_get()]
    result = [ob for ob in chosen if ob.type in RENDERABLE_TYPES and ob.name in scene.objects]
    if not result:
        raise SpriteRenderError("No renderable objects found for the character")
    return result


def find_rig(objs):
    """The armature driving the character, or None."""
    counts = {}
    for ob in objs:
        rigs = {m.object for m in ob.modifiers if m.type == "ARMATURE" and m.object}
        if ob.parent and ob.parent.type == "ARMATURE":
            rigs.add(ob.parent)
        for rig in rigs:
            counts[rig] = counts.get(rig, 0) + 1
    return max(counts, key=counts.get) if counts else None


def animated_object(objs, rig):
    if rig is not None:
        return rig
    for ob in objs:
        if ob.animation_data and ob.animation_data.action:
            return ob
    return None


def action_fcurves(action):
    """F-curves of a layered (4.4+) or legacy action."""
    curves = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                curves.extend(bag.fcurves)
    if not curves and hasattr(action, "fcurves"):
        curves = list(action.fcurves)
    return curves


def suitable_actions(target, objs) -> list:
    """Actions that can drive the character's animated object."""
    if target is None:
        return []
    found = []
    current = target.animation_data.action if target.animation_data else None
    bone_names = set(target.pose.bones.keys()) if target.type == "ARMATURE" and target.pose else set()
    for action in bpy.data.actions:
        if action == current:
            found.append(action)
            continue
        slots = getattr(action, "slots", None)
        if slots is not None and len(slots) and not any(s.target_id_type == "OBJECT" for s in slots):
            continue
        paths = [fc.data_path for fc in action_fcurves(action)]
        if bone_names:
            bones = {m.group(1) for p in paths for m in [re.match(r'pose\.bones\["(.+?)"\]', p)] if m}
            if bones & bone_names:
                found.append(action)
        elif any(p in ("location", "rotation_euler", "rotation_quaternion", "scale") for p in paths):
            found.append(action)
    return found


def action_range(action) -> tuple[int, int, bool]:
    start, end = action.frame_range
    return int(round(start)), int(round(end)), bool(getattr(action, "use_cyclic", False))


def assign_action(target, action) -> None:
    ad = target.animation_data or target.animation_data_create()
    ad.action = action
    if action is not None and getattr(ad, "action_slot", None) is None:
        suitable = list(getattr(ad, "action_suitable_slots", []))
        if suitable:
            ad.action_slot = suitable[0]


def plan_animations(scene, target, objs, animation, frame_start, frame_end, frame_step, trim_loop):
    """[(name, action or None, [frames])] to render."""
    step = max(1, int(frame_step))
    wanted = (animation or "").strip()
    actions = suitable_actions(target, objs)
    current = target.animation_data.action if target is not None and target.animation_data else None

    if wanted.lower() in ("static", "none"):
        chosen = [None]
    elif wanted.lower() == "all":
        chosen = actions or [None]
    elif wanted:
        by_name = {a.name: a for a in bpy.data.actions}
        chosen = []
        for name in [n.strip() for n in wanted.split(",") if n.strip()]:
            if name not in by_name:
                options = ", ".join(a.name for a in actions) or "none"
                raise SpriteRenderError(f"Animation '{name}' not found (actions: {options})")
            chosen.append(by_name[name])
    else:
        chosen = [current] if current is not None else [None]

    plans = []
    for action in chosen:
        if action is None:
            name = "static"
            if frame_start or frame_end:
                first = frame_start or scene.frame_start
                last = frame_end or first
                frames = list(range(first, last + 1, step))
                name = "scene"
            else:
                frames = [scene.frame_current]
        else:
            name = action.name
            start, end, cyclic = action_range(action)
            first = frame_start or start
            last = frame_end or end
            if (cyclic or trim_loop) and not frame_end and last > first:
                last -= 1  # the loop's last key repeats its first
            frames = list(range(first, last + 1, step))
        if not frames:
            raise SpriteRenderError(f"Animation '{name}' has an empty frame range")
        plans.append((name, action, frames))
    return plans


# ---------------------------------------------------------------- framing

def _world_points(objs, depsgraph):
    pts = []
    for ob in objs:
        ev = ob.evaluated_get(depsgraph)
        try:
            mesh = ev.to_mesh()
        except RuntimeError:
            mesh = None
        if mesh is None:
            continue
        n = len(mesh.vertices)
        if n:
            co = np.empty(n * 3, np.float64)
            mesh.vertices.foreach_get("co", co)
            co = co.reshape(n, 3)
            if n > MAX_POINTS_PER_OBJECT:
                co = co[:: math.ceil(n / MAX_POINTS_PER_OBJECT)]
            mw = np.array(ev.matrix_world, np.float64)
            pts.append(co @ mw[:3, :3].T + mw[:3, 3])
        ev.to_mesh_clear()
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def measure(scene, objs, target, plans):
    """World points of the character across every frame that will be rendered."""
    samples = []
    for _, action, frames in plans:
        if action is not None:
            assign_action(target, action)
        picks = frames if len(frames) <= MAX_BOUNDS_FRAMES else \
            [frames[int(i)] for i in np.linspace(0, len(frames) - 1, MAX_BOUNDS_FRAMES)]
        for f in picks:
            scene.frame_set(f)
            samples.append(_world_points(objs, bpy.context.evaluated_depsgraph_get()))
    points = np.concatenate([s for s in samples if len(s)]) if any(len(s) for s in samples) else None
    if points is None or not len(points):
        raise SpriteRenderError("The character has no geometry to frame")
    return points, samples[0] if len(samples[0]) else points


def compute_framing(points, first_points, angles, elevation, margin, size, pivot=None):
    """Orthographic scale, camera lift and ground pixel shared by all directions."""
    if pivot is None:
        lo, hi = first_points[:, :2].min(axis=0), first_points[:, :2].max(axis=0)
        pivot = (lo + hi) / 2
    pivot = np.asarray(pivot, np.float64)
    ground = float(points[:, 2].min())
    e = math.radians(elevation)
    rel = points.copy()
    rel[:, :2] -= pivot
    rel[:, 2] -= ground
    half_w, y_lo, y_hi = 0.0, math.inf, -math.inf
    for a in angles:
        t = math.radians(-a)
        x = rel[:, 0] * math.cos(t) - rel[:, 1] * math.sin(t)
        y = rel[:, 0] * math.sin(t) + rel[:, 1] * math.cos(t)
        ys = y * math.sin(e) + rel[:, 2] * math.cos(e)
        half_w = max(half_w, float(np.abs(x).max()))
        y_lo, y_hi = min(y_lo, float(ys.min())), max(y_hi, float(ys.max()))
    span = max(2 * half_w, y_hi - y_lo, 1e-4)
    ortho = span * (1 + 2 * margin)
    lift = (y_hi + y_lo) / 2
    # Snap the ground line to a whole pixel row of the final frame.
    ground_row = round(size / 2 + lift * size / ortho)
    lift = (ground_row - size / 2) * ortho / size
    return {"pivot": pivot, "ground": ground, "ortho": ortho, "lift": lift,
            "ground_row": ground_row, "height": float(points[:, 2].max()) - ground}


# ---------------------------------------------------------------- rig, lights, materials

def build_rig(scene, framing, elevation, shading):
    created = []
    rig = bpy.data.objects.new("SpriteRig", None)
    scene.collection.objects.link(rig)
    rig.location = (framing["pivot"][0], framing["pivot"][1], framing["ground"])
    created.append(rig)

    e = math.radians(elevation)
    distance = max(framing["ortho"], framing["height"]) * 4 + 10
    cam_data = bpy.data.cameras.new("SpriteCamera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = framing["ortho"]
    cam_data.clip_start = 0.01
    cam_data.clip_end = distance * 3
    cam = bpy.data.objects.new("SpriteCamera", cam_data)
    scene.collection.objects.link(cam)
    cam.parent = rig
    up = Vector((0, math.sin(e), math.cos(e)))
    cam.location = Vector((0, -distance * math.cos(e), distance * math.sin(e))) + up * framing["lift"]
    cam.rotation_euler = Euler((math.radians(90) - e, 0, 0))
    created.append(cam)

    def sun(name, toward, strength, shadow):
        data = bpy.data.lights.new(name, "SUN")
        data.energy = strength
        data.angle = math.radians(0.5)
        data.use_shadow = shadow
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.parent = rig
        light.rotation_euler = Vector(toward).normalized().to_track_quat("-Z", "Y").to_euler()
        created.append(light)

    # Light comes from the upper left of the screen and a little in front.
    if shading == "toon":
        sun("SpriteKeyLight", (0.55, 0.6, -0.75), math.pi, True)
    elif shading == "lit":
        sun("SpriteKeyLight", (0.55, 0.6, -0.75), 3.0, True)
        sun("SpriteFillLight", (-0.7, -0.2, -0.3), 1.0, False)
    return rig, cam, created


def _surface_sockets(nt):
    """(output node, base-colour socket or value, alpha socket or value)."""
    outputs = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"]
    out = next((n for n in outputs if n.is_active_output), outputs[0] if outputs else None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    node = out.inputs["Surface"].links[0].from_node if out.inputs["Surface"].links else None
    seen = set()
    while node is not None and node.name not in seen:
        seen.add(node.name)
        if node.type == "BSDF_PRINCIPLED":
            return out, node.inputs["Base Color"], node.inputs["Alpha"]
        if "Color" in node.inputs and node.type in ("BSDF_DIFFUSE", "EMISSION", "BSDF_TOON",
                                                     "SUBSURFACE_SCATTERING", "BSDF_SHEEN"):
            return out, node.inputs["Color"], None
        if node.type in ("MIX_SHADER", "ADD_SHADER"):
            shader_inputs = [s for s in node.inputs if s.type == "SHADER" and s.links]
            node = shader_inputs[-1].links[0].from_node if shader_inputs else None
            continue
        break
    return out, None, None


def _socket_source(nt, socket, fallback):
    """A node output carrying socket's colour/value (its link, or a constant)."""
    if socket is not None and socket.links:
        return socket.links[0].from_socket
    if socket is not None:
        value = socket.default_value
    else:
        value = fallback
    if isinstance(value, float):
        node = nt.nodes.new("ShaderNodeValue")
        node.outputs[0].default_value = value
    else:
        node = nt.nodes.new("ShaderNodeRGB")
        node.outputs[0].default_value = tuple(value)[:4] if len(tuple(value)) >= 4 else (*value, 1.0)
    return node.outputs[0]


def make_sprite_material(source, shading, toon_steps, shadow):
    """A toon or flat (unlit) copy of a material, keeping its base colour and alpha."""
    if source is None:
        mat = bpy.data.materials.new("SpriteDefault")
        mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    else:
        mat = source.copy()
    mat.name = f"_sprite_{shading}_{source.name if source else 'default'}"
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    out, base_socket, alpha_socket = _surface_sockets(nt)
    fallback = tuple(source.diffuse_color) if source is not None else (0.8, 0.8, 0.8, 1.0)
    base = _socket_source(nt, base_socket, fallback)

    emission = nt.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    if shading == "flat":
        nt.links.new(base, emission.inputs["Color"])
    else:
        diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (1, 1, 1, 1)
        to_rgb = nt.nodes.new("ShaderNodeShaderToRGB")
        nt.links.new(diffuse.outputs["BSDF"], to_rgb.inputs["Shader"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        stops = [(0.0, shadow), (0.3, 1.0)] if toon_steps <= 2 else \
            [(0.0, shadow), (0.12, (shadow + 1) / 2), (0.55, 1.0)]
        elements = ramp.color_ramp.elements
        while len(elements) < len(stops):
            elements.new(0.5)
        for element, (position, level) in zip(elements, stops):
            element.position = position
            element.color = (level, level, level, 1.0)
        nt.links.new(to_rgb.outputs["Color"], ramp.inputs["Fac"])
        multiply = nt.nodes.new("ShaderNodeVectorMath")
        multiply.operation = "MULTIPLY"
        nt.links.new(base, multiply.inputs[0])
        nt.links.new(ramp.outputs["Color"], multiply.inputs[1])
        nt.links.new(multiply.outputs["Vector"], emission.inputs["Color"])
    shader = emission.outputs["Emission"]

    has_alpha = alpha_socket is not None and (alpha_socket.links or alpha_socket.default_value < 0.999)
    if has_alpha:
        alpha = _socket_source(nt, alpha_socket, 1.0)
        transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(alpha, mix.inputs["Fac"])
        nt.links.new(transparent.outputs["BSDF"], mix.inputs[1])
        nt.links.new(shader, mix.inputs[2])
        shader = mix.outputs["Shader"]
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"
    nt.links.new(shader, out.inputs["Surface"])
    for name in ("Volume", "Displacement"):
        if name in out.inputs:
            for link in list(out.inputs[name].links):
                nt.links.remove(link)
    return mat


class SceneState:
    """Everything the render changes, so it can be put back afterwards."""

    RENDER_ATTRS = ("engine", "resolution_x", "resolution_y", "resolution_percentage",
                    "film_transparent", "filter_size", "filepath", "use_motion_blur",
                    "use_compositing", "use_sequencer", "use_freestyle",
                    "pixel_aspect_x", "pixel_aspect_y", "fps", "fps_base", "dither_intensity")

    def __init__(self, scene):
        self.scene = scene
        r = scene.render
        self.render = {a: getattr(r, a) for a in self.RENDER_ATTRS if hasattr(r, a)}
        ims = r.image_settings
        self.image = {a: getattr(ims, a) for a in ("file_format", "color_mode", "color_depth", "compression")}
        vs = scene.view_settings
        self.view = {a: getattr(vs, a) for a in ("view_transform", "look", "exposure", "gamma")}
        self.taa = scene.eevee.taa_render_samples
        self.camera = scene.camera
        self.world = scene.world
        self.frame = scene.frame_current
        self.hidden = {}
        self.materials = []
        self.appended_slots = []
        self.actions = []
        self.nla = []
        self.created_objects = []
        self.created_data = []

    def restore(self):
        scene = self.scene
        for ob, slots in self.materials:
            for slot, material in zip(ob.material_slots, slots):
                slot.material = material
        for data in self.appended_slots:
            data.materials.pop()
        for ob, value in self.hidden.items():
            ob.hide_render = value
        for target, action, slot in self.actions:
            ad = target.animation_data
            ad.action = action
            if action is not None and slot is not None and getattr(ad, "action_slot", None) != slot:
                try:
                    ad.action_slot = slot
                except (TypeError, RuntimeError):
                    pass
        for track, mute in self.nla:
            track.mute = mute
        for ob in self.created_objects:
            if ob.name in bpy.data.objects:
                bpy.data.objects.remove(ob, do_unlink=True)
        owners = ((bpy.types.Material, bpy.data.materials), (bpy.types.Camera, bpy.data.cameras),
                  (bpy.types.Light, bpy.data.lights), (bpy.types.World, bpy.data.worlds))
        for data in self.created_data:
            for kind, collection in owners:
                if isinstance(data, kind) and data.name in collection:
                    collection.remove(data)
                    break
        r = scene.render
        for a, v in self.render.items():
            setattr(r, a, v)
        for a, v in self.image.items():
            try:
                setattr(r.image_settings, a, v)
            except (TypeError, ValueError):
                pass
        for a, v in self.view.items():
            setattr(scene.view_settings, a, v)
        scene.eevee.taa_render_samples = self.taa
        scene.camera = self.camera
        scene.world = self.world
        scene.frame_set(self.frame)


def prepare(scene, state, objs, target, shading, toon_steps, shadow, supersample, size, swap_actions):
    r = scene.render
    r.engine = "BLENDER_EEVEE"
    r.resolution_x = r.resolution_y = size * supersample
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1
    r.film_transparent = True
    r.dither_intensity = 0.0  # dither noise would add stray colours
    r.use_motion_blur = False
    r.use_compositing = False
    r.use_sequencer = False
    if hasattr(r, "use_freestyle"):
        r.use_freestyle = False
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "8"
    exact = supersample <= 1
    r.filter_size = 0.0 if exact else 1.5
    scene.eevee.taa_render_samples = 1 if exact else 16
    vs = scene.view_settings
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = 0.0
    vs.gamma = 1.0

    world = bpy.data.worlds.new("SpriteWorld")
    world.use_nodes = True
    background = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if background is not None:
        level = 0.25 if shading == "lit" else 0.0
        background.inputs["Color"].default_value = (level, level, level, 1.0)
        background.inputs["Strength"].default_value = 1.0
    scene.world = world
    state.created_data.append(world)

    # Hide everything renderable that is not part of the character, and the
    # scene's own lights: the rig brings its own.
    keep = set(objs)
    for ob in scene.objects:
        if ob in keep or ob.hide_render:
            continue
        if ob.type in RENDERABLE_TYPES or ob.type == "LIGHT":
            state.hidden[ob] = False
            ob.hide_render = True

    if shading in ("toon", "flat"):
        cache = {}
        for ob in objs:
            if not hasattr(ob.data, "materials"):
                continue
            if not ob.material_slots:
                ob.data.materials.append(None)
                state.appended_slots.append(ob.data)
            state.materials.append((ob, [slot.material for slot in ob.material_slots]))
            for slot in ob.material_slots:
                key = slot.material.name if slot.material else None
                if key not in cache:
                    cache[key] = make_sprite_material(slot.material, shading, toon_steps, shadow)
                    state.created_data.append(cache[key])
                slot.material = cache[key]

    if target is not None and swap_actions:
        # Mute NLA tracks so only the action being rendered drives the pose.
        ad = target.animation_data or target.animation_data_create()
        state.actions.append((target, ad.action, getattr(ad, "action_slot", None)))
        for track in ad.nla_tracks:
            state.nla.append((track, track.mute))
            track.mute = True


# ---------------------------------------------------------------- entry points

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "anim"


def direction_plan(directions: int, only: str, front: str):
    if directions not in DIRECTION_NAMES:
        raise SpriteRenderError(f"directions must be one of {sorted(DIRECTION_NAMES)}")
    front = front.upper().replace(" ", "")
    if front not in FRONT_OFFSETS:
        raise SpriteRenderError(f"front must be one of {', '.join(FRONT_OFFSETS)}")
    names = DIRECTION_NAMES[directions]
    step = 360.0 / directions
    plan = [(name, i * step) for i, name in enumerate(names)]
    if only:
        wanted = [n.strip().upper() for n in only.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in names]
        if unknown:
            raise SpriteRenderError(f"Unknown direction(s) {', '.join(unknown)}; with {directions} "
                                    f"directions use {', '.join(names)}")
        plan = [(n, a) for n, a in plan if n in wanted]
    return plan, FRONT_OFFSETS[front]


def list_animations(model: str | None = None, objects: str = "", collection: str = "") -> dict:
    imported = load_model(model)
    scene = bpy.context.scene
    objs = character_objects(scene, objects, collection, imported)
    rig = find_rig(objs)
    target = animated_object(objs, rig)
    current = target.animation_data.action if target is not None and target.animation_data else None
    actions = []
    for action in suitable_actions(target, objs):
        start, end, cyclic = action_range(action)
        actions.append({"name": action.name, "frame_start": start, "frame_end": end,
                        "cyclic": cyclic, "active": action == current})
    points = _world_points(objs, bpy.context.evaluated_depsgraph_get())
    size = (points.max(axis=0) - points.min(axis=0)).round(4).tolist() if len(points) else [0, 0, 0]
    return {"source": os.path.abspath(model) if model else bpy.data.filepath,
            "objects": [ob.name for ob in objs], "armature": rig.name if rig else None,
            "animated_object": target.name if target else None, "animations": actions,
            "fps": scene.render.fps / scene.render.fps_base,
            "scene_frames": [scene.frame_start, scene.frame_end], "dimensions": size}


def render(out_dir: str, model: str | None = None, directions: int = 8, only: str = "",
           size: int = 64, supersample: int = 4, elevation: float = 30.0, front: str = "-Y",
           shading: str = "toon", toon_steps: int = 2, shadow: float = 0.55, animation: str = "",
           frame_start: int = 0, frame_end: int = 0, frame_step: int = 1, trim_loop: bool = False,
           margin: float = 0.08, objects: str = "", collection: str = "", pivot=None,
           progress=print) -> dict:
    """Render sprite frames and write manifest.json; returns the manifest."""
    if not out_dir:
        raise SpriteRenderError("out_dir is required")
    if not 1 <= int(size) <= 2048:
        raise SpriteRenderError("size must be between 1 and 2048")
    if not 1 <= int(supersample) <= 16:
        raise SpriteRenderError("supersample must be between 1 and 16")
    if not -90 <= elevation <= 90:
        raise SpriteRenderError("elevation must be between -90 and 90 degrees")
    if shading not in ("toon", "flat", "lit"):
        raise SpriteRenderError("shading must be toon, flat or lit")
    plan_dirs, front_offset = direction_plan(int(directions), only, front)

    imported = load_model(model)
    scene = bpy.context.scene
    objs = character_objects(scene, objects, collection, imported)
    rig_obj = find_rig(objs)
    target = animated_object(objs, rig_obj)
    plans = plan_animations(scene, target, objs, animation, frame_start, frame_end, frame_step, trim_loop)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    state = SceneState(scene)
    try:
        prepare(scene, state, objs, target, shading, toon_steps, shadow, int(supersample), int(size),
                swap_actions=any(action is not None for _, action, _ in plans))
        points, first = measure(scene, objs, target, plans)
        angles = [a + front_offset for _, a in plan_dirs]
        framing = compute_framing(points, first, angles, elevation, margin, int(size), pivot)
        rig, cam, created = build_rig(scene, framing, elevation, shading)
        state.created_objects.extend(created)
        state.created_data.extend([cam.data] + [ob.data for ob in created if ob.type == "LIGHT"])
        scene.camera = cam

        fps = scene.render.fps / scene.render.fps_base
        total = sum(len(f) for _, _, f in plans) * len(plan_dirs)
        done = 0
        animations = []
        for name, action, frames in plans:
            if action is not None:
                assign_action(target, action)
            folder = _safe(name)
            files = {d: [] for d, _ in plan_dirs}
            for f in frames:
                scene.frame_set(f)
                for (d, a), angle in zip(plan_dirs, angles):
                    rig.rotation_euler = (0, 0, math.radians(angle))
                    path = os.path.join(out_dir, folder, d, f"{f:04d}.png")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    scene.render.filepath = path
                    bpy.ops.render.render(write_still=True)
                    files[d].append(path)
                    done += 1
                    if progress:
                        progress(f"SPRITE_RENDER_PROGRESS {done}/{total}")
            step = frames[1] - frames[0] if len(frames) > 1 else 1
            duration = max(1, round(1000 * step / fps))
            animations.append({
                "name": name, "folder": folder, "action": action.name if action else None,
                "frame_start": frames[0], "frame_end": frames[-1], "frame_step": step,
                "frames": frames, "duration_ms": duration, "durations_ms": [duration] * len(frames),
                "files": files,
            })
    finally:
        state.restore()

    # The same frames as one ordered list (animation, then direction, then
    # frame) with 1-based tags, ready for a frames-to-sprite-sheet step.
    sequence, tags = [], []
    for anim in animations:
        for d, _ in plan_dirs:
            first = len(sequence) + 1
            sequence.extend(anim["files"][d])
            tags.append({"name": f"{anim['name']}_{d}", "from": first, "to": len(sequence),
                         "direction": "forward"})

    s = int(size)
    manifest = {
        "format": "puretty-sprite-render/1",
        "source": os.path.abspath(model) if model else bpy.data.filepath,
        "out_dir": out_dir,
        "frame_size": s,
        "supersample": int(supersample),
        "render_size": s * int(supersample),
        "anti_aliased": int(supersample) > 1,
        "elevation": elevation,
        "front_axis": front.upper(),
        "shading": shading,
        "fps": fps,
        "directions": [d for d, _ in plan_dirs],
        "direction_angles": {d: a for d, a in plan_dirs},
        "pivot": {"x": s / 2, "y": framing["ground_row"]},
        "pivot_normalized": {"x": 0.5, "y": framing["ground_row"] / s},
        "pixels_per_unit": s / framing["ortho"],
        "ortho_scale": framing["ortho"],
        "objects": [ob.name for ob in objs],
        "armature": rig_obj.name if rig_obj else None,
        "animations": animations,
        "frame_count": len(sequence),
        "duration_ms": animations[0]["duration_ms"],
        "sequence": sequence,
        "tags": tags,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _parse(argv):
    p = argparse.ArgumentParser(prog="sprite_render.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="", help="model to load: .blend, .glb/.gltf, .fbx, .obj, ...")
    p.add_argument("--out", default="", help="output folder (required unless --list)")
    p.add_argument("--list", action="store_true", help="print the model's animations as JSON and exit")
    p.add_argument("--directions", type=int, default=8, choices=sorted(DIRECTION_NAMES))
    p.add_argument("--only", default="", help="render a subset of directions, e.g. E,W")
    p.add_argument("--size", type=int, default=64, help="final sprite frame size in pixels")
    p.add_argument("--supersample", type=int, default=4,
                   help="render at size*N with anti-aliasing; 1 = exact size, no anti-aliasing")
    p.add_argument("--elevation", type=float, default=30.0, help="camera pitch in degrees (0 = side view)")
    p.add_argument("--front", default="-Y", help="axis the model faces: -Y (Blender), +Y, +X, -X; write --front=-Y")
    p.add_argument("--shading", default="toon", choices=["toon", "flat", "lit"])
    p.add_argument("--toon-steps", type=int, default=2, choices=[2, 3])
    p.add_argument("--shadow", type=float, default=0.55, help="toon shadow brightness 0-1")
    p.add_argument("--animation", default="", help="action name(s), 'all', or 'static'")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--trim-loop", action="store_true", help="drop the last frame (it repeats the first)")
    p.add_argument("--margin", type=float, default=0.08, help="empty border as a fraction of the frame")
    p.add_argument("--objects", default="", help="comma-separated object names to render")
    p.add_argument("--collection", default="", help="render only this collection")
    p.add_argument("--pivot", default="", help="world x,y of the vertical axis to orbit (default: centre)")
    return p.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    try:
        args = _parse(argv)
        if args.list:
            result = list_animations(args.model or None, args.objects, args.collection)
        else:
            if not args.out:
                raise SpriteRenderError("--out is required")
            pivot = [float(v) for v in args.pivot.split(",")] if args.pivot else None
            manifest = render(args.out, args.model or None, args.directions, args.only, args.size,
                              args.supersample, args.elevation, args.front, args.shading,
                              args.toon_steps, args.shadow, args.animation, args.frame_start,
                              args.frame_end, args.frame_step, args.trim_loop, args.margin,
                              args.objects, args.collection, pivot)
            result = {k: manifest[k] for k in ("out_dir", "frame_size", "render_size", "supersample",
                                                "directions", "pivot", "frame_count", "duration_ms",
                                                "elevation", "shading", "tags")}
            result["animations"] = [{"name": a["name"], "frames": len(a["frames"])} for a in manifest["animations"]]
            result["manifest"] = os.path.join(manifest["out_dir"], "manifest.json")
        print("SPRITE_RENDER_RESULT " + json.dumps(result), flush=True)
    except SpriteRenderError as e:
        print(f"SPRITE_RENDER_ERROR {e}", flush=True)
        if bpy.app.background:
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
