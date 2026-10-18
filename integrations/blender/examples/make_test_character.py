"""Build test_character.blend: a small rigged figure with Idle and Walk actions.

    blender -b --factory-startup -P make_test_character.py

The figure is about 1.8 units tall, stands on z=0 and faces -Y (Blender's
front view), with a nose, eyes and toes on the front and hair on the back,
so sprite directions are easy to tell apart. Its parts are primitives
skinned rigidly to an armature through vertex groups, like a real rigged
character, and both actions loop (their last key repeats the first).
"""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_character.blend")


def srgb(hex_color):
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    h = hex_color.lstrip("#")
    return (lin(int(h[0:2], 16)), lin(int(h[2:4], 16)), lin(int(h[4:6], 16)), 1.0)


def material(name, hex_color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = srgb(hex_color)
    bsdf.inputs["Roughness"].default_value = 0.8
    mat.diffuse_color = srgb(hex_color)
    return mat


def part(name, kind, size, location, bone, mat, rig):
    """A primitive mesh weighted fully to one bone."""
    bm = bmesh.new()
    if kind == "box":
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    elif kind == "sphere":
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=size[0])
        bmesh.ops.scale(bm, vec=Vector((1, size[1] / size[0], size[2] / size[0])), verts=bm.verts)
    else:  # cylinder along Z
        bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=size[0], radius2=size[0], depth=size[2])
    bmesh.ops.translate(bm, vec=Vector(location), verts=bm.verts)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = rig
    ob.vertex_groups.new(name=bone).add(range(len(mesh.vertices)), 1.0, "REPLACE")
    ob.modifiers.new("Armature", "ARMATURE").object = rig
    return ob


def build():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    scene = bpy.context.scene
    scene.render.fps = 12
    scene.frame_start, scene.frame_end = 1, 24

    arm_data = bpy.data.armatures.new("HeroRig")
    rig = bpy.data.objects.new("HeroRig", arm_data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {}

    def bone(name, head, tail, parent=None):
        b = arm_data.edit_bones.new(name)
        b.head, b.tail = head, tail
        if parent:
            b.parent = bones[parent]
        bones[name] = b

    bone("hips", (0, 0, 0.9), (0, 0, 1.0))
    bone("spine", (0, 0, 1.0), (0, 0, 1.45), "hips")
    bone("head", (0, 0, 1.45), (0, 0, 1.8), "spine")
    bone("arm.L", (0.27, 0, 1.4), (0.27, 0, 0.95), "spine")
    bone("arm.R", (-0.27, 0, 1.4), (-0.27, 0, 0.95), "spine")
    bone("leg.L", (0.12, 0, 0.9), (0.12, 0, 0.05), "hips")
    bone("leg.R", (-0.12, 0, 0.9), (-0.12, 0, 0.05), "hips")
    bpy.ops.object.mode_set(mode="OBJECT")

    skin = material("Skin", "#F1C27D")
    shirt = material("Shirt", "#3B5DC9")
    pants = material("Pants", "#333C57")
    shoes = material("Shoes", "#5D3A1A")
    hair = material("Hair", "#B13E53")
    eyes = material("Eyes", "#1A1C2C")

    part("Torso", "box", (0.42, 0.24, 0.5), (0, 0, 1.2), "spine", shirt, rig)
    part("Pelvis", "box", (0.4, 0.22, 0.16), (0, 0, 0.93), "hips", pants, rig)
    part("Head", "sphere", (0.17, 0.17, 0.19), (0, 0, 1.62), "head", skin, rig)
    part("Hair", "box", (0.36, 0.22, 0.18), (0, 0.06, 1.75), "head", hair, rig)
    part("Nose", "box", (0.05, 0.08, 0.05), (0, -0.18, 1.6), "head", skin, rig)
    part("Eye.L", "box", (0.04, 0.03, 0.05), (0.07, -0.16, 1.66), "head", eyes, rig)
    part("Eye.R", "box", (0.04, 0.03, 0.05), (-0.07, -0.16, 1.66), "head", eyes, rig)
    for side, x in (("L", 0.27), ("R", -0.27)):
        part(f"Arm.{side}", "cylinder", (0.06, 0.06, 0.45), (x, 0, 1.17), f"arm.{side}", shirt, rig)
        part(f"Hand.{side}", "sphere", (0.065, 0.065, 0.065), (x, 0, 0.92), f"arm.{side}", skin, rig)
    for side, x in (("L", 0.12), ("R", -0.12)):
        part(f"Leg.{side}", "cylinder", (0.08, 0.08, 0.82), (x, 0, 0.47), f"leg.{side}", pants, rig)
        part(f"Foot.{side}", "box", (0.13, 0.24, 0.08), (x, -0.06, 0.04), f"leg.{side}", shoes, rig)

    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
    rig.animation_data_create()

    def key(action, frame, poses):
        for name, (rx, z) in poses.items():
            pb = rig.pose.bones[name]
            pb.rotation_euler = (math.radians(rx), 0, 0)
            pb.location = (0, z, 0) if name == "hips" else (0, 0, 0)
            pb.keyframe_insert("rotation_euler", frame=frame)
            if name == "hips":
                pb.keyframe_insert("location", frame=frame)

    def make_action(name, keys):
        action = bpy.data.actions.new(name)
        action.use_fake_user = True
        rig.animation_data.action = action
        for frame, poses in keys:
            key(action, frame, poses)
        action.use_frame_range = True
        action.frame_start, action.frame_end = keys[0][0], keys[-1][0]
        action.use_cyclic = True
        return action

    # Hips "location" y is along the bone, i.e. world up: a bob.
    def walk_pose(s):
        return {"hips": (0, 0.0 if s == 0 else 0.04), "leg.L": (30 * s, 0), "leg.R": (-30 * s, 0),
                "arm.L": (-25 * s, 0), "arm.R": (25 * s, 0), "spine": (0, 0), "head": (0, 0)}

    walk = make_action("Walk", [(1, walk_pose(1)), (7, walk_pose(0)), (13, walk_pose(-1)),
                                (19, walk_pose(0)), (25, walk_pose(1))])
    idle_rest = {"hips": (0, 0.0), "arm.L": (3, 0), "arm.R": (-3, 0), "head": (0, 0)}
    idle_up = {"hips": (0, 0.015), "arm.L": (-3, 0), "arm.R": (3, 0), "head": (4, 0)}
    make_action("Idle", [(1, idle_rest), (13, idle_up), (25, idle_rest)])
    rig.animation_data.action = walk

    bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
    print("TEST_CHARACTER_SAVED", OUT)


build()
