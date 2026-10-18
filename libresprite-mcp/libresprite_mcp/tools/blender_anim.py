"""Add motion-capture animations to a rigged character with headless Blender.

The retargeting itself is integrations/blender/retarget.py, which also runs
inside a live Blender. This tool runs it in the background on a .blend made
by blender_auto_rig and adds one action per call, so a character can get
real runs, fights and falls next to the auto-rig's own animations.
"""
import json
import os
import uuid

from ..core.sprite import reject_traversal
from .. import mcp
from .blender import _CHECKOUT_DIR, _run

MOTION_EXTENSIONS = (".bvh", ".fbx")


def retarget_script() -> str:
    return os.path.join(_CHECKOUT_DIR, "integrations", "blender", "retarget.py")


def _summary(result: dict, blend: str) -> str:
    loop = result.get("loop")
    if loop:
        timing = (f"looping: a {loop['seconds']:g} s cycle from {loop['start']:g} s of the clip, "
                  f"played at {loop['speed']:g}x speed")
    else:
        clip = result.get("clip") or {}
        timing = f"one-shot: {clip.get('start', 0):g}-{clip.get('end', 0):g} s of the clip"
    source = result["source"]
    lines = [
        f"Added \"{result['action']}\" to {blend}: {result['frames']} frames at {result['fps']:g} fps, {timing}"
        + (", in place." if result["in_place"] else ", with root motion."),
        f"Source: {source['file']} ({source['naming']} naming, {source['fps']:g} fps, {source['seconds']:g} s); "
        f"turned {result['heading_turned_deg']:g} deg to face -Y "
        f"({'by its starting stance' if result['heading_from'] == 'start' else 'by its average hip line'}); "
        f"motion scaled x{result['scale']:g} to the character's legs.",
        f"Bones driven: {', '.join(result['mapped_bones'])}.",
    ]
    if result["missing_bones"]:
        lines.append(f"Not in this rig (skipped): {', '.join(result['missing_bones'])}.")
    if result["unmapped_bones"]:
        lines.append(f"No matching source bones (left at rest): {', '.join(result['unmapped_bones'])}.")
    if result["replaced"]:
        lines.append(f"Replaced the previous \"{result['action']}\" action.")
    for warning in result.get("warnings", []):
        lines.append(f"Warning: {warning}")
    lines.append(f"Check it with blender_render_sprites(model_path=\"{blend}\", animation=\"{result['action']}\", "
                 "shading=\"lit\", frame_size=192), then render the sprite frames with your usual settings.")
    lines.append("Result: " + json.dumps(result))
    return "\n".join(lines)


@mcp.tool()
async def blender_add_animation(
    rigged_blend: str,
    source_path: str,
    name: str,
    start_seconds: float = 0,
    end_seconds: float = 0,
    loop: bool = True,
    in_place: bool = True,
    fps: float = 12,
    output_blend: str = "",
) -> str:
    """Add a motion-capture animation (BVH or Mixamo FBX) to a rigged character.

    Retargets the clip onto the skeleton made by blender_auto_rig and saves it
    as a new action; render it with blender_render_sprites(animation="Name").
    Call it once per move. An action with the same name is replaced, and
    blender_auto_rig already made Idle, Walk, Run, Attack, Cast, Hit, Death
    and Jump: name clips e.g. "MocapRun" to keep both, or "Run" to swap one.
    A clip that ends lying down widens the camera framing for every
    animation rendered with it; render such falls on their own if needed. The character is turned to face -Y, root motion is
    scaled to its legs and its feet are kept on the floor; the spine, head and
    limbs follow the clip, so a T-pose clip works on an A-pose character. If
    the .blend is open in Blender, reload it there (File > Revert) afterwards.

    Where to get clips:
    - Mixamo (mixamo.com, free Adobe account): pick an animation, tick
      "In Place" for walks and runs, and Download as FBX with Skin:
      "Without Skin". Any Mixamo character works; the rig never matters.
    - CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu, 2500+ clips:
      runs, jumps, fights, falls, dances). Free to use in commercial products;
      don't redistribute the raw files. BVH conversion with an index of all
      clips: github.com/Shriinivas/cmubvh (the "cgspeed" release).
    - Other BVH exports (Rokoko, DeepMotion, Plask, ...) usually work too:
      bone names are matched for Mixamo, CMU and common rigs.

    Args:
        rigged_blend: .blend from blender_auto_rig (armature "CharacterRig")
        source_path: .bvh or .fbx motion file
        name: Action name, e.g. "Run"
        start_seconds: Where to start in the clip, in seconds
        end_seconds: Where to end (0 = the end of the clip). Pick the move
            out of a longer take with these.
        loop: True finds the most seamless 0.4-1.6 s cycle between start and
            end and keys it to repeat (walks, runs, idles; the frame count
            is fitted to the cycle). False keeps start-end as a one-shot
            (attacks, jumps, deaths).
        in_place: True removes the horizontal travel so the character stays
            on its spot (the vertical bob stays, loops keep their sway).
            False keeps the clip's root motion, scaled to the character.
        fps: Frame rate to sample at; keep the scene's (12 for blender_auto_rig)
            so sprite durations come out right
        output_blend: Where to save (default: update rigged_blend in place)
    """
    def fail(message: str) -> str:
        return f"Failed to add animation: {message}"

    if not rigged_blend or not os.path.isfile(rigged_blend) or not rigged_blend.lower().endswith(".blend"):
        return fail(f"rigged .blend {rigged_blend} not found (make one with blender_auto_rig)")
    if not source_path or not os.path.isfile(source_path):
        return fail(f"motion file {source_path} not found")
    if not source_path.lower().endswith(MOTION_EXTENSIONS):
        return fail("use a .bvh file or a Mixamo .fbx (downloaded \"Without Skin\")")
    if not name or not name.strip():
        return fail("give the action a name, e.g. \"Run\"")
    if fps <= 0 or start_seconds < 0 or end_seconds < 0 or (end_seconds and end_seconds <= start_seconds):
        return fail("fps must be positive and 0 <= start_seconds < end_seconds (or end_seconds = 0)")
    for path in (rigged_blend, source_path, output_blend):
        err = reject_traversal(path) if path else None
        if err:
            return err
    if not os.path.isfile(retarget_script()):
        return fail(f"retarget.py not found at {retarget_script()}")

    target = os.path.abspath(output_blend or rigged_blend)
    if not target.lower().endswith(".blend"):
        return fail("output_blend must end in .blend")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    # Save next to the target and swap it in only once Blender has succeeded.
    temp = os.path.join(os.path.dirname(target), f".{os.path.basename(target)}.{uuid.uuid4().hex[:8]}.tmp.blend")
    args = ["-b", "--factory-startup", "--python-exit-code", "1", os.path.abspath(rigged_blend),
            "-P", retarget_script(), "--",
            f"--source={os.path.abspath(source_path)}", f"--name={name.strip()}",
            f"--start={start_seconds}", f"--end={end_seconds}", f"--fps={fps}", f"--out={temp}"]
    if loop:
        args.append("--loop")
    if in_place:
        args.append("--in-place")
    try:
        code, output = await _run(args, 600)
        result, error = None, None
        for line in reversed(output.splitlines()):
            if line.startswith("RETARGET_RESULT "):
                result = json.loads(line[len("RETARGET_RESULT "):])
                break
            if line.startswith("RETARGET_ERROR "):
                error = line[len("RETARGET_ERROR "):]
                break
        if error:
            return fail(error)
        if result is None or not os.path.isfile(temp):
            tail = "\n".join(l for l in output.splitlines() if l.strip())[-1500:]
            return fail(f"Blender exited with code {code} without a result:\n{tail}")
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    result["blend"] = target
    return _summary(result, target)
