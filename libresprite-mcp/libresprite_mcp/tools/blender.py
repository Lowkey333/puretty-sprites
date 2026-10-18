"""Render 3D characters into sprite frames with headless Blender.

The rendering itself is integrations/blender/sprite_render.py, which also
runs inside a live Blender (for example through blender-mcp). These tools
run it in the background with --factory-startup, so user add-ons and
preferences never affect a render.

Blender is found through BLENDER_PATH (blender.exe or its folder, also read
from libresprite-mcp/.env), PATH, and the usual install folders.
"""
import asyncio
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import dotenv

from ..core.sprite import reject_traversal
from .. import mcp

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CHECKOUT_DIR = os.path.dirname(_PROJECT_DIR)
dotenv.load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

_EXE = "blender.exe" if os.name == "nt" else "blender"
MODEL_EXTENSIONS = (".blend", ".glb", ".gltf", ".fbx", ".obj", ".ply", ".stl",
                    ".usd", ".usda", ".usdc", ".usdz", ".abc")
ADDON_INSTALL_HINT = ("uvx mcp-for-blender install-addon, then enable 'Interface: MCP for Blender' "
                      "in Edit > Preferences > Add-ons")


def sprite_render_script() -> str:
    return os.getenv("SPRITE_RENDER_SCRIPT") or os.path.join(
        _CHECKOUT_DIR, "integrations", "blender", "sprite_render.py")


def _version_key(path: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", os.path.basename(os.path.dirname(path))))


def find_blender() -> str | None:
    """Path to the Blender executable, newest install first, or None."""
    candidates = []
    configured = os.getenv("BLENDER_PATH", "").strip().strip('"')
    if configured:
        candidates.append(os.path.join(configured, _EXE) if os.path.isdir(configured) else configured)
    on_path = shutil.which("blender")
    if on_path:
        candidates.append(on_path)
    if os.name == "nt":
        for base in (os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)"),
                     os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs")):
            if base:
                found = glob.glob(os.path.join(base, "Blender Foundation", "Blender*", _EXE))
                candidates += sorted(found, key=_version_key, reverse=True)
        steam = os.path.join(os.getenv("ProgramFiles(x86)", ""), "Steam", "steamapps", "common", "Blender", _EXE)
        candidates.append(steam)
    elif sys.platform == "darwin":
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    else:
        candidates += ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def _not_found() -> str:
    return ("Blender was not found. Install it from https://www.blender.org/download/ and set "
            "BLENDER_PATH to blender.exe or its folder (in the MCP server's environment or in "
            f"{os.path.join(_PROJECT_DIR, '.env')}).")


async def _run(args: list[str], timeout: float) -> tuple[int | None, str]:
    """Run Blender without blocking the server; returns (exit code, output)."""
    exe = find_blender()
    if not exe:
        return None, _not_found()
    try:
        proc = await asyncio.create_subprocess_exec(
            exe, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as e:
        return None, f"Could not start Blender ({exe}): {e}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None, f"Blender did not finish within {timeout:g}s"
    return proc.returncode, out.decode("utf-8", errors="replace")


def _parse_result(code: int | None, output: str) -> dict:
    """The JSON sprite_render.py printed, or raise RuntimeError with the reason."""
    if code is None:
        raise RuntimeError(output)
    for line in reversed(output.splitlines()):
        if line.startswith("SPRITE_RENDER_ERROR "):
            raise RuntimeError(line[len("SPRITE_RENDER_ERROR "):])
        if line.startswith("SPRITE_RENDER_RESULT "):
            return json.loads(line[len("SPRITE_RENDER_RESULT "):])
    tail = "\n".join(l for l in output.splitlines() if l.strip())[-1500:]
    raise RuntimeError(f"Blender exited with code {code} without a result:\n{tail}")


def _script_args(model_path: str) -> tuple[list[str], list[str]]:
    """Blender arguments before -P, and script arguments after --."""
    model = os.path.abspath(model_path)
    if model.lower().endswith(".blend"):
        return ["-b", "--factory-startup", "--python-exit-code", "1", model], []
    return ["-b", "--factory-startup", "--python-exit-code", "1"], [f"--model={model}"]


def _check_model(model_path: str) -> str | None:
    if not model_path or not os.path.isfile(model_path):
        return f"Model {model_path} not found"
    if not model_path.lower().endswith(MODEL_EXTENSIONS):
        return f"Unsupported model format: use one of {', '.join(MODEL_EXTENSIONS)}"
    if not os.path.isfile(sprite_render_script()):
        return f"sprite_render.py not found at {sprite_render_script()} (set SPRITE_RENDER_SCRIPT)"
    return None


def _addon_file(version: str | None) -> str | None:
    match = re.match(r"(\d+)\.(\d+)", version or "")
    if not match or os.name != "nt":
        return None
    folder = os.path.join(os.getenv("APPDATA", ""), "Blender Foundation", "Blender",
                          f"{match.group(1)}.{match.group(2)}", "scripts", "addons")
    return os.path.join(folder, "blender_mcp.py")


@mcp.tool()
async def blender_status() -> str:
    """Report the Blender install used to render 3D characters into sprites.

    Also says whether the MCP for Blender (blender-mcp) add-on is installed
    and enabled; that add-on is what lets an AI agent work in a live
    Blender (modelling, image-to-3D generation), separately from these
    headless render tools.
    """
    exe = find_blender()
    if not exe:
        return _not_found()
    code, out = await _run(["--version"], 60)
    version = None
    if code == 0:
        match = re.search(r"Blender\s+(\d+\.\d+(?:\.\d+)?(?:\s+LTS)?)", out)
        version = match.group(1) if match else None
    lines = [f"Blender: {exe}", f"Version: {version or 'unknown'}",
             f"Sprite render script: {sprite_render_script()}"
             + ("" if os.path.isfile(sprite_render_script()) else " (MISSING)")]
    addon = _addon_file(version)
    if addon and os.path.isfile(addon):
        code, out = await _run(["-b", "--python-expr",
                                "import addon_utils; print('ADDON_STATE', addon_utils.check('blender_mcp'))"], 120)
        state = re.search(r"ADDON_STATE \((\w+), (\w+)\)", out or "")
        enabled = state and state.group(2) == "True"
        lines.append(f"blender-mcp add-on: installed at {addon}, "
                     + ("enabled" if enabled else "not enabled (Edit > Preferences > Add-ons > MCP for Blender)"))
    elif addon:
        lines.append(f"blender-mcp add-on: not installed ({ADDON_INSTALL_HINT})")
    return "\n".join(lines)


@mcp.tool()
async def blender_list_animations(model_path: str) -> str:
    """List a 3D model's animations (actions) without rendering.

    Returns JSON: the character's objects and armature, each animation's
    frame range and whether it loops, the scene fps and the model size in
    Blender units. Use the names with blender_render_sprites(animation=...).

    Args:
        model_path: .blend, .glb/.gltf, .fbx, .obj, .ply, .stl, .usd or .abc file
    """
    err = _check_model(model_path)
    if err:
        return err
    before, after = _script_args(model_path)
    code, out = await _run(before + ["-P", sprite_render_script(), "--"] + after + ["--list"], 600)
    try:
        return json.dumps(_parse_result(code, out), indent=2)
    except RuntimeError as e:
        return f"Failed to list animations: {e}"


@mcp.tool()
async def blender_render_sprites(
    model_path: str,
    output_dir: str,
    directions: int = 8,
    frame_size: int = 64,
    elevation: float = 30,
    animation: str = "",
    frame_start: int = 0,
    frame_end: int = 0,
    frame_step: int = 1,
    shading: str = "toon",
    supersample: int = 4,
    direction_names: str = "",
    front_axis: str = "-Y",
    trim_loop: bool = False,
    margin: float = 0.08,
    toon_steps: int = 2,
    objects: str = "",
    timeout_seconds: int = 3600,
) -> str:
    """Render a 3D character into sprite frames from several directions with Blender.

    An orthographic camera orbits the character with lights fixed to the
    screen (from the upper left), at one shared scale, so every frame and
    direction lines up on the same ground point. Frames are written to
    <output_dir>/<animation>/<direction>/<frame>.png with manifest.json,
    which lists them per animation and direction and also as one ordered
    "sequence" with matching "tags" (e.g. Walk_S) for frames_to_sprite.

    Typical flow: concept art -> 3D model (blender-mcp generate_3d, Hunyuan3D,
    Rodin, Meshy...) -> rig/animate (Mixamo, Rigify) -> this tool ->
    frames_to_sprite(image_paths=manifest sequence, width=frame_size,
    height=frame_size, tags=manifest tags, duration_ms=manifest duration_ms).

    Args:
        model_path: .blend, .glb/.gltf, .fbx, .obj, .ply, .stl, .usd or .abc file
        output_dir: Folder for the frames and manifest.json
        directions: 1, 2, 4, 8 (default) or 16. Named by where the character
            faces on screen: S (towards the viewer), SW, W, NW, N, NE, E, SE
        frame_size: Final sprite size in pixels (default 64)
        elevation: Camera pitch in degrees: 0 = side view, 30 = classic
            top-down RPG (default), 90 = straight down
        animation: Action name(s) separated by commas, "all", "static", or ""
            for the active action (static pose if none); see blender_list_animations
        frame_start: First frame (0 = the action's start)
        frame_end: Last frame (0 = the action's end)
        frame_step: Render every Nth frame (default 1); 2-3 suits pixel art
        shading: "toon" (2-3 flat light bands, default), "flat" (unlit base
            colors) or "lit" (the model's own materials, smooth light)
        supersample: Render at frame_size*N with anti-aliasing for clean
            downscaling (default 4); 1 renders exactly frame_size with hard
            edges and no anti-aliasing
        direction_names: Render only these directions, e.g. "E" or "E,W" for a side-scroller
        front_axis: Axis the model faces: "-Y" (Blender/glTF default), "+Y", "+X", "-X"
        trim_loop: Drop the last frame when it repeats the first (glTF/FBX
            loops; Blender actions marked cyclic are trimmed automatically)
        margin: Empty border around the character as a fraction of the frame (default 0.08)
        toon_steps: 2 or 3 light bands for toon shading
        objects: Comma-separated object names to render (default: every visible mesh)
        timeout_seconds: Give up after this long (default 3600)
    """
    err = _check_model(model_path)
    if err:
        return err
    err = reject_traversal(output_dir)
    if err:
        return err
    if directions not in (1, 2, 4, 8, 16):
        return "directions must be 1, 2, 4, 8 or 16"
    if not 1 <= frame_size <= 2048:
        return "frame_size must be between 1 and 2048"
    if not 1 <= supersample <= 16:
        return "supersample must be between 1 and 16"
    if shading not in ("toon", "flat", "lit"):
        return "shading must be toon, flat or lit"
    if front_axis.upper() not in ("-Y", "+Y", "+X", "-X"):
        return "front_axis must be -Y, +Y, +X or -X"
    if toon_steps not in (2, 3):
        return "toon_steps must be 2 or 3"
    if frame_step < 1:
        return "frame_step must be >= 1"

    before, after = _script_args(model_path)
    # --key=value keeps argparse from reading values like "-Y" or "-10" as options.
    options = {"out": os.path.abspath(output_dir), "directions": directions, "size": frame_size,
               "supersample": supersample, "elevation": elevation, "front": front_axis.upper(),
               "shading": shading, "toon-steps": toon_steps, "frame-start": frame_start,
               "frame-end": frame_end, "frame-step": frame_step, "margin": margin,
               "animation": animation, "only": direction_names, "objects": objects}
    script_args = after + [f"--{key}={value}" for key, value in options.items() if value != ""]
    if trim_loop:
        script_args.append("--trim-loop")

    code, out = await _run(before + ["-P", sprite_render_script(), "--"] + script_args, timeout_seconds)
    try:
        result = _parse_result(code, out)
    except RuntimeError as e:
        return f"Failed to render sprites: {e}"

    anims = ", ".join(f"{a['name']} ({a['frames']} frames)" for a in result["animations"])
    size = result["frame_size"]
    tags = ", ".join(f"{t['name']} {t['from']}-{t['to']}" for t in result["tags"][:24])
    if len(result["tags"]) > 24:
        tags += f", ... ({len(result['tags'])} tags in the manifest)"
    return "\n".join([
        f"Rendered {result['frame_count']} frames: {len(result['directions'])} directions "
        f"({', '.join(result['directions'])}) x {anims}.",
        f"Frame PNGs are {result['render_size']}x{result['render_size']} "
        + (f"(supersample {result['supersample']}: downscale to {size}x{size})" if result["supersample"] > 1
           else "(exact size, no anti-aliasing)")
        + f", {result['shading']} shading, {result['elevation']:g} degree elevation.",
        f"Ground pivot of a {size}x{size} frame: x={result['pivot']['x']:g}, y={result['pivot']['y']}.",
        f"Manifest: {result['manifest']}",
        f"Sequence tags: {tags}",
        f"Next: frames_to_sprite(output_filename=..., manifest=\"{result['manifest']}\") builds the "
        f"{size}x{size} sprite with these tags, durations and the ground pivot (add palette=... "
        f"or outline_color=... as needed).",
    ])


def auto_rig_script() -> str:
    return os.path.join(_CHECKOUT_DIR, "integrations", "blender", "auto_rig.py")


@mcp.tool()
async def blender_auto_rig(
    model_path: str,
    output_blend: str,
    landmarks: str = "",
    target_faces: int = 40000,
    timeout: int = 900,
) -> str:
    """Rig a generated humanoid model and give it Idle and Walk animations.

    For image-to-3D output (TRELLIS.2, Hunyuan3D, ...): one dense unrigged
    mesh. Decimates it, places a humanoid skeleton, skins it with
    anatomy-limited distance weights (Blender's heat weights fail on most
    generated meshes) and adds in-place "Idle" and "Walk" actions, then
    saves a .blend ready for blender_render_sprites(animation="all").
    Suits characters standing in an A- or T-pose; a rough rig meant for
    sprites, not close-ups (use Mixamo or Rigify for production).

    Args:
        model_path: .glb/.gltf/.fbx/.obj/.blend with the character
        output_blend: Where to save the rigged .blend
        landmarks: Optional overrides when a part follows the wrong bone, as
            height fractions (0 = feet, 1 = top) and width fractions of the
            height, e.g. "hips=0.25,neck=0.5,shoulder=0.45,leg_x=0.07".
            Keys: ankle, knee, hips, chest, neck, head_top, shoulder, hand,
            leg_x, shoulder_x, hand_x
        target_faces: Decimate to about this many faces first (0 = keep)
        timeout: Seconds to wait (default 900)
    """
    error = _check_model(model_path)
    if error:
        return f"Failed to rig: {error}"
    if not os.path.isfile(auto_rig_script()):
        return f"Failed to rig: auto_rig.py not found at {auto_rig_script()}"
    err = reject_traversal(output_blend)
    if err:
        return err
    args = ["-b", "--factory-startup", "--python-exit-code", "1", "-P", auto_rig_script(), "--",
            f"--model={os.path.abspath(model_path)}", f"--out={os.path.abspath(output_blend)}",
            f"--faces={target_faces}"]
    if landmarks:
        args.append(f"--landmarks={landmarks}")
    code, output = await _run(args, timeout)
    result = None
    for line in reversed(output.splitlines()):
        if line.startswith("AUTO_RIG_RESULT "):
            result = json.loads(line[len("AUTO_RIG_RESULT "):])
            break
    if result is None:
        tail = "\n".join(l for l in output.splitlines() if l.strip())[-1500:]
        return f"Failed to rig (exit code {code}):\n{tail}"
    marks = ", ".join(f"{k}={v}" for k, v in result["landmarks"].items())
    return "\n".join([
        f"Rigged {os.path.basename(model_path)} -> {result['blend']}",
        f"Mesh {result['faces_before']} -> {result['faces']} faces; weights: {result['weights']}",
        f"Actions: {', '.join(result['actions'])} at {result['fps']} fps",
        f"Landmarks used: {marks}",
        "Check the deformation with blender_render_sprites(..., shading=\"lit\", frame_size=192) and "
        "adjust landmarks if a part follows the wrong bone.",
    ])
