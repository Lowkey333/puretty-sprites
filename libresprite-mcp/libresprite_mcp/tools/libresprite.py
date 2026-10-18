import json
import os
import tempfile
from typing import Dict, List

import numpy as np
from PIL import Image

from ..core import ase, render
from ..core import libresprite as ls
from ..core.sprite import ToolError, load
from .. import mcp

# The other tools edit .ase files directly; these use LibreSprite itself.

_SCRIPT_API = """\
LibreSprite runs JavaScript (QuickJS: let/const, arrow functions,
template strings). Globals:
  app       activeSprite, activeImage, activeLayerNumber, activeFrameNumber,
            open(path), version, platform, pixelColor, command
  sprite    width, height, colorMode, layerCount, filename, palette,
            selection, layer(i) (0 = bottom), resize(w, h), save(),
            saveAs(path, asCopy), loadPalette(path)
  layer     name, isVisible, isEditable, isBackground, celCount, cel(frame)
  cel       x, y, frame, image, setPosition(x, y)
  image     width, height, getPixel(x, y), putPixel(x, y, color), clear(color),
            getImageData(), putImageData(bytes), getPNGData()
  pixelColor  rgba(r, g, b, a), rgbaR/G/B/A(c), graya(v, a), grayaV/A(c)
  palette   length, get(i), set(i, r, g, b, a)
  selection select(x, y, w, h), selectAll(), deselect(), bounds
  command   one method per LibreSprite command, e.g. command.Flip({orientation: "horizontal"})
  console.log(...)  is how you return results
Frames and layers are 0-based here. LibreSprite 1.3 cannot run commands
(NewLayer, NewFrame, ...) from a headless script; nightly builds can, and
also offer layer.createCel(frame), sprite.totalFrames and fs/child_process.
"""


@mcp.tool()
async def libresprite_status() -> str:
    """Report whether LibreSprite is installed, its version, and what it can do headless.

    The drawing and animation tools don't need LibreSprite (they edit the
    .ase files directly); open_in_libresprite, run_libresprite_script,
    run_libresprite_cli and check_libresprite_compat do.
    """
    exe = ls.find_executable()
    if not exe:
        return ls.not_found_message()
    ver = ls.version(exe)
    script_exe = ls.find_executable(for_scripts=True)
    script_ver = ls.version(script_exe) if script_exe and script_exe != exe else ver
    lines = [f"LibreSprite {ver or '(version unknown)'} at {exe}"]
    if script_exe != exe:
        lines.append(f"Scripts run with LibreSprite {script_ver} at {script_exe}")
    if ls.is_dev_build(script_ver):
        lines.append("Headless scripts: full API including commands (nightly build).")
    else:
        lines.append("Headless scripts: pixel/palette access only; commands such as NewLayer "
                     "don't run headless in 1.3. Set LIBRESPRITE_SCRIPT_PATH to a nightly build "
                     "(github.com/LibreSprite/LibreSprite/actions) for more.")
    return "\n".join(lines)


@mcp.tool()
async def open_in_libresprite(filename: str) -> str:
    """Open a sprite in the LibreSprite editor (for the human to look at or edit).

    LibreSprite does not notice when a tool changes a file it has open:
    close the file in LibreSprite (or reopen it) after tool edits, and
    don't save it from LibreSprite while tools are editing it, or one
    side's changes are lost.

    Args:
        filename: Sprite (or image) file to open
    """
    if not os.path.isfile(filename):
        return f"File {filename} not found"
    ok, info = ls.open_in_editor(filename)
    if not ok:
        return f"Failed to open LibreSprite: {info}"
    return f"Opened {filename} in LibreSprite ({info})"


_RUN_SCRIPT_DOC = """Execute JavaScript inside headless LibreSprite (escape hatch).

Use this for things only LibreSprite itself can do; every curated tool
works without it. When filename is given it is opened first and is
app.activeSprite. Changes are NOT saved unless the script calls
sprite.save(). console.log() is the only way to return data.

""" + _SCRIPT_API + """
WARNING: scripts run with your user's permissions (and nightly builds
expose file and process APIs). Only pass scripts you trust.

Args:
    script: JavaScript source
    filename: Optional sprite to open before running
    settings: Optional values readable through the script's storage
        (LibreSprite's --set key=value, nightly builds only)
"""


@mcp.tool(description=_RUN_SCRIPT_DOC)
async def run_libresprite_script(script: str, filename: str = "", settings: Dict[str, str] | None = None) -> str:
    if not script.strip():
        return "Script cannot be empty"
    if filename and not os.path.isfile(filename):
        return f"File {filename} not found"
    ok, output = ls.run_script(script, filename or None, settings=settings)
    if ok:
        return output if output.strip() else "Script executed (no output printed)"
    return f"Script failed: {output}"


@mcp.tool()
async def run_libresprite_cli(args: List[str], timeout_seconds: int = 120) -> str:
    """Run LibreSprite's command line in batch mode (no window).

    "--batch" is added automatically. Useful options: --sheet <png>,
    --data <json>, --format json-hash|json-array, --sheet-type
    horizontal|vertical|rows|columns|packed, --split-layers, --layer <name>,
    --all-layers, --frame-tag <name>, --frame-range from,to, --trim,
    --ignore-empty, --shape-padding/--border-padding/--inner-padding <n>,
    --scale <factor>, --save-as <file>, --list-layers, --list-tags. Options
    apply to the files that come after them (e.g. --frame-tag before the
    sprite), except --scale and --save-as, which apply to files before them.

    Example: ["--list-tags", "--format", "json-array", "hero.ase",
              "--sheet", "hero.png", "--data", "hero.json"]

    Args:
        args: Arguments after the executable name
        timeout_seconds: Give up after this many seconds (default 120)
    """
    full = list(args)
    if not any(a in ("-b", "--batch") for a in full):
        full.insert(0, "--batch")
    ok, output = ls.run(full, timeout=timeout_seconds)
    if not ok:
        return f"Failed to run LibreSprite: {output}"
    return output or "LibreSprite finished (no output)"


_INSPECT_JS = """
var s = app.activeSprite;
var out = {width: s.width, height: s.height, colorMode: s.colorMode, layers: []};
for (var i = 0; i < s.layerCount; i++) {
  var l = s.layer(i), cels = [];
  for (var f = 0; f < %d; f++) {
    var c = l.cel(f);
    if (c) cels.push([f, c.x, c.y, c.image.width, c.image.height]);
  }
  out.layers.push({name: l.name, visible: l.isVisible, background: l.isBackground, cels: cels});
}
console.log("INSPECT:" + JSON.stringify(out));
"""


@mcp.tool()
async def check_libresprite_compat(filename: str) -> str:
    """Load a sprite in real LibreSprite and compare it with what the tools see.

    Checks layers, cels and frame data through LibreSprite's script API,
    then renders every frame with LibreSprite's exporter and compares the
    pixels with this server's renderer. Use it after unusual edits or on
    files from other programs (e.g. Aseprite files with groups or slices).

    Args:
        filename: Sprite file to check
    """
    try:
        spr = load(filename)
    except ToolError as e:
        return f"Failed to check compatibility: {e}"
    problems, notes = [], list(spr.warnings)

    ok, output = ls.run_script(_INSPECT_JS % spr.frame_count, filename)
    if not ok:
        return f"Failed to check compatibility: {output}"
    line = next((l for l in output.splitlines() if l.startswith("INSPECT:")), None)
    if line is None:
        return f"Failed to check compatibility: LibreSprite printed no report ({output[:300]})"
    seen = json.loads(line[len("INSPECT:"):])
    if (seen["width"], seen["height"]) != (spr.width, spr.height):
        problems.append(f"size: LibreSprite {seen['width']}x{seen['height']}, tools {spr.width}x{spr.height}")
    ours = [l.name for l in spr.layers]
    theirs = [l["name"] for l in seen["layers"]]
    if ours != theirs:
        problems.append(f"layers: LibreSprite {theirs}, tools {ours}")
    else:
        for layer, other in zip(spr.layers, seen["layers"]):
            mine = sorted([f, c.x, c.y, c.width, c.height] for f, c in layer.cels.items())
            if mine != sorted(other["cels"]):
                problems.append(f"cels of '{layer.name}' differ")
            if layer.visible != other["visible"]:
                problems.append(f"visibility of '{layer.name}' differs")

    if spr.color_mode == ase.INDEXED:
        notes.append("Pixel comparison skipped: LibreSprite's sheet exporter writes indexed "
                     "sprites by palette index (ignoring opacity), unlike its editor view")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            sheet = os.path.join(tmp, "sheet.png")
            ok, output = ls.run(["--batch", os.path.abspath(filename), "--sheet", sheet,
                                 "--sheet-type", "horizontal"])
            if not ok or not os.path.isfile(sheet):
                problems.append(f"LibreSprite could not export the sprite: {output[:300]}")
            else:
                ref = np.array(Image.open(sheet).convert("RGBA")).astype(int)
                for frame in range(spr.frame_count):
                    mine = render.composite(spr, frame).astype(int)
                    other = ref[:, frame * spr.width:(frame + 1) * spr.width]
                    clear = (mine[..., 3] == 0) & (other[..., 3] == 0)
                    bad = (np.any(mine != other, axis=-1) & ~clear).sum() if other.shape == mine.shape else -1
                    if bad:
                        problems.append(f"frame {frame + 1}: {bad if bad > 0 else 'all'} pixels render differently")

    verdict = "OK: LibreSprite sees the same sprite" if not problems else "MISMATCH:\n- " + "\n- ".join(problems)
    summary = (f"{verdict}\n{spr.width}x{spr.height} {spr.color_mode}, {spr.frame_count} frames, "
               f"{len(spr.layers)} layers, {len(spr.tags)} tags")
    if notes:
        summary += "\nNotes:\n- " + "\n- ".join(notes)
    return summary
