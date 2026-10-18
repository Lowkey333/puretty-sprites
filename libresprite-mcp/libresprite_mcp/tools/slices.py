import json
import os

from ..core.sprite import ToolError, load
from .. import mcp

# LibreSprite's .ase format has no slices (it drops Aseprite's slice
# chunks), so slices live in a sidecar "<sprite>.slices.json" next to the
# sprite. export_spritesheet copies them into the JSON data's meta.slices
# in Aseprite's format, which is where game engines read pivots and
# 9-patch centers from. Moving or renaming the sprite needs the sidecar
# moved along with it.


def sidecar_path(filename: str) -> str:
    return os.path.splitext(filename)[0] + ".slices.json"


def load_slices(filename: str) -> list[dict]:
    path = sidecar_path(filename)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("slices", [])


def _save_slices(filename: str, slices: list[dict]) -> None:
    path = sidecar_path(filename)
    if not slices:
        if os.path.isfile(path):
            os.remove(path)
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sprite": os.path.basename(filename), "slices": slices}, f, indent=1)


def aseprite_meta(filename: str) -> list[dict]:
    """Slices in the shape Aseprite writes to a sheet's meta.slices."""
    out = []
    for s in load_slices(filename):
        key = {"frame": 0, "bounds": {"x": s["x"], "y": s["y"], "w": s["width"], "h": s["height"]}}
        if s.get("center"):
            c = s["center"]
            key["center"] = {"x": c["x"], "y": c["y"], "w": c["width"], "h": c["height"]}
        if s.get("pivot"):
            key["pivot"] = dict(s["pivot"])
        out.append({"name": s["name"], "color": "#0000ffff", "keys": [key]})
    return out


def _get(slices: list[dict], name: str) -> dict:
    for s in slices:
        if s["name"] == name:
            return s
    raise ToolError(f"Slice not found: {name!r}")


@mcp.tool()
async def create_slice(filename: str, name: str, x: int, y: int, width: int, height: int) -> str:
    """Create a named slice (a rectangular region usable by game engines).

    Stored in "<sprite>.slices.json" because LibreSprite files cannot hold
    slices; export_spritesheet includes them in its JSON data.

    Args:
        filename: Sprite file the slice belongs to
        name: Slice name (replaces an existing slice with the same name)
        x: Left edge in sprite pixels
        y: Top edge in sprite pixels
        width: Slice width
        height: Slice height
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        load(filename)
        slices = [s for s in load_slices(filename) if s["name"] != name]
        slices.append({"name": name, "x": x, "y": y, "width": width, "height": height})
        _save_slices(filename, slices)
    except (ToolError, OSError, ValueError) as e:
        return f"Failed to create slice: {e}"
    return f"Slice '{name}' created at ({x},{y}) {width}x{height} in {sidecar_path(filename)}"


@mcp.tool()
async def set_slice_center(filename: str, name: str, x: int, y: int, width: int, height: int) -> str:
    """Set a slice's 9-patch center rectangle (relative to the slice origin).

    Args:
        filename: Sprite file the slice belongs to
        name: Slice name
        x: Center left edge, relative to the slice
        y: Center top edge, relative to the slice
        width: Center width
        height: Center height
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        slices = load_slices(filename)
        _get(slices, name)["center"] = {"x": x, "y": y, "width": width, "height": height}
        _save_slices(filename, slices)
    except (ToolError, OSError, ValueError) as e:
        return f"Failed to set slice center: {e}"
    return f"Slice '{name}' 9-patch center set to ({x},{y}) {width}x{height} in {sidecar_path(filename)}"


@mcp.tool()
async def set_slice_pivot(filename: str, name: str, x: int, y: int) -> str:
    """Set a slice's pivot point (relative to the slice origin), e.g. a character's feet.

    Args:
        filename: Sprite file the slice belongs to
        name: Slice name
        x: Pivot x, relative to the slice
        y: Pivot y, relative to the slice
    """
    try:
        slices = load_slices(filename)
        _get(slices, name)["pivot"] = {"x": x, "y": y}
        _save_slices(filename, slices)
    except (ToolError, OSError, ValueError) as e:
        return f"Failed to set slice pivot: {e}"
    return f"Slice '{name}' pivot set to ({x},{y}) in {sidecar_path(filename)}"


@mcp.tool()
async def list_slices(filename: str) -> str:
    """List all slices with their bounds, 9-patch centers, and pivots.

    Returns:
        JSON array of {name, x, y, width, height, center?, pivot?}.
    """
    if not os.path.isfile(filename):
        return f"File {filename} not found"
    try:
        return json.dumps(load_slices(filename))
    except (OSError, ValueError) as e:
        return f"Failed to list slices: {e}"


@mcp.tool()
async def delete_slice(filename: str, name: str) -> str:
    """Delete a slice by name.

    Args:
        filename: Sprite file the slice belongs to
        name: Slice to delete
    """
    try:
        slices = load_slices(filename)
        _get(slices, name)
        _save_slices(filename, [s for s in slices if s["name"] != name])
    except (ToolError, OSError, ValueError) as e:
        return f"Failed to delete slice: {e}"
    return f"Slice '{name}' deleted from {filename}"
