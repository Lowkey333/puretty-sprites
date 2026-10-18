import asyncio
import base64
import io
import json
import logging
import os
import socket
import sys

from mcp.server.mcpserver import Image
from PIL import Image as PILImage

from .. import mcp

# httpx logs every request at INFO, which floods the MCP server log.
logging.getLogger("httpx").setLevel(logging.WARNING)

# These tools talk to the Puretty Bridge plugin running inside Krita
# (integrations/krita). On start it writes its port and a per-session token
# to puretty_bridge.json in Krita's app-data folder (%APPDATA%\krita on
# Windows); KRITA_BRIDGE_FILE overrides that path. Paths are made absolute
# here because Krita resolves them against its own working directory.

NOT_RUNNING = ("Krita is not reachable. Start Krita; the Puretty Bridge plugin starts with it "
               "(install it with integrations/krita/install.py, then check it is enabled in "
               "Settings > Configure Krita > Python Plugin Manager).")


class KritaBridgeError(Exception):
    """The bridge could not be reached or refused the request."""


def discovery_file() -> str:
    override = os.getenv("KRITA_BRIDGE_FILE")
    if override:
        return override
    if os.name == "nt":
        base = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "krita")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support/krita")
    else:
        base = os.path.expanduser("~/.local/share/krita")
    return os.path.join(base, "puretty_bridge.json")


def _connection() -> tuple[int, str]:
    try:
        with open(discovery_file(), encoding="utf-8") as f:
            info = json.load(f)
        return int(info["port"]), str(info["token"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise KritaBridgeError(NOT_RUNNING) from e


def request_sync(command: str, params: dict | None = None, timeout: float = 120.0):
    """Send one command to the bridge and return its result."""
    port, token = _connection()
    message = {"id": 1, "token": token, "command": command, "params": params or {}}
    buf = bytearray()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
            sock.sendall(json.dumps(message).encode("utf-8") + b"\n")
            while b"\n" not in buf:
                chunk = sock.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
    except socket.timeout as e:
        raise KritaBridgeError(f"Krita did not answer within {timeout:g}s (is a dialog open?)") from e
    except OSError as e:
        raise KritaBridgeError(NOT_RUNNING) from e
    if not buf.strip():
        raise KritaBridgeError("Krita closed the connection without answering")
    response = json.loads(bytes(buf).split(b"\n", 1)[0])
    if not response.get("ok"):
        raise KritaBridgeError(response.get("error", "unknown error"))
    return response.get("result")


async def _request(command: str, params: dict | None = None, timeout: float = 120.0):
    return await asyncio.to_thread(request_sync, command, params, timeout)


def _abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path)) if path else path


@mcp.tool()
async def krita_status() -> str:
    """Check Krita is running with the Puretty Bridge plugin and list its open documents.

    Krita is the painting surface for the ComfyUI/Krita workflow: generate
    or edit images with the comfyui_* tools (or the AI Image Generation
    docker inside Krita), push them in with krita_add_image_layer, paint
    over them by hand, then export and pixelize the result for LibreSprite.
    """
    try:
        result = await _request("status")
    except KritaBridgeError as e:
        return f"Failed to reach Krita: {e}"
    return json.dumps(result, indent=2)


@mcp.tool()
async def krita_new_document(width: int, height: int, name: str = "Untitled", resolution: int = 72,
                             white_background: bool = False) -> str:
    """Create a new RGBA 8-bit document in Krita and show it.

    It starts with one empty, transparent layer ("Layer 1") unless
    white_background is set.

    Args:
        width: Width in pixels
        height: Height in pixels
        name: Document name shown in Krita's tab (default "Untitled")
        resolution: Resolution in DPI (default 72)
        white_background: Start with Krita's opaque white "Background" layer instead
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        info = await _request("new_document", {"width": width, "height": height, "name": name,
                                               "resolution": resolution,
                                               "background": "white" if white_background else "transparent"})
    except KritaBridgeError as e:
        return f"Failed to create document: {e}"
    return f"Created Krita document '{info['name']}' ({info['width']}x{info['height']})"


@mcp.tool()
async def krita_open(path: str) -> str:
    """Open an image or .kra file in Krita and make it the active document.

    Args:
        path: File to open (.kra, .png, .psd, .ora, ...)
    """
    path = _abs(path)
    if not os.path.isfile(path):
        return f"File {path} not found"
    try:
        info = await _request("open_document", {"path": path})
    except KritaBridgeError as e:
        return f"Failed to open in Krita: {e}"
    return f"Opened {info['path']} in Krita ({info['width']}x{info['height']}, {info['color_model']} {info['color_depth']})"


@mcp.tool()
async def krita_list_layers(document: str = "") -> str:
    """List the layers of a Krita document as JSON (bottom to top, with groups nested).

    Args:
        document: Path or name of an open document (default: the active one)
    """
    try:
        result = await _request("list_layers", {"document": document})
    except KritaBridgeError as e:
        return f"Failed to list layers: {e}"
    return json.dumps(result, indent=2)


@mcp.tool()
async def krita_add_image_layer(image_path: str, layer_name: str = "", x: int = 0, y: int = 0,
                                opacity: int = 255, document: str = "") -> str:
    """Add an image file (PNG, JPG, ...) to a Krita document as a new top paint layer.

    Use it to bring ComfyUI generations, Blender renders or LibreSprite
    exports into Krita for painting over or comparison.

    Args:
        image_path: Image to load
        layer_name: Name for the new layer (default: the file name)
        x: Left position of the image on the canvas
        y: Top position of the image on the canvas
        opacity: Layer opacity 0-255 (default 255)
        document: Path or name of an open document (default: the active one)
    """
    image_path = _abs(image_path)
    if not os.path.isfile(image_path):
        return f"File {image_path} not found"
    if not 0 <= opacity <= 255:
        return "Opacity must be between 0 and 255"
    try:
        info = await _request("add_image_layer", {"path": image_path, "name": layer_name, "x": x, "y": y,
                                                  "opacity": opacity, "document": document})
    except KritaBridgeError as e:
        return f"Failed to add image layer: {e}"
    return (f"Added layer '{info['layer']}' ({info['width']}x{info['height']} at {info['x']},{info['y']}) "
            f"to {info['document']}")


@mcp.tool()
async def krita_export(output_path: str, layer_name: str = "", scale: int = 1, document: str = "") -> str:
    """Export a Krita document (flattened) or one of its layers to an image file.

    PNG keeps transparency. Use scale for nearest-neighbour upscaling;
    to turn painted art into pixel art, export at full size and run the
    pixelize tools on the result.

    Args:
        output_path: Destination file (.png, .jpg, .webp, .bmp, .tif)
        layer_name: Export only this layer, at canvas size (default: all layers)
        scale: Integer upscale factor (default 1)
        document: Path or name of an open document (default: the active one)
    """
    if scale < 1:
        return "Scale must be >= 1"
    output_path = _abs(output_path)
    try:
        info = await _request("export_document", {"path": output_path, "layer": layer_name,
                                                  "scale": scale, "document": document})
    except KritaBridgeError as e:
        return f"Failed to export from Krita: {e}"
    return f"Exported {info['width']}x{info['height']} image to {info['path']}"


@mcp.tool()
async def krita_view_canvas(layer_name: str = "", max_size: int = 768, document: str = "") -> list[str | Image]:
    """Show the current Krita canvas (or one layer) as an image you can look at.

    Large canvases are scaled down to max_size; small ones (pixel art)
    are scaled up with nearest-neighbour so individual pixels stay sharp.

    Args:
        layer_name: Show only this layer (default: the flattened canvas)
        max_size: Longest side of the returned image in pixels (default 768)
        document: Path or name of an open document (default: the active one)
    """
    if max_size < 16:
        return ["max_size must be >= 16"]
    try:
        info = await _request("get_canvas", {"layer": layer_name, "max_size": max_size, "document": document})
    except KritaBridgeError as e:
        return [f"Failed to read the Krita canvas: {e}"]
    png = base64.b64decode(info["png_base64"])
    longest = max(info["width"], info["height"])
    factor = max_size // longest if longest * 2 <= max_size else 1
    if factor > 1:
        img = PILImage.open(io.BytesIO(png))
        img = img.resize((img.width * factor, img.height * factor), PILImage.NEAREST)
        out = io.BytesIO()
        img.save(out, "PNG")
        png = out.getvalue()
    summary = (f"{info['document']}: canvas {info['canvas_width']}x{info['canvas_height']}"
               + (f", layer '{layer_name}'" if layer_name else "")
               + (f", shown at {factor}x" if factor > 1 else ""))
    return [summary, Image(data=png, format="png")]


@mcp.tool()
async def krita_save_document(path: str = "", document: str = "") -> str:
    """Save a Krita document, to its own file or to a new path (.kra keeps layers).

    Args:
        path: Where to save (default: the document's current file)
        document: Path or name of an open document (default: the active one)
    """
    try:
        info = await _request("save_document", {"path": _abs(path), "document": document})
    except KritaBridgeError as e:
        return f"Failed to save Krita document: {e}"
    return f"Saved {info['path']}"


@mcp.tool()
async def krita_close_document(document: str = "") -> str:
    """Close a Krita document WITHOUT saving it.

    Args:
        document: Path or name of an open document (default: the active one)
    """
    try:
        info = await _request("close_document", {"document": document})
    except KritaBridgeError as e:
        return f"Failed to close Krita document: {e}"
    return f"Closed {info['closed']} (unsaved changes discarded)"


@mcp.tool()
async def krita_run_python(code: str) -> str:
    """Run Python inside Krita (escape hatch), like Krita's Scripter.

    The code runs on Krita's GUI thread with `Krita`, `InfoObject`, `app`
    (Krita.instance()) and `doc` (the active document) predefined. print()
    output is returned; assign to `result` to return a value. API docs:
    https://api.kde.org/krita/html/classKrita.html

    WARNING: this executes unrestricted code inside Krita. Only pass code
    you trust.

    Args:
        code: Python source to execute
    """
    if not code.strip():
        return "Code cannot be empty"
    try:
        result = await _request("run_python", {"code": code}, timeout=300)
    except KritaBridgeError as e:
        return f"Failed to run Python in Krita: {e}"
    parts = []
    if result.get("stdout"):
        parts.append(result["stdout"].rstrip())
    if result.get("result") is not None:
        parts.append(f"result = {result['result']}")
    return "\n".join(parts) if parts else "Code ran (no output)"
