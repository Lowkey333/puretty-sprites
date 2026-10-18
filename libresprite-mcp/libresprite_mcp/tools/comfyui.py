import asyncio
import io
import json
import logging
import os
import random
import re
import time
import uuid
from typing import Any, Dict

import httpx
from mcp.server.mcpserver import Image
from PIL import Image as PILImage

from .. import mcp
from ..core import comfy_graph

# httpx logs every request at INFO, which floods the MCP server log.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Client for a ComfyUI server's HTTP API. COMFYUI_URL picks the server;
# otherwise 127.0.0.1:8188 (ComfyUI's default, also used by the server Krita
# AI Diffusion installs and manages) and 127.0.0.1:8000 (ComfyUI Desktop)
# are tried. Results are saved under output_dir (COMFYUI_OUTPUT_DIR, default
# ./comfyui_output) and are meant to go through the pixelize tools next:
# diffusion models paint "pixel art" off-grid and with too many colors.

DEFAULT_URLS = ("http://127.0.0.1:8188", "http://127.0.0.1:8000")

NOT_RUNNING = ("ComfyUI is not reachable at http://127.0.0.1:8188 or :8000. Start ComfyUI, or open "
               "Krita's AI Image Generation docker and start its local server (Krita AI Diffusion runs "
               "ComfyUI on 127.0.0.1:8188). Set COMFYUI_URL to use another address.")

PIXEL_ART_POSITIVE = ("pixel art, {prompt}, clean pixel art sprite, crisp hard-edged pixels, "
                      "limited color palette, flat colors, plain background")
PIXEL_ART_NEGATIVE = ("blurry, soft focus, smooth shading, gradients, anti-aliasing, photo, "
                      "photorealistic, 3d render, noise, jpeg artifacts, text, watermark, signature")


class ComfyError(Exception):
    """ComfyUI could not be reached or rejected/failed the workflow."""


# ------------------------------------------------------------ graphs

def txt2img_graph(checkpoint: str, positive: str, negative: str, width: int, height: int,
                  steps: int, cfg: float, sampler: str, scheduler: str, seed: int,
                  batch_size: int = 1, lora: str = "", lora_strength: float = 1.0,
                  filename_prefix: str = "puretty") -> dict:
    """ComfyUI's default text-to-image graph (API format), plus an optional LoRA."""
    graph = {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
            "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height,
                                                           "batch_size": batch_size}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]}},
    }
    if lora:
        _add_lora(graph, lora, lora_strength)
    return graph


def img2img_graph(checkpoint: str, image_name: str, positive: str, negative: str, steps: int,
                  cfg: float, sampler: str, scheduler: str, seed: int, denoise: float,
                  lora: str = "", lora_strength: float = 1.0, filename_prefix: str = "puretty") -> dict:
    """Image-to-image: encode an uploaded image and re-sample it with partial denoise."""
    graph = txt2img_graph(checkpoint, positive, negative, 512, 512, steps, cfg, sampler, scheduler, seed,
                          filename_prefix=filename_prefix)
    del graph["5"]
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    graph["11"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}}
    graph["3"]["inputs"]["latent_image"] = ["11", 0]
    graph["3"]["inputs"]["denoise"] = denoise
    if lora:
        _add_lora(graph, lora, lora_strength)
    return graph


def remove_background_graph(image_name: str, model: str, filename_prefix: str) -> dict:
    # RemoveBackground's mask is 1 on the subject and JoinImageWithAlpha
    # uses 1 - mask as alpha, so the mask is inverted in between.
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": model}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": filename_prefix}},
    }


def _add_lora(graph: dict, lora: str, strength: float) -> None:
    graph["12"] = {"class_type": "LoraLoader", "inputs": {
        "lora_name": lora, "strength_model": strength, "strength_clip": strength,
        "model": ["4", 0], "clip": ["4", 1]}}
    graph["3"]["inputs"]["model"] = ["12", 0]
    graph["6"]["inputs"]["clip"] = ["12", 1]
    graph["7"]["inputs"]["clip"] = ["12", 1]


def pixel_art_prompts(prompt: str, negative: str, pixel_art: bool) -> tuple[str, str]:
    if not pixel_art:
        return prompt, negative
    positive = PIXEL_ART_POSITIVE.format(prompt=prompt.strip())
    return positive, ", ".join(p for p in (negative.strip(), PIXEL_ART_NEGATIVE) if p)


def is_sdxl(checkpoint: str) -> bool:
    return bool(re.search(r"xl|pony|illustrious|noob", checkpoint, re.I))


# ------------------------------------------------------------ HTTP

async def _base_url(client: httpx.AsyncClient) -> str:
    configured = os.getenv("COMFYUI_URL", "").strip().rstrip("/")
    for url in ([configured] if configured else DEFAULT_URLS):
        try:
            r = await client.get(url + "/system_stats", timeout=3)
            if r.status_code == 200:
                return url
        except httpx.HTTPError:
            continue
    if configured:
        raise ComfyError(f"ComfyUI is not reachable at {configured} (COMFYUI_URL)")
    raise ComfyError(NOT_RUNNING)


def _combo_options(spec) -> list[str]:
    """Options of a combo input in /object_info, old or new format."""
    if not spec:
        return []
    if isinstance(spec[0], list):
        return [str(o) for o in spec[0]]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return [str(o) for o in spec[1].get("options", [])]
    return []


async def _node_options(client, base: str, node: str, field: str) -> list[str]:
    r = await client.get(f"{base}/object_info/{node}")
    if r.status_code != 200:
        return []
    info = r.json().get(node, {}).get("input", {})
    spec = info.get("required", {}).get(field) or info.get("optional", {}).get(field)
    return _combo_options(spec)


async def _pick_checkpoint(client, base: str, wanted: str, lora: str) -> str:
    names = await _node_options(client, base, "CheckpointLoaderSimple", "ckpt_name")
    if wanted:
        if names and wanted not in names:
            match = [n for n in names if wanted.lower() in n.lower()]
            if len(match) != 1:
                raise ComfyError(f"Checkpoint '{wanted}' not found. Available: {', '.join(names) or 'none'}")
            return match[0]
        return wanted
    if not names:
        raise ComfyError("ComfyUI has no checkpoints in models/checkpoints. Install one (Krita AI Diffusion's "
                         "installer can download them) or pass checkpoint=...")
    if lora and is_sdxl(lora):
        names = sorted(names, key=lambda n: not is_sdxl(n))
    return names[0]


def _prompt_error(r: httpx.Response) -> str:
    try:
        data = r.json()
    except ValueError:
        return f"HTTP {r.status_code}: {r.text[:500]}"
    parts = []
    err = data.get("error")
    if isinstance(err, dict):
        parts.append(err.get("message", "") + (f" ({err['details']})" if err.get("details") else ""))
    elif err:
        parts.append(str(err))
    for node_id, node_err in (data.get("node_errors") or {}).items():
        for e in node_err.get("errors", []):
            parts.append(f"node {node_id} ({node_err.get('class_type', '?')}): {e.get('message', '')} "
                         f"{e.get('details', '')}".strip())
    return "; ".join(p for p in parts if p) or f"HTTP {r.status_code}"


async def _run_graph(client, base: str, graph: dict, timeout: float) -> tuple[str, dict]:
    r = await client.post(f"{base}/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
    if r.status_code != 200:
        raise ComfyError("ComfyUI rejected the workflow: " + _prompt_error(r))
    data = r.json()
    if data.get("node_errors"):
        raise ComfyError("ComfyUI rejected the workflow: " + _prompt_error(r))
    prompt_id = data["prompt_id"]
    deadline = time.monotonic() + timeout
    delay = 0.25
    while time.monotonic() < deadline:
        h = await client.get(f"{base}/history/{prompt_id}")
        entry = h.json().get(prompt_id) if h.status_code == 200 else None
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                messages = [m[1] for m in status.get("messages", []) if m and m[0] == "execution_error"]
                detail = "; ".join(f"{m.get('node_type', '?')}: {m.get('exception_message', '').strip()}"
                                   for m in messages) or "execution error"
                raise ComfyError(f"Workflow failed in ComfyUI: {detail}")
            if status.get("completed", True):
                return prompt_id, entry
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 2.0)
    raise ComfyError(f"Timed out after {timeout:g}s; prompt {prompt_id} may still be running in ComfyUI")


def _output_dir(output_dir: str) -> str:
    folder = os.path.abspath(output_dir or os.getenv("COMFYUI_OUTPUT_DIR") or "comfyui_output")
    os.makedirs(folder, exist_ok=True)
    return folder


def _unique(folder: str, name: str) -> str:
    base, ext = os.path.splitext(os.path.basename(name) or "output.png")
    path = os.path.join(folder, base + ext)
    n = 1
    while os.path.exists(path):
        path = os.path.join(folder, f"{base}_{n}{ext}")
        n += 1
    return path


async def _download_outputs(client, base: str, entry: dict, folder: str) -> list[str]:
    """Download every file-like output (images, gifs, meshes) a run produced."""
    saved = []
    for node_output in (entry.get("outputs") or {}).values():
        for items in node_output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                r = await client.get(f"{base}/view", params={
                    "filename": item["filename"], "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output")})
                if r.status_code != 200:
                    raise ComfyError(f"Could not download {item['filename']}: HTTP {r.status_code}")
                path = _unique(folder, item["filename"])
                with open(path, "wb") as f:
                    f.write(r.content)
                saved.append(path)
    return saved


async def _upload(client, base: str, path: str) -> str:
    with open(path, "rb") as f:
        r = await client.post(f"{base}/upload/image", data={"overwrite": "true", "type": "input"},
                              files={"image": (os.path.basename(path), f, "image/png")})
    if r.status_code != 200:
        raise ComfyError(f"Upload of {path} failed: HTTP {r.status_code} {r.text[:200]}")
    info = r.json()
    return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]


def _preview(paths: list[str], max_size: int = 512) -> Image | None:
    for path in paths:
        try:
            img = PILImage.open(path)
            img.load()
        except OSError:
            continue
        img.thumbnail((max_size, max_size))
        out = io.BytesIO()
        img.save(out, "PNG")
        return Image(data=out.getvalue(), format="png")
    return None


def _result(lines: list[str], paths: list[str], preview: bool) -> list[str | Image]:
    text = "\n".join(lines + ["Saved:"] + [f"  {p}" for p in paths])
    image = _preview(paths) if preview else None
    return [text, image] if image else [text]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))


# ------------------------------------------------------------ tools

@mcp.tool()
async def comfyui_status() -> str:
    """Check the ComfyUI server: version, GPU/VRAM and queue length.

    ComfyUI runs the image models. Start it yourself, or let Krita AI
    Diffusion install and run a local one (127.0.0.1:8188); COMFYUI_URL
    points these tools elsewhere.
    """
    async with _client() as client:
        try:
            base = await _base_url(client)
            stats = (await client.get(f"{base}/system_stats")).json()
            queue = (await client.get(f"{base}/queue")).json()
        except (ComfyError, httpx.HTTPError, ValueError) as e:
            return f"Failed to reach ComfyUI: {e}"
    system = stats.get("system", {})
    lines = [f"ComfyUI at {base}",
             f"version: {system.get('comfyui_version', '?')}, python {str(system.get('python_version', '?')).split()[0]}"]
    for d in stats.get("devices", []):
        total, free = d.get("vram_total", 0), d.get("vram_free", 0)
        lines.append(f"device: {d.get('name', '?')} ({d.get('type', '?')}), "
                     f"VRAM {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB")
    lines.append(f"queue: {len(queue.get('queue_running', []))} running, "
                 f"{len(queue.get('queue_pending', []))} pending")
    return "\n".join(lines)


@mcp.tool()
async def comfyui_list_models() -> str:
    """List the checkpoints, LoRAs, VAEs and upscale models ComfyUI can load, as JSON.

    Pixel-art LoRAs (e.g. Pixel Art XL for SDXL checkpoints) go in
    ComfyUI's models/loras folder; pass their file name as lora=... to
    comfyui_generate.
    """
    async with _client() as client:
        try:
            base = await _base_url(client)
            result = {
                "checkpoints": await _node_options(client, base, "CheckpointLoaderSimple", "ckpt_name"),
                "loras": await _node_options(client, base, "LoraLoader", "lora_name"),
                "vae": await _node_options(client, base, "VAELoader", "vae_name"),
                "upscale_models": await _node_options(client, base, "UpscaleModelLoader", "model_name"),
            }
        except (ComfyError, httpx.HTTPError, ValueError) as e:
            return f"Failed to list ComfyUI models: {e}"
    return json.dumps(result, indent=2)


@mcp.tool()
async def comfyui_generate(
    prompt: str,
    negative_prompt: str = "",
    checkpoint: str = "",
    lora: str = "",
    lora_strength: float = 1.0,
    pixel_art: bool = True,
    width: int = 0,
    height: int = 0,
    steps: int = 25,
    cfg: float = 6.5,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
    seed: int = -1,
    batch_size: int = 1,
    output_dir: str = "",
    filename_prefix: str = "puretty",
    preview: bool = True,
    timeout: int = 600,
) -> list[str | Image]:
    """Generate images from a text prompt with ComfyUI (SD 1.5 / SDXL checkpoints).

    pixel_art wraps the prompt with pixel-art keywords and adds a negative
    prompt against blur and smooth shading. The output is still painted
    at full resolution: run pixelize_image (or import it pixelized into a
    sprite) to snap it to a real pixel grid and palette. For a character,
    ask for one pose on a plain background; for several directions or
    animation frames, the Blender route keeps them consistent.

    Args:
        prompt: What to draw, e.g. "a knight with a red cape, side view, full body"
        negative_prompt: Things to avoid (added to the pixel-art negative prompt)
        checkpoint: Checkpoint file name (default: first SDXL-looking one when the
            LoRA is SDXL, otherwise the first available; see comfyui_list_models)
        lora: Optional LoRA file name, e.g. a pixel-art LoRA such as "pixel-art-xl.safetensors"
        lora_strength: LoRA strength (default 1.0)
        pixel_art: Add pixel-art prompt keywords (default True)
        width: Image width (default 1024 for SDXL checkpoints, 512 otherwise)
        height: Image height (default 1024 for SDXL checkpoints, 512 otherwise)
        steps: Sampling steps (default 25)
        cfg: Classifier-free guidance scale (default 6.5)
        sampler: ComfyUI sampler name (default "dpmpp_2m")
        scheduler: ComfyUI scheduler name (default "karras")
        seed: Seed; -1 picks a random one (reported in the result)
        batch_size: Images to generate at once (default 1)
        output_dir: Folder for the results (default COMFYUI_OUTPUT_DIR or ./comfyui_output)
        filename_prefix: Prefix for the files ComfyUI writes (default "puretty")
        preview: Also return the first image so you can look at it (default True)
        timeout: Seconds to wait for the result (default 600)
    """
    if not prompt.strip():
        return ["Prompt cannot be empty"]
    if batch_size < 1 or steps < 1:
        return ["batch_size and steps must be >= 1"]
    if width < 0 or height < 0 or width % 8 or height % 8:
        return ["width and height must be multiples of 8"]
    seed = random.randint(0, 2**32 - 1) if seed < 0 else seed
    positive, negative = pixel_art_prompts(prompt, negative_prompt, pixel_art)
    async with _client() as client:
        try:
            base = await _base_url(client)
            ckpt = await _pick_checkpoint(client, base, checkpoint, lora)
            default = 1024 if is_sdxl(ckpt) else 512
            w, h = width or default, height or default
            graph = txt2img_graph(ckpt, positive, negative, w, h, steps, cfg, sampler, scheduler, seed,
                                  batch_size, lora, lora_strength, filename_prefix)
            prompt_id, entry = await _run_graph(client, base, graph, timeout)
            paths = await _download_outputs(client, base, entry, _output_dir(output_dir))
        except (ComfyError, httpx.HTTPError, ValueError, OSError) as e:
            return [f"Failed to generate: {e}"]
    if not paths:
        return [f"Failed to generate: ComfyUI finished prompt {prompt_id} without saving images"]
    lines = [f"Generated {len(paths)} image(s) with {ckpt}" + (f" + {lora} ({lora_strength:g})" if lora else ""),
             f"seed {seed}, {w}x{h}, {steps} steps, cfg {cfg:g}, {sampler}/{scheduler}"]
    return _result(lines, paths, preview)


@mcp.tool()
async def comfyui_img2img(
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    denoise: float = 0.55,
    checkpoint: str = "",
    lora: str = "",
    lora_strength: float = 1.0,
    pixel_art: bool = True,
    steps: int = 25,
    cfg: float = 6.5,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
    seed: int = -1,
    output_dir: str = "",
    filename_prefix: str = "puretty_i2i",
    preview: bool = True,
    timeout: int = 600,
) -> list[str | Image]:
    """Re-draw an existing image with ComfyUI (image-to-image).

    Good for turning a rough sketch, a Krita painting or a Blender render
    into a styled image while keeping its composition. The output keeps
    the input's size (SD models prefer sides that are multiples of 64).

    Args:
        image_path: Input image (PNG/JPG); it is uploaded to ComfyUI's input folder
        prompt: Description of the desired result
        negative_prompt: Things to avoid
        denoise: How much to change, 0-1 (0.3 = touch-up, 0.55 = restyle, 0.8 = loose reinterpretation)
        checkpoint: Checkpoint file name (default: auto, see comfyui_generate)
        lora: Optional LoRA file name
        lora_strength: LoRA strength (default 1.0)
        pixel_art: Add pixel-art prompt keywords (default True)
        steps: Sampling steps (default 25)
        cfg: Guidance scale (default 6.5)
        sampler: Sampler name (default "dpmpp_2m")
        scheduler: Scheduler name (default "karras")
        seed: Seed; -1 picks a random one
        output_dir: Folder for the results (default COMFYUI_OUTPUT_DIR or ./comfyui_output)
        filename_prefix: Prefix for the files ComfyUI writes
        preview: Also return the first image (default True)
        timeout: Seconds to wait (default 600)
    """
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        return [f"File {image_path} not found"]
    if not 0 < denoise <= 1:
        return ["denoise must be between 0 (exclusive) and 1"]
    seed = random.randint(0, 2**32 - 1) if seed < 0 else seed
    positive, negative = pixel_art_prompts(prompt, negative_prompt, pixel_art)
    async with _client() as client:
        try:
            base = await _base_url(client)
            ckpt = await _pick_checkpoint(client, base, checkpoint, lora)
            name = await _upload(client, base, image_path)
            graph = img2img_graph(ckpt, name, positive, negative, steps, cfg, sampler, scheduler, seed,
                                  denoise, lora, lora_strength, filename_prefix)
            prompt_id, entry = await _run_graph(client, base, graph, timeout)
            paths = await _download_outputs(client, base, entry, _output_dir(output_dir))
        except (ComfyError, httpx.HTTPError, ValueError, OSError) as e:
            return [f"Failed to run img2img: {e}"]
    if not paths:
        return [f"Failed to run img2img: ComfyUI finished prompt {prompt_id} without saving images"]
    lines = [f"img2img from {os.path.basename(image_path)} with {ckpt}" + (f" + {lora}" if lora else ""),
             f"seed {seed}, denoise {denoise:g}, {steps} steps, cfg {cfg:g}"]
    return _result(lines, paths, preview)


@mcp.tool()
async def comfyui_remove_background(
    image_path: str,
    output_dir: str = "",
    filename_prefix: str = "puretty_cutout",
    preview: bool = True,
    timeout: int = 300,
) -> list[str | Image]:
    """Cut the subject out of an image with an AI matting model (BiRefNet) on ComfyUI.

    Better than pixelize_image's built-in background removal for generated
    art: it also removes floor shadows and background enclosed by the
    subject (between legs, inside a broom). Pass the result to
    pixelize_image / frames_to_sprite with remove_background="none".
    Needs ComfyUI's native background-removal nodes and a model in
    models/background_removal (e.g. birefnet.safetensors).

    Args:
        image_path: Input image (PNG/JPG)
        output_dir: Folder for the RGBA result (default COMFYUI_OUTPUT_DIR or ./comfyui_output)
        filename_prefix: Prefix for the file ComfyUI writes
        preview: Also return the result as an image (default True)
        timeout: Seconds to wait (default 300)
    """
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        return [f"File {image_path} not found"]
    async with _client() as client:
        try:
            base = await _base_url(client)
            models = await _node_options(client, base, "LoadBackgroundRemovalModel", "bg_removal_name")
            if not models:
                raise ComfyError("No background-removal model installed (put birefnet.safetensors in "
                                 "ComfyUI/models/background_removal)")
            name = await _upload(client, base, image_path)
            graph = remove_background_graph(name, models[0], filename_prefix)
            prompt_id, entry = await _run_graph(client, base, graph, timeout)
            paths = await _download_outputs(client, base, entry, _output_dir(output_dir))
        except (ComfyError, httpx.HTTPError, ValueError, OSError) as e:
            return [f"Failed to remove background: {e}"]
    if not paths:
        return [f"Failed to remove background: ComfyUI finished prompt {prompt_id} without saving images"]
    return _result([f"Cut out {os.path.basename(image_path)} with {models[0]}"], paths, preview)


@mcp.tool()
async def comfyui_run_workflow(
    workflow_path: str,
    overrides: Dict[str, Any] | None = None,
    images: Dict[str, str] | None = None,
    randomize_seed: bool = False,
    output_dir: str = "",
    preview: bool = True,
    timeout: int = 900,
) -> list[str | Image]:
    """Run any ComfyUI workflow and download what it produces.

    Accepts API-format files ("Export (API)"), regular UI-format saves and
    ComfyUI's bundled templates by name ("template:3d_hunyuan3d_image_to_model",
    see comfyui_list_templates); UI-format graphs are converted with the
    server's node definitions. Every file-like output (images, animations,
    3D meshes) is downloaded. Our own templates are in integrations/comfyui/workflows.

    Args:
        workflow_path: Workflow JSON file (API or UI format), or "template:<name>"
        overrides: Input values to change, keyed "node_id.input_name",
            e.g. {"6.text": "pixel art, a slime monster", "3.steps": 30}
        images: Local images to upload into LoadImage-style inputs, keyed the
            same way, e.g. {"10.image": "C:/art/concept.png"}
        randomize_seed: Give every "seed"/"noise_seed" input a new random value
        output_dir: Folder for the results (default COMFYUI_OUTPUT_DIR or ./comfyui_output)
        preview: Also return the first image (default True)
        timeout: Seconds to wait (default 900)
    """
    template = workflow_path[len("template:"):] if workflow_path.startswith("template:") else ""
    if not template:
        workflow_path = os.path.abspath(workflow_path)
        if not os.path.isfile(workflow_path):
            return [f"File {workflow_path} not found"]
        try:
            with open(workflow_path, encoding="utf-8") as f:
                graph = json.load(f)
        except (OSError, ValueError) as e:
            return [f"Failed to read workflow: {e}"]
    notes: list[str] = []
    if not template:
        if isinstance(graph, dict) and isinstance(graph.get("prompt"), dict):
            graph = graph["prompt"]

    def target(key: str):
        node_id, _, field = key.partition(".")
        if node_id not in graph or not field:
            raise ComfyError(f"Unknown override target '{key}' (use \"node_id.input_name\"; "
                             f"nodes: {', '.join(sorted(graph, key=lambda k: (len(k), k)))})")
        return graph[node_id].setdefault("inputs", {}), field

    async with _client() as client:
        try:
            if template or comfy_graph.is_ui_format(graph):
                base = await _base_url(client)
                if template:
                    r = await client.get(f"{base}/templates/{template}.json")
                    if r.status_code != 200:
                        raise ComfyError(f"No template named {template!r} (see comfyui_list_templates)")
                    graph = r.json()
                info = (await client.get(f"{base}/object_info")).json()
                try:
                    graph, notes = comfy_graph.ui_to_api(graph, info)
                except comfy_graph.ConversionError as e:
                    raise ComfyError(f"Cannot convert the UI-format workflow: {e}")
            if not isinstance(graph, dict) or not all(isinstance(n, dict) and "class_type" in n
                                                      for n in graph.values()):
                raise ComfyError("not a workflow (expected UI format or API format "
                                 "{node_id: {class_type, inputs}})")
            for key, value in (overrides or {}).items():
                inputs, field = target(key)
                inputs[field] = value
            if randomize_seed:
                for node in graph.values():
                    for field in ("seed", "noise_seed"):
                        if isinstance(node.get("inputs", {}).get(field), int):
                            node["inputs"][field] = random.randint(0, 2**32 - 1)
            base = await _base_url(client)
            for key, path in (images or {}).items():
                inputs, field = target(key)
                path = os.path.abspath(path)
                if not os.path.isfile(path):
                    raise ComfyError(f"Image {path} not found")
                inputs[field] = await _upload(client, base, path)
            prompt_id, entry = await _run_graph(client, base, graph, timeout)
            paths = await _download_outputs(client, base, entry, _output_dir(output_dir))
        except (ComfyError, httpx.HTTPError, ValueError, OSError) as e:
            return [f"Failed to run workflow: {e}"]
    lines = [f"Ran {template or os.path.basename(workflow_path)} (prompt {prompt_id})"]
    lines += [f"note: {n}" for n in notes]
    if not paths:
        return ["\n".join(lines + ["The workflow finished but produced no downloadable files."])]
    return _result(lines, paths, preview)


@mcp.tool()
async def comfyui_list_templates(search: str = "", category: str = "") -> str:
    """List the workflow templates bundled with ComfyUI (run one with
    comfyui_run_workflow("template:<name>")).

    Args:
        search: Only templates whose name, title or description contains this
            text, e.g. "3d", "upscale", "inpaint"
        category: Only this category (as listed in the reply)
    """
    async with _client() as client:
        try:
            base = await _base_url(client)
            r = await client.get(f"{base}/templates/index.json")
            if r.status_code != 200:
                raise ComfyError("this ComfyUI serves no templates (install comfyui-workflow-templates)")
            index = r.json()
        except (ComfyError, httpx.HTTPError, ValueError) as e:
            return f"Failed to list templates: {e}"
    needle = search.lower()
    lines = []
    for cat in index if isinstance(index, list) else index.get("categories", []):
        title = cat.get("title") or cat.get("moduleName") or ""
        if category and category.lower() not in title.lower():
            continue
        for t in cat.get("templates", []):
            text = " ".join(str(t.get(k, "")) for k in ("name", "title", "description")).lower()
            if needle and needle not in text:
                continue
            if t.get("name", "").startswith("api_"):
                continue  # paid cloud API nodes
            models = ", ".join(t.get("models") or [])
            lines.append(f"{t.get('name')}  [{title}] {t.get('title', '')}" + (f" ({models})" if models else "")
                         + f"\n    {str(t.get('description', ''))[:160]}")
    return "\n".join(lines[:80]) if lines else "No matching templates"
