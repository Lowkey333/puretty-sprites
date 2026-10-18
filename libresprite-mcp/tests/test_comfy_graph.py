import json
import urllib.request

import pytest

from libresprite_mcp.core import comfy_graph
from libresprite_mcp.tools import comfyui

INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"], {}]}},
                               "output": ["MODEL", "CLIP", "VAE"]},
    "KSampler": {"input": {"required": {
        "model": ["MODEL", {}], "seed": ["INT", {"control_after_generate": True, "default": 0}],
        "steps": ["INT", {"default": 20}], "cfg": ["FLOAT", {"default": 8.0}],
        "latent_image": ["LATENT", {}]}}, "output": ["LATENT"]},
    "LoadImage": {"input": {"required": {"image": [["x.png"], {"image_upload": True}]}}, "output": ["IMAGE", "MASK"]},
    "EmptyLatent": {"input": {"required": {"width": ["INT", {"default": 512}], "height": ["INT", {"default": 512}],
                                           "batch": ["INT", {"default": 1}]}}, "output": ["LATENT"]},
    "Remesh": {"input": {"required": {
        "latent": ["LATENT", {}],
        "mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "udf", "inputs": {"required": {"qef": ["BOOLEAN", {"default": False}]}}},
            {"key": "sdf", "inputs": {"required": {"manifold": ["BOOLEAN", {"default": False}]}}}]}],
        "band": ["FLOAT", {"default": 1.0}], "color": ["COLOR", {"default": "#000000"}]}},
        "output": ["LATENT"]},
    "MakeColor": {"input": {"required": {}}, "output": ["COLOR"]},
    "Blur": {"input": {"required": {"latent": ["LATENT", {}]}}, "output": ["LATENT"]},
}


def _ui():
    return {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "mode": 0, "widgets_values": ["b.safetensors"]},
            {"id": 2, "type": "EmptyLatent", "mode": 0, "widgets_values": [768, 1024]},  # saved before "batch"
            {"id": 3, "type": "KSampler", "mode": 0, "widgets_values": [42, "randomize", 30, 6.5],
             "inputs": [{"name": "model", "link": 10}, {"name": "latent_image", "link": 13}]},
            {"id": 4, "type": "Reroute", "mode": 0, "inputs": [{"name": "", "link": 11}]},
            {"id": 5, "type": "LoadImage", "mode": 0, "widgets_values": ["photo.png", "image"]},
            {"id": 6, "type": "Note", "mode": 0, "widgets_values": ["ignore me"]},
            {"id": 7, "type": "Blur", "mode": 4, "inputs": [{"name": "latent", "link": 12, "type": "LATENT"}]},
            {"id": 8, "type": "Remesh", "mode": 0, "widgets_values": ["udf", True, 0.5, "#ff00ff"],
             "inputs": [{"name": "latent", "link": 14}]},
            {"id": 9, "type": "PrimitiveNode", "mode": 0, "widgets_values": [7]},
            {"id": 10, "type": "KSampler", "mode": 0, "widgets_values": [1, "fixed", 5, 1.0],
             "inputs": [{"name": "model", "link": 16}, {"name": "latent_image", "link": 17},
                        {"name": "steps", "link": 15, "widget": {"name": "steps"}}]},
        ],
        "links": [
            [10, 1, 0, 3, 0, "MODEL"],
            [11, 2, 0, 4, 0, "LATENT"],
            [12, 4, 0, 7, 0, "LATENT"],
            [13, 7, 0, 3, 4, "LATENT"],   # through a bypassed Blur and a Reroute back to node 2
            [14, 3, 0, 8, 0, "LATENT"],
            [15, 9, 0, 10, 2, "INT"],     # editor-only primitive feeding a converted widget
            [16, 1, 0, 10, 0, "MODEL"],
            [17, 8, 0, 10, 4, "LATENT"],
        ],
    }


def test_ui_to_api_rules():
    api, warnings = comfy_graph.ui_to_api(_ui(), INFO)
    assert set(api) == {"1", "2", "3", "5", "8", "10"}  # note, reroute, bypassed and primitive dropped
    assert api["1"]["inputs"] == {"ckpt_name": "b.safetensors"}
    assert api["2"]["inputs"] == {"width": 768, "height": 1024, "batch": 1}  # newer input -> default
    # seed's "randomize" companion value is skipped; links survive reroute + bypass
    assert api["3"]["inputs"] == {"model": ["1", 0], "seed": 42, "steps": 30, "cfg": 6.5, "latent_image": ["2", 0]}
    assert api["5"]["inputs"] == {"image": "photo.png"}  # upload widget value skipped
    # dynamic combo: chosen option's inputs follow it, sent as "mode.qef"; COLOR is a widget
    assert api["8"]["inputs"] == {"latent": ["3", 0], "mode": "udf", "mode.qef": True, "band": 0.5,
                                  "color": "#ff00ff"}
    assert api["10"]["inputs"]["steps"] == 7  # primitive folded in
    assert any("batch" in w for w in warnings)


def test_ui_to_api_unknown_node():
    ui = {"nodes": [{"id": 1, "type": "MysteryNode", "mode": 0}], "links": []}
    with pytest.raises(comfy_graph.ConversionError, match="MysteryNode"):
        comfy_graph.ui_to_api(ui, INFO)


def test_remove_background_graph():
    g = comfyui.remove_background_graph("in.png", "birefnet.safetensors", "cut")
    assert g["3"]["inputs"]["bg_removal_model"] == ["2", 0]
    assert g["4"]["class_type"] == "InvertMask" and g["5"]["inputs"]["alpha"] == ["4", 0]


def _comfyui_up() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _comfyui_up(), reason="ComfyUI is not running on 127.0.0.1:8188")
def test_bundled_3d_template_converts():
    ui = json.load(urllib.request.urlopen("http://127.0.0.1:8188/templates/3d_pixal3d_trellis2_image_to_model.json"))
    info = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info"))
    api, _warnings = comfy_graph.ui_to_api(ui, info)
    remesh = next(n for n in api.values() if n["class_type"] == "RemeshMesh")
    assert remesh["inputs"]["sign_mode"] == "udf" and "sign_mode.qef" in remesh["inputs"]
    assert remesh["inputs"]["band"] >= 0.5
    assert all(n["class_type"] not in ("Note", "MarkdownNote") for n in api.values())
