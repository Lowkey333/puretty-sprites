"""Regenerate workflows/*.json from the graphs the comfyui_* tools build.

    uv run --project libresprite-mcp python integrations/comfyui/make_templates.py

The tools build these graphs in code (libresprite_mcp/tools/comfyui.py);
the JSON copies are for loading into ComfyUI or running with
comfyui_run_workflow. tests/test_comfyui.py checks they are in sync.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "libresprite-mcp"))

from libresprite_mcp.tools.comfyui import img2img_graph, pixel_art_prompts, txt2img_graph  # noqa: E402

EXAMPLE = "a knight with a red cape and a sword, full body, side view"


def templates() -> dict[str, dict]:
    positive, negative = pixel_art_prompts(EXAMPLE, "", True)
    return {
        "txt2img_sdxl_pixel_art_lora.json": txt2img_graph(
            "sd_xl_base_1.0.safetensors", positive, negative, 1024, 1024, 25, 6.5, "dpmpp_2m", "karras",
            42, 1, "pixel-art-xl.safetensors", 1.0, "puretty"),
        "txt2img_sd15.json": txt2img_graph(
            "v1-5-pruned-emaonly.safetensors", positive, negative, 512, 512, 25, 7.0, "dpmpp_2m", "karras",
            42, 1, "", 1.0, "puretty"),
        "img2img_sdxl.json": img2img_graph(
            "sd_xl_base_1.0.safetensors", "example.png", positive, negative, 25, 6.5, "dpmpp_2m", "karras",
            42, 0.55, "", 1.0, "puretty_i2i"),
    }


def main() -> None:
    folder = os.path.join(HERE, "workflows")
    os.makedirs(folder, exist_ok=True)
    for name, graph in templates().items():
        with open(os.path.join(folder, name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(graph, f, indent=2)
            f.write("\n")
        print("wrote", os.path.join(folder, name))


if __name__ == "__main__":
    main()
