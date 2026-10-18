# ComfyUI tools

The `comfyui_*` tools in libresprite-mcp let the agent generate and restyle images on a local ComfyUI server, then hand the results to the pixelize tools and LibreSprite. For how this fits with Krita, see `../krita/README.md`.

## Finding the server

The tools try `COMFYUI_URL` if set, otherwise `http://127.0.0.1:8188` (ComfyUI's default, also used by the server Krita AI Diffusion manages) and then `http://127.0.0.1:8000` (ComfyUI Desktop). Results are downloaded to `output_dir`, or `COMFYUI_OUTPUT_DIR`, or `./comfyui_output`, and are never overwritten.

On an RTX 50-series card, ComfyUI needs a PyTorch build for CUDA 12.8 or newer. Krita AI Diffusion's installer (CUDA 13.0 builds), current ComfyUI Desktop and the current portable release all qualify.

| Tool | Does |
|---|---|
| `comfyui_status` | Version, GPU and free VRAM, queue length |
| `comfyui_list_models` | Checkpoints, LoRAs, VAEs and upscale models ComfyUI can load |
| `comfyui_generate` | Text-to-image for SD 1.5 / SDXL checkpoints, with an optional LoRA. `pixel_art=True` (the default) adds pixel-art keywords and a negative prompt against blur and smooth shading. The size defaults to 1024 for SDXL and 512 for SD 1.5. A random seed is reported so a good image can be reproduced. |
| `comfyui_img2img` | Re-draw an image (a sketch, a Krita painting, a Blender render) at a chosen `denoise` |
| `comfyui_run_workflow` | Run any workflow: an **API-format** file, a regular **UI-format** save (converted with the server's node definitions) or one of ComfyUI's bundled templates as `"template:<name>"`. Change inputs with `{"node_id.input": value}`, upload local images into LoadImage inputs, optionally randomize seeds. Downloads every file it produces (images, animations, 3D meshes). |
| `comfyui_list_templates` | Search the templates bundled with ComfyUI (e.g. `"3d"`), to run with `comfyui_run_workflow("template:<name>")` |
| `comfyui_remove_background` | Cut the subject out with the BiRefNet matting model. Much better than the pixelize tools' own background removal on generated art: it also removes floor shadows and background enclosed by the character |

All three generating tools return the saved file paths and, by default, a preview image the agent can see.

## Installed on this machine

ComfyUI 0.41 is installed in `C:\AI\ComfyUI` (Python 3.13 venv, PyTorch 2.14.1 with CUDA 13.0, running on the RTX 5060 Ti). Start it with `C:\AI\ComfyUI\run_comfyui.bat`; it serves `http://127.0.0.1:8188`, where Krita AI Diffusion (set to "external server") and the `comfyui_*` tools connect. About 32 GB in all.

| What | Files |
|---|---|
| Krita AI Diffusion's custom nodes, at the commits it pins | `custom_nodes\comfyui_controlnet_aux`, `ComfyUI_IPAdapter_plus`, `comfyui-tooling-nodes`, `comfyui-inpaint-nodes` |
| Krita AI Diffusion's SDXL models (its own `download_models.py --sdxl`) | CLIP vision, IP-Adapter SDXL, Fooocus inpaint, MAT inpaint, Hyper-SDXL LoRA, NMKD/OmniSR upscalers |
| Checkpoint | `checkpoints\zavychromaxl_v80.safetensors` (ZavyChroma XL, the "Digital Artwork" style in Krita) |
| Pixel art | `loras\pixel-art-xl.safetensors`, `vae\sdxl_vae_fp16_fix.safetensors` |
| Image to 3D (ComfyUI's own TRELLIS.2 / Pixal3D template) | `diffusion_models\trellis_2_int8_convrot.safetensors`, `pixal3d_int8_convrot.safetensors`, `vae\trellis_2_*_vae_bf16.safetensors`, `clip_vision\dino_v3_L_naf_fp32.safetensors`, `geometry_estimation\moge_2_vitl_normal_fp16.safetensors`, `background_removal\birefnet.safetensors` |

To remove it all, delete `C:\AI\ComfyUI`.

## What it did here (RTX 5060 Ti 16 GB)

| Step | Time | Result |
|---|---|---|
| 4 SDXL images at 1024 px, 25 steps, Pixel Art XL LoRA | 68 s (first run, model loading included) | Convincing pixel-art-style characters, but off-grid "pixels" with hundreds of colors and floor shadows |
| BiRefNet cutout | ~5 s per image | Clean alpha, shadows and enclosed background gone |
| Pixelize to 128 px / 64 px with a 16-color palette | <1 s | 128 px (the LoRA's own 8× scale) is good pixel art; at 64 px faces and outlines fall apart and need redrawing |
| Krita AI Diffusion refine (strength 0.6) | ~20 s | Swapped the broom for a glowing staff, keeping pose, outfit and style |
| TRELLIS.2 image to 3D from one concept image | 6.5 min | 700k-triangle textured model, coherent from the front, sides and back |

The workflow examples are in `examples\workflow` (01_generate ... 07_sprite).

## Models for pixel art (16 GB card)

None of these are downloaded. Put checkpoints in ComfyUI's `models/checkpoints` and LoRAs in `models/loras`. Krita AI Diffusion's managed server keeps its models in `%APPDATA%\krita\ai_diffusion\server\models\` (`checkpoints`, `loras`, ...), registered with ComfyUI through `extra_model_paths.yaml`, so put files there.

- **SDXL + Pixel Art XL (recommended start).** Checkpoint: *ZavyChroma XL* (offered by the Krita AI Diffusion installer as "ZavyChroma (SDXL – Artwork)") or [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0). LoRA: [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl) (no trigger word needed; its author suggests downscaling 8× with nearest-neighbour, which is what the pixelize step does, and a fixed SDXL VAE such as the 0.9 VAE or [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) to avoid artifacts). Example: `comfyui_generate(prompt="a knight with a red cape, full body, side view", lora="pixel-art-xl.safetensors")`.
- **Alternative SDXL LoRA:** [artificialguybr/PixelArtRedmond](https://huggingface.co/artificialguybr/PixelArtRedmond) (trigger words "Pixel Art, PixArFK").
- **SD 1.5 (fast, 512 px):** [PublicPrompts/All-In-One-Pixel-Model](https://huggingface.co/PublicPrompts/All-In-One-Pixel-Model), a checkpoint with trigger words "pixelsprite" (sprites) and "16bitscene" (scenes).
- **Flux** (Flux Krea / Flux Kontext from the Krita installer, fp8, fits in 16 GB) needs a Flux graph, not the SD template. Export one from ComfyUI in API format and use `comfyui_run_workflow`. Flux Kontext is good at "same character, different pose or outfit" edits, which helps with consistency.
- **Image to 3D (for the Blender route):** ComfyUI now runs TRELLIS.2 natively (installed here, see above): `comfyui_run_workflow("template:3d_pixal3d_trellis2_image_to_model", overrides={"316.value": True}, images={"122.image": "concept.png"})` turns one concept image into a textured `.glb` (`"316.value": False` uses Pixal3D instead). Native Hunyuan3D 2.0/2.1 templates exist too, but they make untextured shapes.

Tips:

- Ask for **one subject on a plain background** ("full body, side view, plain white background"). Clean silhouettes pixelize well.
- Generate at the model's native size (1024 for SDXL) and let the pixelize tools shrink to the sprite size (for example 64 px tall) with your palette. Don't ask the model for 64×64 directly.
- To restyle a Blender render or a rough Krita sketch, use `comfyui_img2img` with `denoise` 0.35 to 0.55 so the pose survives.
- Keep the seed of a good result and change one prompt detail at a time for variants.

## Workflow templates

`workflows/` contains API-format graphs you can load into ComfyUI or run with `comfyui_run_workflow`:

| File | Graph |
|---|---|
| `txt2img_sdxl_pixel_art_lora.json` | SDXL text-to-image with a LoRA (node 12), 1024×1024 |
| `txt2img_sd15.json` | SD 1.5 text-to-image, 512×512 |
| `img2img_sdxl.json` | Image-to-image: LoadImage (node 10) → VAEEncode → KSampler with denoise 0.55 |

Node ids match ComfyUI's default graph: 3 KSampler, 4 checkpoint, 5 latent size, 6 positive prompt, 7 negative prompt, 9 SaveImage. So `{"6.text": "...", "3.seed": 1}` works on all of them. The checkpoint and LoRA names are placeholders (`sd_xl_base_1.0.safetensors`, `pixel-art-xl.safetensors`); override `4.ckpt_name` / `12.lora_name` to match yours. The files are generated from the graphs the tools build; regenerate them with `uv run --project libresprite-mcp python integrations/comfyui/make_templates.py`.

## Tests

`libresprite-mcp/tests/test_comfyui.py` runs the tools against an in-process fake of ComfyUI's HTTP API (graphs, uploads, polling, downloads, error reporting, templates in sync). `tests/test_comfy_graph.py` covers the UI-to-API conversion, and converts the real TRELLIS.2 template when ComfyUI is running. Every tool was also run against the real ComfyUI above.
