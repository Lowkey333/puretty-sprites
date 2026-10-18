# Krita + ComfyUI + an AI agent for sprites

**The question:** *"Is it possible to use ComfyUI and Krita and an AI agent to draw sprites?"*

**Yes, with caveats.** Each program does the part it is good at:

| Piece | Job |
|---|---|
| **ComfyUI** | Runs the image models on your GPU: text-to-image, image-to-image, inpainting. |
| **Krita + [Krita AI Diffusion](https://github.com/Acly/krita-ai-diffusion)** | Where *you* paint: sketch, inpaint, refine with AI in real time. It uses the same ComfyUI. |
| **The agent (Claude Code + libresprite-mcp)** | Drives both: `comfyui_*` tools queue generations, `krita_*` tools (through the Puretty Bridge plugin) put images into Krita and read your painting back. |
| **LibreSprite + the pixelize tools** | Turns the result into real pixel art (a true pixel grid and a small palette) in an `.ase` file you clean up and animate. |

What to expect:

- **Image models paint fake pixel art.** Their "pixels" are off-grid and uneven in size, with soft edges and hundreds of colors. Every generation has to be snapped to a grid and quantized to a palette (the pixelize tools do that), then cleaned up by hand. Small sprites (16 to 32 px) need the most hand work.
- **Frame-to-frame consistency is weak.** One good still is easy; eight matching walk-cycle frames are not. For several directions or animations, use the Blender route (model once, render every frame), or animate the cleaned-up still by hand in LibreSprite.
- **It is great for concepts, palettes, item icons, portraits and tiles,** and for getting from a rough sketch to a styled base quickly.

A typical loop, asked of the agent in plain language:

1. "Generate a knight sprite concept with the pixel-art LoRA" → `comfyui_generate`.
2. "Put it in Krita so I can fix the helmet" → `krita_new_document` + `krita_add_image_layer`. You paint; Krita AI Diffusion can inpaint the area you select.
3. "Take my Krita painting and make it a 64 px sprite with the Endesga 32 palette" → `krita_export`, then the pixelize tools into an `.ase`.
4. Clean up and animate in LibreSprite (by hand or with the drawing and animation tools).

## What is installed

| What | Where |
|---|---|
| Puretty Bridge plugin | `%APPDATA%\krita\pykrita\puretty_bridge\` + `puretty_bridge.desktop` |
| Krita AI Diffusion 1.53.0 | `%APPDATA%\krita\pykrita\ai_diffusion\` + `ai_diffusion.desktop` |
| Both enabled | `%LOCALAPPDATA%\kritarc`, section `[python]`: `enable_puretty_bridge=true`, `enable_ai_diffusion=true` |
| kritarc backups | `%LOCALAPPDATA%\kritarc.bak-<date>-<time>` (one per install run; the oldest is the original) |

Krita AI Diffusion uses the ComfyUI installed in `C:\AI\ComfyUI` (with all the custom nodes and SDXL models it requires): its `settings.json` (`%APPDATA%\krita\ai_diffusion\settings.json`, original saved as `settings.json.bak-20261018`) is set to the external server `127.0.0.1:8188`. Start ComfyUI with `C:\AI\ComfyUI\run_comfyui.bat` before using the AI docker; it connects by itself. Use the **Digital Artwork** style (ZavyChroma XL).

To install again (on another machine, or after a Krita reset), close Krita and run:

```
python install.py --with-ai-diffusion
```

## Puretty Bridge

A small Krita Python plugin. When Krita starts it listens on **127.0.0.1:8765** (or the next free port; set `PURETTY_BRIDGE_PORT` to move it) and writes the port and a random per-session token to `%APPDATA%\krita\puretty_bridge.json`. The MCP tools read that file. Requests without the token are refused, so only programs running as you can use it. The file is removed when Krita quits. **Tools > Scripts > Puretty Bridge Status** shows the port.

Krita's API is only called on Krita's GUI thread, so the bridge is safe to use while you paint. A long export briefly pauses the UI.

| Tool | Does |
|---|---|
| `krita_status` | Krita version, bridge version, open documents |
| `krita_new_document` | New RGBA 8-bit document with one transparent layer (`white_background` for Krita's usual white one) |
| `krita_open` | Open a `.kra`/`.png`/`.psd`/... and make it active |
| `krita_list_layers` | Layer tree with type, visibility, opacity, blend mode and bounds |
| `krita_add_image_layer` | Load a PNG/JPG as a new top layer at x, y |
| `krita_export` | Flattened canvas or one layer to PNG/JPG/WEBP/BMP/TIFF, optional nearest-neighbour upscale |
| `krita_view_canvas` | Return the canvas as an image the agent can look at (pixel art is shown enlarged with nearest-neighbour) |
| `krita_save_document` | Save to its file, or Save As (`.kra` keeps layers) |
| `krita_close_document` | Close **without saving** |
| `krita_run_python` | Run Python inside Krita (`app`, `doc`, `Krita` predefined; `print()` and `result` are returned) |

`krita_run_python` runs any code inside Krita, like Krita's Scripter. To stop local programs from controlling Krita, disable the plugin in **Settings > Configure Krita > Python Plugin Manager**.

## Tested loop

With ComfyUI running, the agent put a generated witch on a Krita canvas
(`krita_new_document`, `krita_add_image_layer`), asked Krita AI Diffusion for a
refine pass at strength 0.6 ("holding a glowing green magic staff") through
`krita_run_python`, applied the result as a layer and exported it
(`examples/workflow/03_krita`). The broom became a crystal staff in about 20 s, with
pose, outfit and pixel style kept. A person gets the same from the AI docker. The
refined image was then cut out, pixelized and animated in LibreSprite
(`examples/workflow/07_sprite/witch_staff.ase`).

## Getting ComfyUI (another machine)

The simplest option is to let Krita AI Diffusion manage it:

1. Start Krita and open **Settings > Dockers > AI Image Generation**.
2. Click **Configure** → **Local Managed Server** → **Install**. Pick the CUDA backend (on Windows it installs PyTorch for CUDA 13.0, which supports the RTX 50 series) and one or two checkpoints. For pixel art, **ZavyChroma (SDXL – Artwork)** pairs well with the Pixel Art XL LoRA (see `../comfyui/README.md`).
3. It installs ComfyUI into `%APPDATA%\krita\ai_diffusion\server` and runs it on **127.0.0.1:8188**, the address the `comfyui_*` tools try first. ComfyUI only runs while Krita (with that server started) is open.

If you already have ComfyUI (portable on port 8188, or ComfyUI Desktop on port 8000), the tools find it. Point Krita AI Diffusion at it with **Custom Server** instead of installing a second copy.

## Uninstall

```
python install.py --uninstall --with-ai-diffusion
```

That removes both plugins and their `[python]` keys (leave out `--with-ai-diffusion` to keep Krita AI Diffusion). Delete `%APPDATA%\krita\ai_diffusion` to remove its managed ComfyUI and downloaded models. To return kritarc exactly to how it was, copy the oldest `kritarc.bak-*` back over `%LOCALAPPDATA%\kritarc` while Krita is closed.

## Known issue

In testing, Krita 5.3.3 crashed once while starting, before any document opened. The crash was in its input manager (`KisInputManager::toolProxy`, while setting up shortcuts), with no plugin code on the stack. It did not happen again in later launches. If Krita closes right after launching, start it again.

## Tests

`libresprite-mcp/tests/test_krita.py` runs against a live Krita with the bridge. It creates, exports and closes its own temporary documents and is skipped when Krita isn't running. Verified on Krita 5.3.3 (Windows 11): 7 passed.
