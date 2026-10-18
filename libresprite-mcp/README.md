# LibreSprite MCP

An MCP server that lets an AI agent (Claude Code, Claude Desktop, any MCP client) draw and animate pixel-art sprites as [LibreSprite](https://libresprite.github.io) files. It is a port of [aseprite-mcp](https://github.com/diivi/aseprite-mcp) by Divyansh Singh (MIT): the same tool names and arguments, for LibreSprite instead of Aseprite, plus tools for turning AI-generated images and Blender renders into pixel art.

**141 tools**: canvas, drawing, text, layers, regions, effects and filters, animation, palettes, transforms, slices, export, reading pixels, analysis, quality checks, the AI-image pipeline, Blender, ComfyUI, Krita, and LibreSprite itself.

![Witch idle animation drawn with the tools](../examples/witch_idle.gif)

*`examples/witch.ase`: drawn with nothing but these tools (shapes, outlines, a 4-frame `oscillate_cel_positions` bob), checked in LibreSprite with `check_libresprite_compat`, exported with `export_tag`.*

## How it works with LibreSprite

aseprite-mcp generates Lua and runs Aseprite in batch mode for every call. LibreSprite can't be driven that way:

- LibreSprite 1.3's headless scripts can edit the pixels of existing cels and nothing else: commands such as `NewLayer` and `NewFrame` are refused in batch mode, and there is no script API at all for layer opacity, blend modes, frame durations or tags (not even in the nightly builds).

So this server reads and writes LibreSprite's `.ase` format directly. `core/ase.py` mirrors LibreSprite 1.3's `ase_format.cpp`, and `core/render.py` ports its blend math, so what the tools see and export is what LibreSprite shows. Both are tested against the real program: files written here load in LibreSprite with the same layers, cels, tags and durations, and the renderer matches LibreSprite's own output pixel for pixel across all 16 blend modes and 80 randomized multi-layer sprites. Tool calls take milliseconds because no editor starts per call.

LibreSprite itself is used where it adds something:

| Tool | Uses LibreSprite to |
|---|---|
| `open_in_libresprite` | open a file in the editor for you |
| `run_libresprite_script` | run JavaScript in LibreSprite's own scripting engine (the counterpart of `run_lua_script`) |
| `run_libresprite_cli` | use its command line: `--sheet`, `--data`, `--split-layers`, `--frame-tag`, ... |
| `check_libresprite_compat` | load a file in LibreSprite and compare layers, cels and rendered pixels with the server's view |
| `libresprite_status` | report which LibreSprite is installed and what it can do |

LibreSprite is found through `LIBRESPRITE_PATH` (the executable or its folder; also read from `libresprite-mcp/.env`), PATH, the usual install folders and `../tools/libresprite`. Scripts prefer `LIBRESPRITE_SCRIPT_PATH` or `../tools/libresprite-nightly`, because nightly builds can run commands headless and 1.3 can't. Everything except the five tools above works without LibreSprite installed.

## Differences from aseprite-mcp

- **Layers are flat.** LibreSprite has no layer groups (it flattens them to `Group-Layer` names when opening an Aseprite file, and so does this server). `add_group` and the `group` arguments are gone.
- **No tilemaps** (LibreSprite has none), so the five tilemap tools are gone.
- **Slices live next to the sprite** in `<name>.slices.json`, because LibreSprite's file format has no slices. `export_spritesheet` puts them in the JSON data (`meta.slices`), where engines read pivots and 9-patch centers.
- **16 blend modes:** Aseprite's addition, subtract and divide don't exist in LibreSprite. Tags have no `pingpong_reverse`.
- **New frames are empty** (`add_frame`, `add_frames`; pass `copy_previous=True` to copy the last frame's cels). In Aseprite, `spr:newFrame()` copies the previous frame, which surprised more than it helped.
- **Active layer and frame** (`set_layer`, `set_frame`) are remembered by the server for the session, because LibreSprite stores neither in the file. Without them, `draw_pixels` and friends draw on the topmost layer, frame 1.
- **Native filters** (Aseprite's `app.command` outline, HSL, brightness/contrast, invert, convolution, quantization) are computed by the server; the convolution matrices are LibreSprite's own (`data/convmatr.def`). `outline_native` and `adjust_hsl_native` are covered by `outline_cel(place=...)` and `adjust_hsl` with a region.
- **New:** `view_frame` and `view_animation` return the image itself, so the agent sees its work without opening files; `canvas_size` (LibreSprite's Canvas Size); the pipeline, Blender, ComfyUI and Krita tools.
- Error replies are always `Failed to <action>: <reason>`, and missing cels are reported instead of silently succeeding.
- Indexed sprites: exports and previews follow LibreSprite's editor view. LibreSprite's own sheet exporter writes indexed sprites by palette index (ignoring layer opacity) and marks index 0 transparent even when another index is, so its sheets of indexed sprites can differ.
- Requires the 2.x MCP Python SDK (`MCPServer`); upstream's unpinned `mcp>=1.6` breaks on 2.x.

## Setup

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). LibreSprite is optional (see above).

```
cd libresprite-mcp
uv sync
uv run pytest            # 139 tests; live Krita/ComfyUI ones skip when those aren't running
```

Claude Code picks the server up from `../.mcp.json` when started in the Puretty-sprites folder. For other clients:

```json
{
  "mcpServers": {
    "libresprite": {
      "command": "uv",
      "args": ["--directory", "C:\\AI\\Puretty-sprites\\libresprite-mcp", "run", "libresprite-mcp"]
    }
  }
}
```

Environment variables (all optional): `LIBRESPRITE_PATH`, `LIBRESPRITE_SCRIPT_PATH`, `BLENDER_PATH`, `COMFYUI_URL`, `COMFYUI_OUTPUT_DIR`, `PURETTY_BRIDGE_PORT`.

## Recommended workflow for agents

1. **Plan the palette:** `generate_color_ramp` per material, or `apply_palette_preset` (pico8, endesga32, sweetie16, dawnbringer32, ...).
2. **Build in layers:** body, clothing, weapon, effects, so parts animate independently.
3. **Coarse to fine:** silhouettes with `draw_polygon` / `draw_ellipse_at` / `fill_area_at`, details with `draw_pixels_at`, then `outline_cel`.
4. **Look constantly:** `view_frame` after every few edits, `view_animation` for a strip of frames, `render_onion_skin` for motion, `get_color_stats` to keep the palette tight.
5. **Animate by moving cels** before redrawing: `propagate_cels` or `copy_frame`, then `tween_cel_positions_eased` / `oscillate_cel_positions`; `set_tag` per loop.
6. **Export:** `export_spritesheet` (PNG + LibreSprite/Aseprite JSON) or `export_tag` (GIF); hand the `.ase` to a human with `open_in_libresprite`.

`animation_workflow_guide` returns this as a checklist (`character`, `environment`, or `ai_import` for generated images).

## Tools

Frame indices are 1-based. Coordinates are sprite pixels. Drawing tools replace pixels (no blending) and convert colors to the sprite's mode.

### Canvas and drawing

| Tool | Description |
|---|---|
| `create_canvas` | New sprite (rgb, gray or indexed) with one layer and the DawnBringer 32 palette |
| `add_layer`, `add_frame`, `set_frame_duration` | Layers on top, empty frames, per-frame timing |
| `set_layer`, `set_frame` | Choose the active layer/frame for the tools below that have no layer argument |
| `draw_pixels`, `draw_line`, `draw_rectangle`, `draw_circle`, `fill_area` | Draw on the active cel |
| `draw_pixels_at`, `draw_line_at`, `draw_rectangle_at`, `draw_circle_at`, `draw_ellipse_at`, `fill_area_at` | Draw on a given layer and frame (cel created when missing) |
| `draw_polygon`, `draw_path` | Filled/outlined polygons, thick polylines |
| `apply_gradient_rect` | Smooth gradient (use `apply_dither_gradient` for palette-friendly blends) |
| `draw_text`, `measure_text`, `list_text_fonts` | Bitmap or TrueType text, rendered without antialiasing; fonts in `~/.libresprite-mcp/fonts` |

### Layers, regions, effects, filters

| Tool | Description |
|---|---|
| `delete_layer`, `rename_layer`, `duplicate_layer`, `reorder_layer` | Layer stack management |
| `set_layer_blend_mode`, `merge_layer_down`, `flatten_sprite` | LibreSprite's 16 blend modes; merges render exactly as LibreSprite does |
| `move_region`, `copy_region`, `erase_region`, `erase_color` | Rectangles of pixels; magic eraser |
| `outline_cel` | 1px outline outside or inside the shape, optionally with diagonals |
| `replace_color`, `adjust_hsl` | Recolor with tolerance; hue/saturation/lightness shift (optionally in a region) |
| `apply_dither_gradient`, `apply_dither_pattern` | Bayer 4x4 ordered dithering between two colors |
| `adjust_brightness_contrast`, `invert_colors`, `apply_convolution`, `list_convolution_matrices` | Filters with LibreSprite's 39 convolution matrices |

### Animation

| Tool | Description |
|---|---|
| `add_frames`, `delete_frame`, `duplicate_frame_range`, `set_frame_duration_all` | Frames and timing |
| `copy_frame`, `propagate_frame_to_range`, `copy_cel`, `propagate_cels` | Reuse poses across frames |
| `create_cel`, `clear_cel`, `set_cel_position`, `set_cel_opacity` | Cel lifecycle and placement |
| `tween_cel_positions`, `tween_cel_positions_eased`, `offset_cel_positions`, `oscillate_cel_positions` | Motion by moving cels (linear, ease_in/out, smoothstep, sine bobbing) |
| `tween_cel_opacity_eased`, `tween_cel_scale_eased` | Fades and nearest-neighbor scaling |
| `set_tag`, `delete_tag` | Animation tags (forward, reverse, pingpong) |
| `set_layer_visibility`, `set_layer_opacity` | Layer display |
| `set_onion_skin` | LibreSprite's onion-skin defaults in `libresprite.ini` (for previews use `render_onion_skin`) |
| `get_sprite_info` | Size, mode, durations, layers with their cel frames, tags, warnings |

### Palette and transform

| Tool | Description |
|---|---|
| `get_palette`, `set_palette`, `list_palette_presets`, `apply_palette_preset` | 10 retro presets |
| `generate_color_ramp` | Hue-shifted shading ramps |
| `quantize_to_palette`, `remap_colors_in_cel_range`, `extract_palette` | Make art conform to a palette; build an optimal palette from the art |
| `set_color_mode` | rgb, grayscale, indexed (adds a transparent entry at index 0 when needed, so black stays black) |
| `flip_layer`, `rotate_layer` | Per-cel flips and 90° rotations |
| `resize_canvas`, `crop_canvas`, `canvas_size` | Scale everything, crop, or grow/shrink the canvas with an anchor |

### Export, viewing and reading

| Tool | Description |
|---|---|
| `view_frame`, `view_animation` | Return the frame (or a numbered strip of frames) as an image for the agent to look at |
| `export_spritesheet` | Horizontal/vertical/rows/columns/packed sheets with LibreSprite-format JSON (frames, durations, frameTags, layers, slices) |
| `export_sprite`, `export_frame`, `export_tag`, `export_layers` | PNG/GIF/JPG/WebP, scaled frames, tag GIFs honoring the direction, one PNG per layer |
| `import_image_as_layer`, `copy_sprite`, `copy_layers_between_sprites` | Bring images or layers in |
| `create_slice`, `set_slice_center`, `set_slice_pivot`, `list_slices`, `delete_slice` | Named regions, 9-patch centers, pivots (sidecar file) |
| `get_pixel_color`, `get_pixels_rect`, `get_composite_pixel`, `get_composite_rect` | Read one layer or the visible composite |
| `render_onion_skin`, `compare_frames`, `get_color_stats` | Motion ghosts (optionally LibreSprite's red/blue tint), frame diffs, color histograms |
| `ensure_layers_present`, `validate_scene`, `audit_animation`, `animation_sanitize` | Animation consistency checks and fixes |
| `start_preview_server`, `stop_preview_server`, `animation_workflow_guide` | Local HTTP preview of exports; workflow checklist |

### From AI images and 3D renders to pixel art

Generated images aren't pixel art: their "pixels" are off-grid, with soft edges, hundreds of colors and a background. `core/pixelize.py` fixes that: background removal by flood fill from the border (interior pixels of the same color survive), colors snapped to a palette (a preset, the sprite's own, or k-means), block downscaling where each output pixel takes its block's dominant color (no blurry in-between colors), stray-pixel cleanup and an optional outline. It can also detect the hidden grid of upscaled "fake pixel art".

| Tool | Description |
|---|---|
| `pixelize_image` | Image to pixel-art PNG at a target size, shows you the result |
| `import_pixelized_image` | The same, straight onto a layer of a sprite (snaps to the sprite's palette by default) |
| `extract_palette_from_image` | k-means palette from concept art, optionally set on a sprite |
| `frames_to_sprite` | Folder or list of frames (or a Blender `manifest.json`) to an animated `.ase` with ONE shared palette, tags, durations and the ground pivot |
| `import_spritesheet_image` | Slice an existing sheet image (grid) into frames |

### Blender, ComfyUI, Krita

| Tool | Description |
|---|---|
| `blender_status`, `blender_list_animations`, `blender_render_sprites` | Render a 3D character from 1-16 directions with an orthographic camera and toon/flat shading, then feed the manifest to `frames_to_sprite` (see `../integrations/blender`) |
| `blender_auto_rig` | Give a generated (unrigged) humanoid mesh a skeleton plus Idle and Walk actions, ready to render |
| `comfyui_status`, `comfyui_list_models`, `comfyui_generate`, `comfyui_img2img` | Generate and restyle images on a local ComfyUI (see `../integrations/comfyui`) |
| `comfyui_run_workflow`, `comfyui_list_templates` | Run any workflow: API format, a UI-format save (converted with the server's node definitions) or a bundled template such as `template:3d_pixal3d_trellis2_image_to_model` (image to textured 3D model) |
| `comfyui_remove_background` | AI cutout (BiRefNet): removes floor shadows and enclosed background before pixelizing |
| `krita_status`, `krita_new_document`, `krita_open`, `krita_list_layers`, `krita_add_image_layer`, `krita_export`, `krita_view_canvas`, `krita_save_document`, `krita_close_document`, `krita_run_python` | Work in a running Krita through the Puretty Bridge plugin (see `../integrations/krita`) |

### LibreSprite

`libresprite_status`, `open_in_libresprite`, `run_libresprite_script`, `run_libresprite_cli`, `check_libresprite_compat` (see the table at the top).

`run_libresprite_script` and `krita_run_python` run arbitrary code with your permissions, like upstream's `run_lua_script`. Only pass code you trust.

## Layout

```
libresprite_mcp/
  core/ase.py        .ase reader/writer (LibreSprite's subset)
  core/render.py     LibreSprite's compositing and blend modes
  core/raster.py     lines, ellipses (Zingl), polygons, flood fill, outlines, dithering
  core/pixelize.py   generated image -> pixel art
  core/sprite.py     load/edit/save helpers shared by the tools
  core/libresprite.py  finding and running LibreSprite
  tools/*.py         one module per tool category
tests/               139 tests; test_libresprite_parity.py checks against the real LibreSprite
```
