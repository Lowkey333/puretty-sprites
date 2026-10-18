# Blender: characters from concept art to sprites

Drawing every frame of a character by hand (or asking an image model to) gives
you characters that change shape from frame to frame and from one direction to
the next. Building the character once in 3D and rendering it fixes that: every
direction and every animation frame comes from the same model, at the same
scale, lit the same way. This folder turns a 3D character into sprite frames;
the LibreSprite MCP then pixelizes them into an editable `.ase` sprite.

```
concept art ──> 3D model ──> rig + animations ──> sprite_render.py ──> frames_to_sprite ──> LibreSprite
  (PNG)       (image-to-3D)   (Mixamo/Rigify)     (8 directions,      (pixelize, one       (.ase you can
                                                    PNG + manifest)     palette, tags)       clean up)
```

## What's here

- `sprite_render.py` renders a character from 1, 2, 4, 8 or 16 directions with an
  orthographic camera, for one or every animation, and writes the frames plus a
  `manifest.json`. It runs headless or inside a live Blender session.
- `examples/test_character.blend` is a small rigged figure with `Idle` and `Walk`
  actions, rebuilt by `examples/make_test_character.py`. The tests use it, and it's
  a handy way to try settings.
- `auto_rig.py` gives a generated (unrigged) humanoid mesh a quick skeleton and
  in-place `Idle` and `Walk` actions, so image-to-3D output can be animated.
- The LibreSprite MCP server (`libresprite-mcp/libresprite_mcp/tools/blender.py`)
  wraps them as four tools: `blender_status`, `blender_list_animations`,
  `blender_auto_rig` and `blender_render_sprites`.

Tested with Blender 5.2.1 LTS. Blender is found through `BLENDER_PATH` (the exe or
its folder), then PATH, then `Program Files\Blender Foundation\Blender *` (newest first).

## 1. Concept art to a 3D model

Pick whichever fits your budget and hardware. All of these produce a `.glb`/`.fbx`
that `sprite_render.py` can load directly.

- **MCP for Blender (blender-mcp)**: its `generate_3d` tool turns a prompt or a
  reference image into a model inside your open Blender, through Hunyuan3D, Hyper3D
  Rodin, Tripo or Meshy. Those need your own API key for that service (or the
  project's paid Premium plan); Meshy needs a paid Meshy plan. The same server can
  also pull free assets from Poly Haven (no key), Poly Pizza (free key) and
  Sketchfab (key). See "Setting up MCP for Blender" below.
- **Locally on your GPU (set up on this machine)**: ComfyUI's own TRELLIS.2
  template turns one concept image into a textured `.glb`:
  `comfyui_run_workflow("template:3d_pixal3d_trellis2_image_to_model", overrides={"316.value": True}, images={"122.image": "concept.png"})`.
  On the RTX 5060 Ti it took 6.5 minutes and gave a 700k-triangle model that holds up
  from every side. Nothing leaves your machine. The rigged result is in the repo as
  `examples/workflow/06_blender/witch_rigged_1k.blend` (and `.glb` with its Idle and
  Walk animations), with a 1K texture to keep it small; the full 55 MB model isn't
  committed.
  Concept art for it should be one character, front view, arms away from the body,
  on a plain background (see `examples/workflow/04_concept_for_3d`).
- **By hand or with the AI in Blender**: blocking out a character from primitives
  works well at sprite sizes. At 32-64 pixels tall, a clean silhouette matters far
  more than topology or texture detail.

Generated models are usually one mesh with a baked texture, often facing an odd
direction and at an odd scale. Rotate it so it faces -Y (Blender's front view),
apply scale, and put its feet on z=0 before rigging. If it faces another axis, pass
`front_axis` (`+Y`, `+X`, `-X`) instead.

## 2. Rig and animate

- **`auto_rig.py` / the `blender_auto_rig` tool** (the five-minute version, used for
  the examples): decimates a generated mesh, places a humanoid skeleton, skins it
  with distance weights limited by anatomy (Blender's automatic heat weights fail on
  most generated meshes) and adds in-place `Idle` and `Walk` actions. Fine for
  chibi/stylized characters in an A- or T-pose at sprite sizes. Check the walk with a
  `lit` render at 192 px; if a part follows the wrong bone, pass `landmarks`
  (e.g. `"hips=0.25,neck=0.5"`).
- **Mixamo** (free, Adobe account): upload the model as FBX or OBJ, place the
  auto-rig markers, pick animations and download FBX. Tick **In Place** for walks
  and runs; animations that move the character forward walk out of the frame. The
  better choice for production animation.
- **Rigify** (built into Blender): more control and works offline, more manual work.
- **No rig at all**: for idle poses, props and turnarounds, render `animation="static"`.

A `.blend` can hold several actions (Idle, Walk, Attack, ...). `blender_list_animations`
shows them and `animation="all"` renders every one.

## 3. Render the frames

From the MCP (any agent connected to the LibreSprite server):

```
blender_render_sprites(model_path="C:/art/hero.fbx", output_dir="C:/art/renders/hero",
                       directions=8, frame_size=64, elevation=30, animation="Walk",
                       frame_step=2, shading="toon")
```

Or from a terminal:

```
blender -b --factory-startup -P sprite_render.py -- --model=C:/art/hero.fbx --out=C:/art/renders/hero --directions=8 --size=64 --animation=Walk --frame-step=2
blender -b --factory-startup C:/art/hero.blend -P sprite_render.py -- --out=C:/art/renders/hero --animation=all
blender -b --factory-startup -P sprite_render.py -- --model=C:/art/hero.glb --list
```

Write options as `--key=value`: `--front -Y` would read `-Y` as another option.

What the settings do:

- **directions** are named by where the character faces on screen: `S` faces the
  viewer, then clockwise `SW, W, NW, N, NE, E, SE`. `direction_names="E,W"` renders a
  side-scroller's two views; mirror one of them in LibreSprite instead to halve the work.
- **elevation** is the camera pitch: `0` for a side view, `30` for a classic top-down
  RPG, `45`-`60` for steeper overhead views.
- **shading**: `toon` gives two (or three, `toon_steps=3`) flat light bands per
  material, which pixelizes into a small palette; `flat` renders unlit base colors,
  for shading by hand afterwards; `lit` keeps the model's own materials and smooth
  lighting. The light always comes from the upper left of the screen, in every direction.
- **supersample** (default 4) renders at 4x the frame size with anti-aliasing so the
  pixelize step can pick clean pixels. `supersample=1` renders the exact size with hard
  edges and no anti-aliasing, already close to pixel art.
- **frame_step**: pixel-art animations rarely need every frame; 2 or 3 keeps motion
  readable and the sheet small. Actions marked cyclic in Blender drop their last
  frame automatically (it repeats the first); use `trim_loop=True` for looping glTF/FBX
  animations, which don't carry that flag.

The camera framing is computed once from every frame and direction being rendered,
so the character never changes size between views, and the ground point is snapped
to a whole pixel. `manifest.json` records that **pivot**: where the character's
vertical axis meets the floor, in frame pixels (for example `x=32, y=58` in a 64x64
frame). With a tilted camera, the foot nearer the viewer is drawn slightly below it.

Output: `<output_dir>/<animation>/<direction>/<frame>.png` and `manifest.json`,
which lists the frames per animation and direction, and once more as a single
ordered `sequence` (animation, then direction, then frame) with matching 1-based
`tags` such as `Walk_S`.

### Inside a live Blender

With MCP for Blender connected, the agent can model or generate a character in your
open Blender and render it in place through `execute_blender_code`:

```python
import sys; sys.path.append(r"C:\AI\Puretty-sprites\integrations\blender")
import sprite_render
sprite_render.render(out_dir=r"C:\art\renders\hero", directions=8, size=64, animation="Walk")
```

Your scene is left as it was: materials, render settings, the active action and
object visibility are restored, and the temporary camera rig is deleted.

## 4. Pixelize and assemble in LibreSprite

Feed the manifest to the LibreSprite MCP's `frames_to_sprite`:

```
frames_to_sprite(output_filename="C:/art/hero_walk.ase", image_paths=<manifest "sequence">,
                 width=64, height=64, duration_ms=<manifest "duration_ms">, tags=<manifest "tags">,
                 palette="auto:16", outline_color="#1A1C2C")
```

That builds one animated sprite with a shared palette (so colors don't flicker
between frames) and one tag per animation and direction. Then open it in LibreSprite
(`open_in_libresprite`) to fix stray pixels and add hand-drawn details; that cleanup
pass is what makes rendered sprites look drawn.

## Setting up MCP for Blender (blender-mcp)

The headless tools above don't need it. It's for working with the AI inside a live
Blender: modelling, image-to-3D generation and scene inspection. It has two parts, an
add-on running inside Blender and an MCP server your AI client starts.

**On this machine both are set up**: the add-on (1.8) is installed and enabled in
Blender 5.2, and the server is in the project's `.mcp.json`. Open Blender and the
add-on starts listening on `localhost:9876` by itself; it was tested by driving the
open Blender through `get_scene_info`, `execute_blender_code` and `look`
(`examples/workflow/06_blender/blender_mcp_look.png`). To set it up elsewhere:

1. Install and enable the add-on (copies it to
   `%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons\blender_mcp.py`):

   ```
   uvx mcp-for-blender install-addon
   ```

   Then in Blender: **Edit > Preferences > Add-ons**, enable **Interface: MCP for
   Blender**. Or enable it from a terminal (this saves your preferences, so close
   Blender first):

   ```
   blender -b --python-expr "import bpy; bpy.ops.preferences.addon_enable(module='blender_mcp'); bpy.ops.wm.save_userpref()"
   ```

2. Add the server to your MCP client. For Claude Code, in the project's `.mcp.json`:

   ```json
   "blender": {
     "command": "uvx",
     "args": ["mcp-for-blender"],
     "env": { "DISABLE_TELEMETRY": "true" }
   }
   ```

   `DISABLE_TELEMETRY` turns off the anonymous usage counts the server sends by
   default (prompts, code and screenshots are only collected if you opt in). If uv
   picks a Python that breaks the install, add `"--python", "3.12"` before the package name.

3. Open Blender. The add-on starts its server on `localhost:9876` when Blender opens
   (never in `blender -b`); check the **MCP for Blender** tab in the 3D view's sidebar (N).

Be aware that the add-on's socket has no authentication: anything on your machine
that can reach port 9876 can run Python inside Blender, and the AI's
`execute_blender_code` tool runs whatever it writes. Save your work before letting it
loose, and set `BLENDER_MCP_SAFE_MODE=1` in the server's `env` to block scripts that
touch files, the network or other programs. API keys for the generators go in the
add-on's preferences or in `BLENDERMCP_HYPER3D_API_KEY`, `BLENDERMCP_HUNYUAN3D_SECRET_ID`
/ `_SECRET_KEY`, `BLENDERMCP_MESHY_API_KEY`, `BLENDERMCP_SKETCHFAB_API_KEY` and
`BLENDERMCP_POLYPIZZA_API_KEY`.

To remove it: disable the add-on in Preferences and delete `blender_mcp.py` from the
add-ons folder, remove the `blender` entry from `.mcp.json`, and run
`uv cache clean mcp-for-blender` to drop uv's cached copy.

## Tests

`libresprite-mcp/tests/test_blender.py` renders the example character and checks the
directions, loop trimming, durations and tags, transparent backgrounds, that the head
and feet line up across directions, that exact-size flat renders contain only the
material colors with hard edges, and that glTF imports work. It skips when Blender
isn't found. Run it with `uv run pytest tests/test_blender.py` from `libresprite-mcp`.
