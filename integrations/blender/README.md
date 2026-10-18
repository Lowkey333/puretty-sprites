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
- `auto_rig.py` gives a generated (unrigged) humanoid mesh a skeleton with feet
  and hands, heat-weighted skinning and eight in-place animations (Idle, Walk,
  Run, Attack, Cast, Hit, Death, Jump), so image-to-3D output can be animated.
- `retarget.py` puts motion capture (a BVH file, or a Mixamo FBX) on that rig as a
  new action: real runs, fights, jumps and falls.
- The LibreSprite MCP server (`libresprite-mcp/libresprite_mcp/tools/blender.py` and
  `blender_anim.py`) wraps them as five tools: `blender_status`,
  `blender_list_animations`, `blender_auto_rig`, `blender_add_animation` and
  `blender_render_sprites`.

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
  `examples/workflow/06_blender/witch_rigged_1k.blend` (and `.glb` with its eight
  animations), with a 1K texture to keep it small; the full 55 MB model isn't
  committed. `witch_rigged_skeleton.png` beside it shows where the auto-rig put her joints.
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

- **`auto_rig.py` / the `blender_auto_rig` tool** (used for the examples; about 15
  seconds): for a generated mesh standing in an A- or T-pose. See "The auto-rig"
  below. Check the result in the skeleton picture it returns, and the motion with
  a `lit` render at 192 px; if a joint is off, rig again with `landmarks`
  (e.g. `"hips=0.25,neck=0.5"`).
- **Motion capture on the auto-rig (`retarget.py` / `blender_add_animation`)**: adds
  real moves to a rig made by `blender_auto_rig`, one action per call, from a BVH
  file or a Mixamo FBX downloaded "Without Skin". Each bone follows the matching
  segment of the clip, so a T-pose actor can drive an A-pose chibi; the clip is
  turned to face -Y, scaled to the character's legs and kept on the floor.
  `loop=True` cuts the cleanest 0.4-1.6 s cycle (runs, walks); `start_seconds` /
  `end_seconds` pick a one-shot such as an attack, jump or death out of a longer
  take. Clips: Mixamo (tick "In Place" for runs) or the
  [CMU database](http://mocap.cs.cmu.edu) (2500+ clips, usable in commercial
  products, but don't redistribute the files; BVH at
  [github.com/Shriinivas/cmubvh](https://github.com/Shriinivas/cmubvh)). Each clip
  takes about 2 seconds. From a terminal:
  `blender -b hero.blend -P retarget.py -- --source=run.bvh --name=MocapRun --loop --in-place`.
  Limits: wrists stay in line with the forearms, feet aren't locked (a one-shot that
  steps slides the standing foot with `in_place`; pass `in_place=False` to keep the
  footwork), chibis get small strides because motion scales with leg length, and a
  clip that ends lying down widens the framing of everything rendered with it.
- **Mixamo** (free, Adobe account): upload the model as FBX or OBJ, place the
  auto-rig markers, pick animations and download FBX. Tick **In Place** for walks
  and runs; animations that move the character forward walk out of the frame. The
  better choice for production animation.
- **Rigify** (built into Blender): more control and works offline, more manual work.
- **No rig at all**: for idle poses, props and turnarounds, render `animation="static"`.

A `.blend` can hold several actions (Idle, Walk, Attack, ...). `blender_list_animations`
shows them and `animation="all"` renders every one.

### The auto-rig

1. **Joints from the model itself.** A white silhouette of the front view gives the
   gap between the legs (followed up to the crotch, or to the hem of a robe), the
   neck (the first strong narrowing below the head), the hands (the farthest points
   between legs and neck) and the armpits; the vertices give the ankles (where the
   foot's front-to-back depth drops to the leg's) and the toes. Knees, elbows and
   wrists follow from those by proportion.
2. **A pose detector, when it can see a skeleton.** With ComfyUI running (and the
   `comfyui_controlnet_aux` nodes), the tool renders the front view with EEVEE and
   runs DWPose on it. When it finds a complete skeleton that lies on the model, its
   hips, knees, ankles, shoulders, elbows and wrists replace the estimates. It does
   on realistic proportions; on the chibi witch it only found an ankle or two, so
   her joints come from step 1 (the result says which source each limb used).
3. **Skeleton:** root, hips, chest, head, thighs, shins, feet, upper arms, forearms
   and hands (`.L` is the character's left, +X).
4. **Skinning:** Blender's heat weights fail on the original mesh (generated meshes
   are full of holes and loose pieces), so they are computed on a watertight voxel
   copy and transferred back, then limited by anatomy with soft edges: legs below
   the hips, arms out from the shoulders and not behind the body (hair), everything
   clearly above the neck on the head. A robe or skirt is split between the legs
   and the hips, so it stretches as the legs move instead of tearing. Distance
   weights are the fallback (`weights="distance"` forces them).
5. **Animations,** posed with forward kinematics so the lowest foot stays on the
   ground (the body bobs by itself): `Idle`, `Walk`, `Run` loop (cyclic, last frame
   repeats the first); `Attack`, `Cast`, `Hit`, `Death` and `Jump` play once. Arms
   are lowered from the A-pose, and big heads (chibis, large hats) get toned-down
   spine and head tilts so the hat doesn't swing wildly. The jump only hops: every
   animation is framed by one camera, so a high jump would shrink the character in
   all of them; move the sprite up in the game instead. `animations="Idle,Walk"`
   picks a subset.

The skeleton picture and the pose detector's files go in a `<name>_rig` folder next to
the `.blend` (`skeleton.png`, and with pose detection `front.png`, `front_pose.png`,
`pose.json`). From a terminal:

```
blender -b --factory-startup -P auto_rig.py -- --model=hero.glb --out=hero.blend
blender -b --factory-startup -P auto_rig.py -- --model=hero.glb --out=hero.blend --animations=Idle,Walk,Attack --landmarks=hips=0.48
```

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
material colors with hard edges, and that glTF imports work.
`tests/test_auto_rig.py` builds an A-pose figure (and one in a robe) in Blender, rigs
it and measures the result: joints near where they belong, every bone moving part of
the mesh, feet on the ground and alternating through the walk, the jump leaving the
floor, the death ending low and inside the frame, the robe detected, pose keypoints
taking over the legs and arms. `tests/test_retarget.py` writes BVH clips with known
motion in Mixamo, CMU and ASF naming and checks the retargeted actions: frame counts,
legs in opposite phase, facing -Y, no drift in place, the vertical bob kept, the
planted loop length found, a T-pose lead-in dropped, missing foot bones reported.
They skip when Blender isn't found. Run them with
`uv run pytest tests/test_blender.py tests/test_auto_rig.py tests/test_retarget.py`
from `libresprite-mcp`.
