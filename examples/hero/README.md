# A second character: realistic proportions

The witch in `../workflow` is a chibi. This adventurer went through the same steps
to show the route on realistic proportions:

1. `hero_concept.png`: `comfyui_generate` with ZavyChroma XL, asking for a classic
   A-pose ("arms held out diagonally downward at 45 degrees", seed 5151).
2. TRELLIS.2 in ComfyUI turned it into a textured model in 3 minutes (the 66 MB
   `.glb` isn't committed).
3. `blender_auto_rig` (15 s): DWPose found all 18 joints on a front render, so the legs
   and arms use its joints; `hero_skeleton.png` is the picture the tool returned.
4. `blender_render_sprites` + `frames_to_sprite`: `hero_96.ase`, 96 px, 8 directions,
   the eight built-in animations (`hero_96_all_animations.gif`, `hero_96_Walk_SE.gif`,
   `hero_96_Run_SE.gif`).
5. `blender_add_animation`: a run, a high kick and a punch combo from motion capture,
   rendered the same way (`hero_96_mocap.gif`). The clips themselves aren't in the
   repository.

`hero_96_mocap.gif` uses motion data from [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu)
(clips 09_01, 143_24, 143_23 in the cgspeed BVH conversion). The database was created
with funding from NSF EIA-0196217.
