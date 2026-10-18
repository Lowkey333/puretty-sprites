from .. import mcp

_COMMON = [
    "Look at your work constantly: view_frame after every few edits (it shows you the image), "
    "view_animation for a strip of frames, render_onion_skin for motion continuity.",
    "Plan the palette first: generate_color_ramp per material, or apply_palette_preset for a retro look.",
    "Build in layers (body, clothing, weapon, effects) so parts can be animated independently.",
]

_GUIDES = {
    "character": [
        "Block the silhouette on frame 1 with draw_rectangle_at/draw_ellipse_at/fill_area_at, "
        "then refine with draw_pixels_at; outline_cel for readability.",
        "Copy the base pose with copy_frame or propagate_cels, then change only what moves.",
        "Animate by moving cels (tween_cel_positions_eased, oscillate_cel_positions) before redrawing pixels.",
        "Add secondary motion on separate layers (hair, cape, accessories).",
        "Tag loops with set_tag (idle, walk, attack), check them with view_animation, export with "
        "export_tag (GIF) or export_spritesheet (PNG + JSON for the engine).",
        "Run audit_animation or animation_sanitize to validate layer coverage and overlaps.",
    ],
    "environment": [
        "Build the base scene once and copy it with copy_sprite for variants.",
        "Use propagate_cels for static layers (sky, mountains, ground).",
        "Animate only the moving layers (clouds, water, birds) with tween/offset tools.",
        "Use apply_dither_gradient and palette tools for a consistent color mood.",
        "Copy reusable assets across scenes with copy_layers_between_sprites.",
    ],
    "ai_import": [
        "Generated images (ComfyUI, Krita AI Diffusion, Blender renders) are not pixel art yet: "
        "run pixelize_image (or import_pixelized_image) to snap them to a pixel grid and a palette.",
        "Pick the target size first (e.g. 32 or 64 px tall) and a palette (preset, the sprite's own, "
        "or extracted with extract_palette_from_image) so every frame matches.",
        "For frame sequences (Blender turnarounds, ComfyUI batches) use frames_to_sprite: one call "
        "builds the animated .ase with tags and a shared palette.",
        "Then clean up by hand: stray pixels (get_color_stats shows near-duplicate colors), "
        "outline_cel, replace_color, and check with view_frame.",
    ],
}


@mcp.tool()
async def animation_workflow_guide(use_case: str = "character") -> str:
    """Return a concise workflow guide: "character", "environment" or "ai_import"."""
    use_case = (use_case or "character").strip().lower()
    bullets = _COMMON + _GUIDES.get(use_case, _GUIDES["character"])
    lines = ["LibreSprite Sprite Workflow Guide", f"Use case: {use_case}"]
    lines.extend(f"- {b}" for b in bullets)
    lines.append("- Files are LibreSprite .ase files: open_in_libresprite to let a human edit them "
                 "(close them in LibreSprite before the tools edit again).")
    return "\n".join(lines)
