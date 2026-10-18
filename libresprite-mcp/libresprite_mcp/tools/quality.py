import json
from typing import List

from ..core import ase
from ..core.sprite import ToolError, blank_image, content_bounds, edit, find_layer, frame_range, load, save
from .. import mcp

# Cel bounds below are measured on opaque pixels, in sprite coordinates.
# The drawing tools keep cels canvas-sized, so the cel image rectangle
# would make every pair of drawn layers "overlap" and every cel look
# "full canvas"; the painted area is what these checks are about.


def _parse_layer_frame_ranges(layer_frame_ranges: List[str] | None) -> dict[str, list[tuple[int, int]]]:
    """["layer:1-8,17-24", "clouds:1-12"] -> {"layer": [(1, 8), (17, 24)], ...} (1-based)."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    for entry in layer_frame_ranges or []:
        if not entry or ":" not in entry:
            continue
        layer, ranges_part = entry.split(":", 1)
        layer = layer.strip()
        if not layer:
            continue
        spans = []
        for span in ranges_part.split(","):
            span = span.strip()
            if "-" in span:
                left, right = span.split("-", 1)
                try:
                    start, end = int(left), int(right)
                except ValueError:
                    continue
                if start > 0 and end >= start:
                    spans.append((start, end))
        if spans:
            ranges[layer] = spans
    return ranges


def _parse_overlap_pairs(overlap_pairs: List[str] | None) -> list[tuple[str, str]]:
    """["layerA,layerB", "layerC:layerD"] -> [("layerA", "layerB"), ...]."""
    pairs = []
    for entry in overlap_pairs or []:
        if not entry:
            continue
        if "," in entry:
            left, right = entry.split(",", 1)
        elif ":" in entry:
            left, right = entry.split(":", 1)
        else:
            continue
        left, right = left.strip(), right.strip()
        if left and right:
            pairs.append((left, right))
    return pairs


def _in_ranges(ranges: dict, layer_name: str, frame_index: int) -> bool:
    spans = ranges.get(layer_name)
    return spans is None or any(start <= frame_index <= end for start, end in spans)


def _target_layers(spr: ase.Sprite, layer_names: List[str] | None) -> list[ase.Layer]:
    if layer_names is None:
        return list(spr.layers)
    found = []
    for name in layer_names:
        layer = find_layer(spr, name)
        if layer is not None and layer not in found:
            found.append(layer)
    return found


def _rects_overlap(a, b) -> bool:
    return a[0] < b[0] + b[2] and a[0] + a[2] > b[0] and a[1] < b[1] + b[3] and a[1] + a[3] > b[1]


def _is_full_canvas(spr: ase.Sprite, bounds) -> bool:
    return bounds is not None and bounds[0] <= 0 and bounds[1] <= 0 \
        and bounds[0] + bounds[2] >= spr.width and bounds[1] + bounds[3] >= spr.height


def _end_frame_check(start_frame: int, end_frame: int | None) -> str | None:
    if start_frame < 1:
        return "Start frame must be >= 1"
    if end_frame is not None and end_frame < start_frame:
        return "End frame must be >= start frame"
    return None


@mcp.tool()
async def ensure_layers_present(
    filename: str,
    layer_names: List[str],
    start_frame: int = 1,
    end_frame: int | None = None
) -> str:
    """Ensure cels exist for layers across a frame range.

    Missing cels are created empty (canvas-sized, fully transparent), so
    every frame of the range has something to draw on or move.

    Args:
        filename: Sprite file to modify
        layer_names: Layers to fill in
        start_frame: First frame, starting at 1 (default 1)
        end_frame: Last frame, inclusive (default: the last frame)
    """
    if not layer_names:
        return "Layer names list cannot be empty"
    created = 0
    try:
        with edit(filename) as spr:
            frames = frame_range(spr, start_frame, end_frame)
            targets = _target_layers(spr, layer_names)
            if not targets:
                raise ToolError("No layers found: " + ", ".join(layer_names))
            for frame in frames:
                for layer in targets:
                    if frame not in layer.cels:
                        layer.cels[frame] = ase.Cel(blank_image(spr, spr.width, spr.height))
                        created += 1
    except ToolError as e:
        return f"Failed to ensure cels: {e}"
    return (
        f"Ensured cels for layers {', '.join(layer_names)} "
        f"on frames {start_frame}-{end_frame or 'end'} in {filename} ({created} created)"
    )


@mcp.tool()
async def validate_scene(
    filename: str,
    required_layers: List[str],
    start_frame: int = 1,
    end_frame: int | None = None
) -> str:
    """Validate presence of layers and cels across a frame range.

    Args:
        filename: Sprite file to inspect
        required_layers: Layers that must exist and have a cel on every frame
        start_frame: First frame, starting at 1 (default 1)
        end_frame: Last frame, inclusive (default: the last frame)

    Returns:
        JSON with missing layers and missing cels.
    """
    if not required_layers:
        return "Required layers list cannot be empty"
    try:
        spr = load(filename)
        frames = frame_range(spr, start_frame, end_frame)
    except ToolError as e:
        return f"Failed to validate scene: {e}"
    missing_layers, missing_cels = [], []
    for name in required_layers:
        layer = find_layer(spr, name)
        if layer is None:
            missing_layers.append(name)
            continue
        missing_cels += [{"layer": name, "frame": f + 1} for f in frames if f not in layer.cels]
    return json.dumps({
        "frames": spr.frame_count,
        "range": {"start": frames.start + 1, "end": frames.stop},
        "missing_layers": missing_layers,
        "missing_cels": missing_cels,
    })


@mcp.tool()
async def audit_animation(
    filename: str,
    start_frame: int = 1,
    end_frame: int | None = None,
    layer_names: List[str] | None = None,
    overlap_pairs: List[str] | None = None,
    layer_frame_ranges: List[str] | None = None,
    report_cels: bool = False,
    report_bounds: bool = False,
    max_overlaps: int = 200,
    max_out_of_range: int = 200
) -> str:
    """Audit animation frames for overlaps and out-of-range layer activity.

    overlap_pairs format: ["layerA,layerB", "layerC:layerD"]; two layers
    overlap on a frame when the bounding boxes of their painted pixels
    intersect.
    layer_frame_ranges format: ["layer:1-8,17-24", "clouds:1-12"]; a cel
    on a frame outside its layer's ranges is reported as out of range.
    Bounds are [x, y, w, h] of a cel's painted pixels in sprite coordinates
    (empty cels report w=h=0).

    Returns JSON for AI consumption (summary, overlaps, out_of_range,
    optional cels).
    """
    err = _end_frame_check(start_frame, end_frame)
    if err:
        return err
    if max_overlaps < 0 or max_out_of_range < 0:
        return "Max limits must be >= 0"
    try:
        spr = load(filename)
        frames = frame_range(spr, start_frame, end_frame)
    except ToolError as e:
        return f"Failed to audit animation: {e}"

    targets = _target_layers(spr, layer_names)
    by_name = {layer.name: layer for layer in targets}
    pairs = _parse_overlap_pairs(overlap_pairs)
    ranges = _parse_layer_frame_ranges(layer_frame_ranges)

    overlaps, out_of_range, cel_entries = [], [], []
    overlaps_total, total_cels = 0, 0
    for frame in frames:
        fi = frame + 1
        frame_cels = []
        for layer in targets:
            cel = layer.cels.get(frame)
            if cel is None:
                continue
            total_cels += 1
            if report_cels:
                entry = {"layer": layer.name}
                if report_bounds:
                    bounds = content_bounds(spr, cel) or (cel.x, cel.y, 0, 0)
                    entry.update(x=bounds[0], y=bounds[1], w=bounds[2], h=bounds[3])
                frame_cels.append(entry)
            if not _in_ranges(ranges, layer.name, fi) and len(out_of_range) < max_out_of_range:
                out_of_range.append({"frame": fi, "layer": layer.name})
        if report_cels:
            cel_entries.append({"frame": fi, "cels": frame_cels})

        for name_a, name_b in pairs:
            layer_a, layer_b = by_name.get(name_a), by_name.get(name_b)
            if layer_a is None or layer_b is None:
                continue
            cel_a, cel_b = layer_a.cels.get(frame), layer_b.cels.get(frame)
            if cel_a is None or cel_b is None:
                continue
            a_bounds, b_bounds = content_bounds(spr, cel_a), content_bounds(spr, cel_b)
            if a_bounds is None or b_bounds is None or not _rects_overlap(a_bounds, b_bounds):
                continue
            overlaps_total += 1
            if len(overlaps) < max_overlaps:
                entry = {"frame": fi, "a": layer_a.name, "b": layer_b.name}
                if report_bounds:
                    entry["a_bounds"] = list(a_bounds)
                    entry["b_bounds"] = list(b_bounds)
                overlaps.append(entry)

    result = {
        "frames": {"start": frames.start + 1, "end": frames.stop},
        "summary": {
            "total_layers": len(spr.layers),
            "layers_checked": len(targets),
            "total_cels": total_cels,
            "overlaps": len(overlaps),
            "overlaps_total": overlaps_total,
            "overlaps_truncated": overlaps_total > len(overlaps),
            "out_of_range": len(out_of_range),
        },
        "overlaps": overlaps,
        "out_of_range": out_of_range,
    }
    if report_cels:
        result["cels"] = cel_entries
    return json.dumps(result)


@mcp.tool()
async def animation_sanitize(
    filename: str,
    start_frame: int = 1,
    end_frame: int | None = None,
    layer_names: List[str] | None = None,
    layer_order: List[str] | None = None,
    layer_frame_ranges: List[str] | None = None,
    ensure_layers: List[str] | None = None,
    overlap_pairs: List[str] | None = None,
    report_bounds: bool = False,
    max_overlaps: int = 200,
    ignore_full_canvas_overlaps: bool = True,
    out_of_range_action: str = "set_opacity_zero",
    out_of_range_opacity: int = 0,
    report_only: bool = False,
    include_stats: bool = True
) -> str:
    """Normalize animation consistency and optionally apply fixes.

    layer_order: layer names bottom to top; unnamed layers keep their
    relative order above them (a background layer stays at the bottom).
    layer_frame_ranges format: ["layer:1-8,17-24", "clouds:1-12"]
    ensure_layers: create empty cels for these layers on every frame of the range
    out_of_range_action: "set_opacity_zero", "delete_cels", "none"
    ignore_full_canvas_overlaps: skip overlap checks when a cel's painted
    pixels cover the whole canvas (backgrounds overlap everything)
    report_only: analyse and report what would change without saving

    Returns JSON for AI consumption (summary, layer_stats, alerts, overlaps).
    """
    err = _end_frame_check(start_frame, end_frame)
    if err:
        return err
    if max_overlaps < 0:
        return "max_overlaps must be >= 0"
    if out_of_range_action not in {"set_opacity_zero", "delete_cels", "none"}:
        return "Unsupported out_of_range_action"
    if out_of_range_opacity < 0 or out_of_range_opacity > 255:
        return "out_of_range_opacity must be 0-255"

    ranges = _parse_layer_frame_ranges(layer_frame_ranges)
    pairs = _parse_overlap_pairs(overlap_pairs)
    try:
        spr = load(filename)
        frames = frame_range(spr, start_frame, end_frame)
    except ToolError as e:
        return f"Failed to sanitize animation: {e}"

    targets = _target_layers(spr, layer_names)
    by_name = {layer.name: layer for layer in targets}
    sanitized = {"reordered": False, "ensured": 0, "out_of_range": 0, "opacity_set": 0, "deleted": 0}
    changed = False

    if layer_order is not None:
        bottom = [spr.layers[0]] if spr.layers and spr.layers[0].is_background else []
        ordered = []
        for name in layer_order:
            layer = find_layer(spr, name)
            if layer is not None and layer not in ordered and layer not in bottom:
                ordered.append(layer)
        new_order = bottom + ordered + [l for l in spr.layers if l not in ordered and l not in bottom]
        if not report_only and new_order != spr.layers:
            spr.layers = new_order
            changed = True
        sanitized["reordered"] = True

    for name in ensure_layers or []:
        layer = find_layer(spr, name)
        if layer is None:
            continue
        for frame in frames:
            if frame not in layer.cels:
                sanitized["ensured"] += 1
                if not report_only:
                    layer.cels[frame] = ase.Cel(blank_image(spr, spr.width, spr.height))
                    changed = True

    stats = {layer.name: {"frames_active": 0, "cel_count": 0, "full_canvas_cels": 0, "bounds": None}
             for layer in targets}
    total_cels = empty_frames = overlaps_total = 0
    overlaps = []
    for frame in frames:
        fi = frame + 1
        has_cel = False
        for layer in targets:
            cel = layer.cels.get(frame)
            if cel is None:
                continue
            has_cel = True
            total_cels += 1
            entry = stats[layer.name]
            entry["cel_count"] += 1
            entry["frames_active"] += 1
            bounds = content_bounds(spr, cel)
            if _is_full_canvas(spr, bounds):
                entry["full_canvas_cels"] += 1
            if bounds is not None:
                x1, y1, x2, y2 = bounds[0], bounds[1], bounds[0] + bounds[2], bounds[1] + bounds[3]
                old = entry["bounds"]
                entry["bounds"] = [x1, y1, x2, y2] if old is None else \
                    [min(old[0], x1), min(old[1], y1), max(old[2], x2), max(old[3], y2)]
            if not _in_ranges(ranges, layer.name, fi):
                sanitized["out_of_range"] += 1
                if not report_only and out_of_range_action == "delete_cels":
                    del layer.cels[frame]
                    sanitized["deleted"] += 1
                    changed = True
                elif not report_only and out_of_range_action == "set_opacity_zero":
                    if any(f != frame and c is cel for f, c in layer.cels.items()):
                        cel = layer.cels[frame] = cel.copy()  # unlink: only hide this frame
                    cel.opacity = out_of_range_opacity
                    sanitized["opacity_set"] += 1
                    changed = True
        if not has_cel:
            empty_frames += 1

        for name_a, name_b in pairs:
            layer_a, layer_b = by_name.get(name_a), by_name.get(name_b)
            if layer_a is None or layer_b is None:
                continue
            cel_a, cel_b = layer_a.cels.get(frame), layer_b.cels.get(frame)
            if cel_a is None or cel_b is None:
                continue
            a_bounds, b_bounds = content_bounds(spr, cel_a), content_bounds(spr, cel_b)
            if a_bounds is None or b_bounds is None:
                continue
            if ignore_full_canvas_overlaps and (_is_full_canvas(spr, a_bounds) or _is_full_canvas(spr, b_bounds)):
                continue
            if not _rects_overlap(a_bounds, b_bounds):
                continue
            overlaps_total += 1
            if len(overlaps) < max_overlaps:
                entry = {"frame": fi, "a": layer_a.name, "b": layer_b.name}
                if report_bounds:
                    entry["a_bounds"] = list(a_bounds)
                    entry["b_bounds"] = list(b_bounds)
                overlaps.append(entry)

    if changed and not report_only:
        try:
            save(spr, filename)
        except ToolError as e:
            return f"Failed to sanitize animation: {e}"

    result = {
        "frames": {"start": frames.start + 1, "end": frames.stop},
        **sanitized,
        "analysis": {
            "total_layers": len(spr.layers),
            "layers_checked": len(targets),
            "total_cels": total_cels,
            "empty_frames": empty_frames,
            "overlaps": overlaps_total,
            "overlaps_total": overlaps_total,
            "overlaps_truncated": overlaps_total > len(overlaps),
            "inactive_layers": [l.name for l in targets if stats[l.name]["cel_count"] == 0],
        },
    }
    if include_stats:
        result["layer_stats"] = stats
    alerts = []
    if empty_frames:
        alerts.append("empty_frames_detected")
    if sanitized["out_of_range"]:
        alerts.append("cels_out_of_range")
    alerts += [f"full_canvas_cels:{l.name}" for l in targets if stats[l.name]["full_canvas_cels"]]
    if alerts:
        result["alerts"] = alerts
    if overlaps:
        result["overlap_samples"] = overlaps
    return json.dumps(result)
