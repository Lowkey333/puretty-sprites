from typing import Any, Dict, List

import numpy as np

from ..core import raster
from ..core.colors import parse_hex_color
from ..core.sprite import (ToolError, active_target, alpha_channel, color_value, drawing_cel,
                           edit, paste_rgba)
from .. import mcp

# Coordinates are sprite-global. Drawing makes the target cel canvas-sized
# at (0, 0) first (baking in any cel offset), then replaces pixels; colors
# are converted to the sprite's color mode (nearest palette entry when
# indexed).


def _pixel_list(spr, pixels: List[Dict[str, Any]]) -> list[tuple[int, int, tuple]]:
    plotted = []
    for pixel in pixels:
        color = pixel.get("color", "#000000")
        if parse_hex_color(color) is None:
            raise ToolError(f"Invalid color value: {color}")
        plotted.append((int(pixel.get("x", 0)), int(pixel.get("y", 0)), color_value(spr, color)))
    return plotted


def _plot(img, plotted) -> None:
    h, w = img.shape[:2]
    for x, y, value in plotted:
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = value


@mcp.tool()
async def draw_pixels(filename: str, pixels: List[Dict[str, Any]]) -> str:
    """Draw pixels on the active cel with specified colors.

    The active cel is the layer/frame chosen with set_layer/set_frame
    (default: the topmost layer, frame 1).

    Args:
        filename: Name of the sprite file to modify
        pixels: List of pixel data, each containing:
            {"x": int, "y": int, "color": str}
            where color is a hex code like "#FF0000" or "#FF000080"
    """
    try:
        with edit(filename) as spr:
            layer, frame = active_target(spr, filename)
            _plot(drawing_cel(spr, layer, frame).image, _pixel_list(spr, pixels))
    except ToolError as e:
        return f"Failed to draw pixels: {e}"
    return f"Pixels drawn successfully in {filename}"


@mcp.tool()
async def draw_line(filename: str, x1: int, y1: int, x2: int, y2: int, color: str = "#000000", thickness: int = 1) -> str:
    """Draw a line on the active cel.

    Args:
        filename: Name of the sprite file to modify
        x1: Starting x coordinate
        y1: Starting y coordinate
        x2: Ending x coordinate
        y2: Ending y coordinate
        color: Hex color code (default: "#000000")
        thickness: Line thickness in pixels (default: 1)
    """
    if thickness < 1:
        return "Thickness must be >= 1"
    try:
        with edit(filename) as spr:
            layer, frame = active_target(spr, filename)
            raster.line(drawing_cel(spr, layer, frame).image, x1, y1, x2, y2, color_value(spr, color), thickness)
    except ToolError as e:
        return f"Failed to draw line: {e}"
    return f"Line drawn successfully in {filename}"


@mcp.tool()
async def draw_rectangle(filename: str, x: int, y: int, width: int, height: int, color: str = "#000000", fill: bool = False) -> str:
    """Draw a rectangle on the active cel.

    Args:
        filename: Name of the sprite file to modify
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Width of the rectangle in pixels (must be > 0)
        height: Height of the rectangle in pixels (must be > 0)
        color: Hex color code (default: "#000000")
        fill: Whether to fill the rectangle (default: False)
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        with edit(filename) as spr:
            layer, frame = active_target(spr, filename)
            raster.rect(drawing_cel(spr, layer, frame).image, x, y, width, height, color_value(spr, color), fill)
    except ToolError as e:
        return f"Failed to draw rectangle: {e}"
    return f"Rectangle drawn successfully in {filename}"


@mcp.tool()
async def fill_area(filename: str, x: int, y: int, color: str = "#000000") -> str:
    """Fill an area with color using the paint bucket (contiguous, exact match).

    Args:
        filename: Name of the sprite file to modify
        x: X coordinate to fill from
        y: Y coordinate to fill from
        color: Hex color code (default: "#000000")
    """
    try:
        with edit(filename) as spr:
            layer, frame = active_target(spr, filename)
            count = raster.flood_fill(drawing_cel(spr, layer, frame).image, x, y,
                                      color_value(spr, color), alpha_channel(spr))
            if count == 0:
                raise ToolError(f"({x},{y}) is outside the canvas")
    except ToolError as e:
        return f"Failed to fill area: {e}"
    return f"Area filled successfully in {filename} ({count} pixels)"


@mcp.tool()
async def draw_circle(filename: str, center_x: int, center_y: int, radius: int, color: str = "#000000", fill: bool = False) -> str:
    """Draw a circle on the active cel.

    Args:
        filename: Name of the sprite file to modify
        center_x: X coordinate of circle center
        center_y: Y coordinate of circle center
        radius: Radius of the circle in pixels
        color: Hex color code (default: "#000000")
        fill: Whether to fill the circle (default: False)
    """
    if radius < 0:
        return "Radius must be >= 0"
    try:
        with edit(filename) as spr:
            layer, frame = active_target(spr, filename)
            raster.ellipse(drawing_cel(spr, layer, frame).image, center_x - radius, center_y - radius,
                           center_x + radius, center_y + radius, color_value(spr, color), fill)
    except ToolError as e:
        return f"Failed to draw circle: {e}"
    return f"Circle drawn successfully in {filename}"


@mcp.tool()
async def draw_pixels_at(
    filename: str,
    layer_name: str,
    frame_index: int,
    pixels: List[Dict[str, Any]],
    create_if_missing: bool = True
) -> str:
    """Draw pixels on a specific layer/frame.

    Args:
        filename: Name of the sprite file to modify
        layer_name: Layer name to target
        frame_index: Frame index starting at 1
        pixels: List of pixel data with x/y/color, e.g.
            [{"x": 3, "y": 4, "color": "#D04648"}]
        create_if_missing: Create cel if it does not exist
    """
    try:
        with edit(filename) as spr:
            plotted = _pixel_list(spr, pixels)
            _plot(drawing_cel(spr, layer_name, frame_index, create_if_missing).image, plotted)
    except ToolError as e:
        return f"Failed to draw pixels: {e}"
    return f"Pixels drawn on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def draw_line_at(
    filename: str,
    layer_name: str,
    frame_index: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "#000000",
    thickness: int = 1,
    create_if_missing: bool = True
) -> str:
    """Draw a line on a specific layer/frame.

    Args:
        filename: Name of the sprite file to modify
        layer_name: Layer name to target
        frame_index: Frame index starting at 1
        x1: Starting x coordinate
        y1: Starting y coordinate
        x2: Ending x coordinate
        y2: Ending y coordinate
        color: Hex color code (default: "#000000")
        thickness: Line thickness in pixels (default: 1)
        create_if_missing: Create cel if it does not exist
    """
    if thickness < 1:
        return "Thickness must be >= 1"
    try:
        with edit(filename) as spr:
            value = color_value(spr, color)
            raster.line(drawing_cel(spr, layer_name, frame_index, create_if_missing).image,
                        x1, y1, x2, y2, value, thickness)
    except ToolError as e:
        return f"Failed to draw line: {e}"
    return f"Line drawn on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def draw_rectangle_at(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
    color: str = "#000000",
    fill: bool = False,
    create_if_missing: bool = True
) -> str:
    """Draw a rectangle on a specific layer/frame.

    Args:
        filename: Name of the sprite file to modify
        layer_name: Layer name to target
        frame_index: Frame index starting at 1
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Width of the rectangle in pixels (must be > 0)
        height: Height of the rectangle in pixels (must be > 0)
        color: Hex color code (default: "#000000")
        fill: Whether to fill the rectangle (default: False)
        create_if_missing: Create cel if it does not exist
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    try:
        with edit(filename) as spr:
            value = color_value(spr, color)
            raster.rect(drawing_cel(spr, layer_name, frame_index, create_if_missing).image,
                        x, y, width, height, value, fill)
    except ToolError as e:
        return f"Failed to draw rectangle: {e}"
    return f"Rectangle drawn on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def draw_circle_at(
    filename: str,
    layer_name: str,
    frame_index: int,
    center_x: int,
    center_y: int,
    radius: int,
    color: str = "#000000",
    fill: bool = False,
    create_if_missing: bool = True
) -> str:
    """Draw a circle on a specific layer/frame.

    Args:
        filename: Name of the sprite file to modify
        layer_name: Layer name to target
        frame_index: Frame index starting at 1
        center_x: X coordinate of circle center
        center_y: Y coordinate of circle center
        radius: Radius of the circle in pixels
        color: Hex color code (default: "#000000")
        fill: Whether to fill the circle (default: False)
        create_if_missing: Create cel if it does not exist
    """
    if radius < 0:
        return "Radius must be >= 0"
    try:
        with edit(filename) as spr:
            value = color_value(spr, color)
            raster.ellipse(drawing_cel(spr, layer_name, frame_index, create_if_missing).image,
                           center_x - radius, center_y - radius, center_x + radius, center_y + radius,
                           value, fill)
    except ToolError as e:
        return f"Failed to draw circle: {e}"
    return f"Circle drawn on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def fill_area_at(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    color: str = "#000000",
    create_if_missing: bool = True
) -> str:
    """Fill an area on a specific layer/frame (paint bucket, contiguous, exact match).

    Args:
        filename: Name of the sprite file to modify
        layer_name: Layer name to target
        frame_index: Frame index starting at 1
        x: X coordinate to fill from
        y: Y coordinate to fill from
        color: Hex color code (default: "#000000")
        create_if_missing: Create cel if it does not exist
    """
    try:
        with edit(filename) as spr:
            value = color_value(spr, color)
            count = raster.flood_fill(drawing_cel(spr, layer_name, frame_index, create_if_missing).image,
                                      x, y, value, alpha_channel(spr))
            if count == 0:
                raise ToolError(f"({x},{y}) is outside the canvas")
    except ToolError as e:
        return f"Failed to fill area: {e}"
    return f"Area filled on '{layer_name}' frame {frame_index} in {filename} ({count} pixels)"


@mcp.tool()
async def draw_polygon(
    filename: str,
    layer_name: str,
    frame_index: int,
    points: List[Dict[str, int]],
    color: str = "#000000",
    fill: bool = False,
    create_if_missing: bool = True
) -> str:
    """Draw a polygon on a specific layer/frame.

    Args:
        filename: Name of the sprite file to modify
        layer_name: Layer to draw on
        frame_index: Frame index starting at 1
        points: At least 3 vertices, e.g. [{"x": 1, "y": 1}, {"x": 8, "y": 1}, {"x": 4, "y": 7}]
        color: Hex color code (default: "#000000")
        fill: Fill the polygon as well as outlining it (default: False)
        create_if_missing: Create cel if it does not exist
    """
    if len(points) < 3:
        return "Polygon requires at least 3 points"
    try:
        pts = [(int(p["x"]), int(p["y"])) for p in points]
    except (KeyError, TypeError, ValueError):
        return "Each point needs integer x and y"
    try:
        with edit(filename) as spr:
            value = color_value(spr, color)
            raster.polygon(drawing_cel(spr, layer_name, frame_index, create_if_missing).image, pts, value, fill)
    except ToolError as e:
        return f"Failed to draw polygon: {e}"
    return f"Polygon drawn on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def draw_path(
    filename: str,
    layer_name: str,
    frame_index: int,
    points: List[Dict[str, int]],
    color: str = "#000000",
    thickness: int = 1,
    create_if_missing: bool = True
) -> str:
    """Draw a path (open polyline) through a list of points on a specific layer/frame.

    Args:
        filename: Name of the sprite file to modify
        layer_name: Layer to draw on
        frame_index: Frame index starting at 1
        points: At least 2 points, e.g. [{"x": 0, "y": 0}, {"x": 5, "y": 3}]
        color: Hex color code (default: "#000000")
        thickness: Stroke thickness in pixels (default: 1)
        create_if_missing: Create cel if it does not exist
    """
    if len(points) < 2:
        return "Path requires at least 2 points"
    if thickness < 1:
        return "Thickness must be >= 1"
    try:
        pts = [(int(p["x"]), int(p["y"])) for p in points]
    except (KeyError, TypeError, ValueError):
        return "Each point needs integer x and y"
    try:
        with edit(filename) as spr:
            value = color_value(spr, color)
            raster.polyline(drawing_cel(spr, layer_name, frame_index, create_if_missing).image, pts, value, thickness)
    except ToolError as e:
        return f"Failed to draw path: {e}"
    return f"Path drawn on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def apply_gradient_rect(
    filename: str,
    layer_name: str,
    frame_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
    color_start: str,
    color_end: str,
    horizontal: bool = True,
    create_if_missing: bool = True
) -> str:
    """Apply a smooth linear gradient fill to a rectangle.

    Every step is a new color; for palette-friendly blends between two
    colors use apply_dither_gradient instead.

    Args:
        filename: Name of the sprite file to modify
        layer_name: Layer to draw on
        frame_index: Frame index starting at 1
        x: Left edge of the rectangle
        y: Top edge of the rectangle
        width: Rectangle width
        height: Rectangle height
        color_start: Hex color at the left (or top) edge
        color_end: Hex color at the right (or bottom) edge
        horizontal: Run left-to-right (default) instead of top-to-bottom
        create_if_missing: Create cel if it does not exist
    """
    if width <= 0 or height <= 0:
        return "Width and height must be > 0"
    start = parse_hex_color(color_start)
    if start is None:
        return f"Invalid color_start value: {color_start}"
    end = parse_hex_color(color_end)
    if end is None:
        return f"Invalid color_end value: {color_end}"
    span = width if horizontal else height
    t = np.arange(span) / (span - 1) if span > 1 else np.zeros(1)
    ramp = np.floor(np.array(start) + (np.array(end) - np.array(start)) * t[:, None] + 0.5).astype(np.uint8)
    rgba = np.broadcast_to(ramp[None, :, :] if horizontal else ramp[:, None, :], (height, width, 4))
    try:
        with edit(filename) as spr:
            img = drawing_cel(spr, layer_name, frame_index, create_if_missing).image
            paste_rgba(spr, img, rgba, x, y)
    except ToolError as e:
        return f"Failed to apply gradient: {e}"
    return f"Gradient applied on '{layer_name}' frame {frame_index} in {filename}"


@mcp.tool()
async def draw_ellipse_at(
    filename: str,
    layer_name: str,
    frame_index: int,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    color: str = "#000000",
    fill: bool = False,
    create_if_missing: bool = True
) -> str:
    """Draw an ellipse on a specific layer/frame.

    Args:
        filename: Sprite file to modify
        layer_name: Layer to draw on
        frame_index: Frame index starting at 1
        center_x: Ellipse center x
        center_y: Ellipse center y
        radius_x: Horizontal radius in pixels
        radius_y: Vertical radius in pixels
        color: Hex color code (default "#000000")
        fill: Fill the ellipse instead of outlining it
        create_if_missing: Create the cel if it does not exist
    """
    if radius_x <= 0 or radius_y <= 0:
        return "radius_x and radius_y must be > 0"
    try:
        with edit(filename) as spr:
            value = color_value(spr, color)
            raster.ellipse(drawing_cel(spr, layer_name, frame_index, create_if_missing).image,
                           center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y,
                           value, fill)
    except ToolError as e:
        return f"Failed to draw ellipse: {e}"
    return f"Ellipse drawn on '{layer_name}' frame {frame_index} in {filename}"
