"""Read and write .ase/.aseprite files the way LibreSprite does.

LibreSprite forked from Aseprite 1.1, so it supports a subset of today's
format. This module mirrors LibreSprite v1.3's src/app/file/ase_format.cpp:

- RGB (32 bpp), grayscale (16 bpp) and indexed (8 bpp) sprites.
- A flat list of image layers with flags, the 16 Aseprite 1.1 blend modes
  and opacity. LibreSprite has no layer groups: when it opens a file that
  has them it moves every nested layer to the top level, named
  "Group-Layer", and drops the groups. read() does the same.
- Cels with position and opacity, stored compressed or linked.
- Per-frame durations, frame tags (forward/reverse/pingpong) and one palette.
- Layer and cel user data (text and color), which is preserved.

write() emits only chunks LibreSprite understands, so it never shows an
"unsupported chunk" warning when opening our files. read() also accepts
newer Aseprite files: tilemap layers become empty placeholder layers and
slices, tilesets and color profiles are skipped, each with a warning.
"""
from __future__ import annotations

import os
import struct
import tempfile
import zlib
from dataclasses import dataclass, field

import numpy as np

RGB = "rgb"
GRAY = "gray"
INDEXED = "indexed"
COLOR_MODES = (RGB, GRAY, INDEXED)

_DEPTH_TO_MODE = {32: RGB, 16: GRAY, 8: INDEXED}
_MODE_TO_DEPTH = {RGB: 32, GRAY: 16, INDEXED: 8}
CHANNELS = {RGB: 4, GRAY: 2, INDEXED: 1}

# Layer flags (ASE_LAYER_FLAG_*).
FLAG_VISIBLE = 1
FLAG_EDITABLE = 2
FLAG_LOCK_MOVEMENT = 4
FLAG_BACKGROUND = 8
FLAG_PREFER_LINKED_CELS = 16

# Blend modes LibreSprite supports, indexed by their value in the file.
BLEND_MODES = (
    "normal", "multiply", "screen", "overlay", "darken", "lighten",
    "color_dodge", "color_burn", "hard_light", "soft_light", "difference",
    "exclusion", "hue", "saturation", "color", "luminosity",
)

# Tag animation directions LibreSprite supports.
DIRECTIONS = ("forward", "reverse", "pingpong")

_MAGIC = 0xA5E0
_FRAME_MAGIC = 0xF1FA
_HEADER_FLAG_LAYER_OPACITY = 1

_CHUNK_OLD_PALETTE_256 = 0x0004
_CHUNK_OLD_PALETTE_64 = 0x000B
_CHUNK_LAYER = 0x2004
_CHUNK_CEL = 0x2005
_CHUNK_CEL_EXTRA = 0x2006
_CHUNK_COLOR_PROFILE = 0x2007
_CHUNK_EXTERNAL_FILES = 0x2008
_CHUNK_MASK = 0x2016
_CHUNK_PATH = 0x2017
_CHUNK_TAGS = 0x2018
_CHUNK_PALETTE = 0x2019
_CHUNK_USER_DATA = 0x2020
_CHUNK_SLICE_OLD = 0x2021
_CHUNK_SLICE = 0x2022
_CHUNK_TILESET = 0x2023

_CEL_RAW = 0
_CEL_LINKED = 1
_CEL_COMPRESSED = 2
_CEL_COMPRESSED_TILEMAP = 3

_MAX_DIMENSION = 65535


class AseError(Exception):
    """The file is not a readable .ase/.aseprite file."""


@dataclass
class UserData:
    text: str = ""
    color: tuple[int, int, int, int] | None = None

    def is_empty(self) -> bool:
        return not self.text and (self.color is None or self.color[3] == 0)


@dataclass
class Cel:
    """Pixels placed on one layer in one frame.

    image is a (height, width, channels) uint8 array: RGBA for RGB sprites,
    (value, alpha) for grayscale sprites and a palette index for indexed
    sprites. x/y place its top-left corner in sprite coordinates.

    Linked cels are the same Cel object stored under several frames of one
    layer, so editing one edits all of them, as in LibreSprite.
    """
    image: np.ndarray
    x: int = 0
    y: int = 0
    opacity: int = 255
    user_data: UserData | None = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def copy(self) -> "Cel":
        user_data = UserData(self.user_data.text, self.user_data.color) if self.user_data else None
        return Cel(self.image.copy(), self.x, self.y, self.opacity, user_data)


@dataclass
class Layer:
    name: str
    flags: int = FLAG_VISIBLE | FLAG_EDITABLE
    blend_mode: int = 0
    opacity: int = 255
    cels: dict[int, Cel] = field(default_factory=dict)  # 0-based frame -> cel
    user_data: UserData | None = None

    @property
    def visible(self) -> bool:
        return bool(self.flags & FLAG_VISIBLE)

    @visible.setter
    def visible(self, value: bool) -> None:
        self.flags = self.flags | FLAG_VISIBLE if value else self.flags & ~FLAG_VISIBLE

    @property
    def is_background(self) -> bool:
        return bool(self.flags & FLAG_BACKGROUND)

    @is_background.setter
    def is_background(self, value: bool) -> None:
        if value:
            self.flags |= FLAG_BACKGROUND | FLAG_LOCK_MOVEMENT
        else:
            self.flags &= ~(FLAG_BACKGROUND | FLAG_LOCK_MOVEMENT)


@dataclass
class Tag:
    name: str
    from_frame: int  # 0-based, inclusive
    to_frame: int  # 0-based, inclusive
    direction: int = 0
    color: tuple[int, int, int] = (0, 0, 0)


@dataclass
class Sprite:
    width: int
    height: int
    color_mode: str = RGB
    palette: list[tuple[int, int, int, int]] = field(default_factory=list)
    transparent_index: int = 0
    durations: list[int] = field(default_factory=lambda: [100])  # ms per frame
    layers: list[Layer] = field(default_factory=list)  # bottom to top
    tags: list[Tag] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # from read()

    @property
    def frame_count(self) -> int:
        return len(self.durations)

    @property
    def channels(self) -> int:
        return CHANNELS[self.color_mode]


# ---------------------------------------------------------------- reading


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise AseError("Unexpected end of file")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def byte(self) -> int:
        return self.take(1)[0]

    def word(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def short(self) -> int:
        return struct.unpack("<h", self.take(2))[0]

    def dword(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def string(self) -> str:
        return self.take(self.word()).decode("utf-8", errors="replace")


def read(path: str) -> Sprite:
    """Load a .ase/.aseprite file into a Sprite."""
    with open(path, "rb") as f:
        data = f.read()
    return read_bytes(data)


def read_bytes(data: bytes) -> Sprite:
    r = _Reader(data)
    if len(data) < 128:
        raise AseError("File is too small to be an .ase file")
    r.dword()  # file size
    if r.word() != _MAGIC:
        raise AseError("Not an .ase/.aseprite file (bad magic number)")
    frame_count = r.word()
    width = r.word()
    height = r.word()
    depth = r.word()
    header_flags = r.dword()
    speed = r.word()
    r.dword()
    r.dword()
    transparent_index = r.byte()
    r.take(3)
    ncolors = r.word() or 256
    r.pos = 128

    if depth not in _DEPTH_TO_MODE:
        raise AseError(f"Unsupported color depth {depth}")
    if width == 0 or height == 0:
        raise AseError("Invalid .ase file: bad width/height")

    mode = _DEPTH_TO_MODE[depth]
    channels = CHANNELS[mode]
    spr = Sprite(width, height, mode, transparent_index=transparent_index)
    spr.durations = [speed or 100] * max(1, frame_count)
    palette = _default_palette(ncolors)
    have_new_palette = False

    # Every layer chunk in file order, groups included: cel chunks refer
    # to layers by this index. Entries are (layer, child_level, is_group).
    file_layers: list[tuple[Layer, int, bool]] = []
    last_with_user_data: Layer | Cel | None = None
    skipped: dict[str, int] = {}

    def skip(what: str) -> None:
        skipped[what] = skipped.get(what, 0) + 1

    for frame in range(frame_count):
        frame_start = r.pos
        frame_size = r.dword()
        magic = r.word()
        old_chunks = r.word()
        duration = r.word()
        r.take(2)
        new_chunks = r.dword()
        if magic != _FRAME_MAGIC:
            raise AseError(f"Frame {frame + 1} has a bad magic number")
        if duration > 0:
            spr.durations[frame] = duration
        chunks = new_chunks if old_chunks == 0xFFFF and new_chunks else old_chunks

        for _ in range(chunks):
            chunk_start = r.pos
            chunk_size = r.dword()
            chunk_type = r.word()
            chunk_end = chunk_start + chunk_size
            if chunk_size < 6 or chunk_end > len(data):
                raise AseError(f"Corrupt chunk in frame {frame + 1}")

            if chunk_type == _CHUNK_PALETTE:
                new_size = r.dword()
                first = r.dword()
                last = r.dword()
                r.take(8)
                if 0 < new_size <= 65536:
                    palette = (palette + _default_palette(new_size))[:new_size]
                for i in range(first, min(last, 65535) + 1):
                    entry_flags = r.word()
                    rgba = tuple(r.take(4))
                    if entry_flags & 1:
                        r.string()
                    if i < len(palette):
                        if frame == 0:
                            palette[i] = rgba
                        elif palette[i] != rgba:
                            skip("per-frame palette changes")
                have_new_palette = True

            elif chunk_type in (_CHUNK_OLD_PALETTE_256, _CHUNK_OLD_PALETTE_64, 0x0011):
                # Pre-1.2 palette chunks; ignored once a 0x2019 chunk was read.
                if not have_new_palette and frame == 0:
                    six_bit = chunk_type != _CHUNK_OLD_PALETTE_256
                    index = 0
                    for _p in range(r.word()):
                        index += r.byte()
                        for _c in range(r.byte() or 256):
                            rgb = r.take(3)
                            if six_bit:
                                rgb = bytes(_scale_6bit(v) for v in rgb)
                            if index >= len(palette):
                                palette.extend(_default_palette(index + 1 - len(palette)))
                            palette[index] = (rgb[0], rgb[1], rgb[2], 255)
                            index += 1

            elif chunk_type == _CHUNK_LAYER:
                flags = r.word()
                layer_type = r.word()
                child_level = r.word()
                r.word()
                r.word()
                blend_mode = r.word()
                opacity = r.byte()
                r.take(3)
                name = r.string()
                if not header_flags & _HEADER_FLAG_LAYER_OPACITY:
                    opacity = 255
                if blend_mode >= len(BLEND_MODES):
                    skip(f"blend modes newer than LibreSprite (layer '{name}' set to normal)")
                    blend_mode = 0
                if flags & FLAG_BACKGROUND:
                    blend_mode, opacity = 0, 255
                layer = Layer(name, flags & 0x1F, blend_mode, opacity)
                if layer_type == 2:
                    skip(f"tilemap layers (layer '{name}' left empty)")
                file_layers.append((layer, child_level, layer_type == 1))
                last_with_user_data = layer

            elif chunk_type == _CHUNK_CEL:
                layer_index = r.word()
                x = r.short()
                y = r.short()
                opacity = r.byte()
                cel_type = r.word()
                r.take(7)
                if layer_index >= len(file_layers) or file_layers[layer_index][2]:
                    skip("cels on missing or group layers")
                    r.pos = chunk_end
                    continue
                layer = file_layers[layer_index][0]
                cel = None
                if cel_type in (_CEL_RAW, _CEL_COMPRESSED):
                    w = r.word()
                    h = r.word()
                    if 0 < w <= _MAX_DIMENSION and 0 < h <= _MAX_DIMENSION:
                        expected = w * h * channels
                        payload = data[r.pos:chunk_end]
                        if cel_type == _CEL_COMPRESSED:
                            try:
                                payload = zlib.decompressobj().decompress(payload, expected)
                            except zlib.error:
                                skip("corrupt compressed cels")
                                payload = b""
                        if len(payload) < expected:
                            payload = payload + bytes(expected - len(payload))
                        pixels = np.frombuffer(payload[:expected], np.uint8).reshape(h, w, channels).copy()
                        cel = Cel(pixels, x, y, opacity)
                elif cel_type == _CEL_LINKED:
                    linked = layer.cels.get(r.word())
                    if linked is not None:
                        if (linked.x, linked.y, linked.opacity) == (x, y, opacity):
                            cel = linked
                        else:
                            cel = linked.copy()
                            cel.x, cel.y, cel.opacity = x, y, opacity
                elif cel_type == _CEL_COMPRESSED_TILEMAP:
                    skip("tilemap cels")
                if cel is not None:
                    layer.cels[frame] = cel
                    last_with_user_data = cel

            elif chunk_type == _CHUNK_TAGS:
                count = r.word()
                r.take(8)
                for _t in range(count):
                    from_frame = r.word()
                    to_frame = r.word()
                    direction = r.byte()
                    if direction >= len(DIRECTIONS):
                        direction = 0
                    r.take(8)
                    rgb = tuple(r.take(3))
                    r.byte()
                    spr.tags.append(Tag(r.string(), from_frame, to_frame, direction, rgb))
                # User data after a tags chunk belongs to the tags.
                last_with_user_data = None

            elif chunk_type == _CHUNK_USER_DATA:
                ud_flags = r.dword()
                ud = UserData()
                if ud_flags & 1:
                    ud.text = r.string()
                if ud_flags & 2:
                    ud.color = tuple(r.take(4))
                if last_with_user_data is not None and not ud.is_empty():
                    last_with_user_data.user_data = ud

            elif chunk_type in (_CHUNK_SLICE, _CHUNK_SLICE_OLD):
                skip("slices")
            elif chunk_type == _CHUNK_TILESET:
                skip("tilesets")
            elif chunk_type in (_CHUNK_COLOR_PROFILE, _CHUNK_CEL_EXTRA, _CHUNK_MASK,
                                _CHUNK_PATH, _CHUNK_EXTERNAL_FILES):
                pass
            else:
                skip(f"unknown chunk type 0x{chunk_type:04X}")

            r.pos = chunk_end

        r.pos = frame_start + frame_size

    if len(palette) > 256:
        skip(f"palette entries past 256 ({len(palette) - 256} dropped)")
    spr.palette = palette[:256] if palette else _default_palette(1)
    spr.layers = _flatten_groups(file_layers, spr)
    for what, count in skipped.items():
        spr.warnings.append(f"Ignored {what} ({count}x): LibreSprite does not support them")
    return spr


def _flatten_groups(file_layers, spr: Sprite) -> list[Layer]:
    """Keep image layers in stack order, naming nested ones like LibreSprite."""
    result = []
    stack: list[tuple[int, str]] = []  # (child_level, group name) of open groups
    had_groups = False
    for layer, level, is_group in file_layers:
        while stack and stack[-1][0] >= level:
            stack.pop()
        if is_group:
            had_groups = True
            stack.append((level, layer.name))
            continue
        if stack:
            layer.name = "-".join([name for _, name in stack] + [layer.name])
        result.append(layer)
    if had_groups:
        spr.warnings.append(
            "The file had layer groups, which LibreSprite does not support: nested layers "
            "were moved to the top level as 'Group-Layer', as LibreSprite does when opening it")
    return result


def _scale_6bit(v: int) -> int:
    return (v << 2) | (v >> 4)


def _default_palette(n: int) -> list[tuple[int, int, int, int]]:
    """Opaque black entries, which is how doc::Palette pads a resize."""
    return [(0, 0, 0, 255)] * max(0, n)


# ---------------------------------------------------------------- writing


class _Writer:
    def __init__(self):
        self.parts: list[bytes] = []

    def put(self, b: bytes) -> None:
        self.parts.append(b)

    def byte(self, v: int) -> None:
        self.put(struct.pack("<B", v & 0xFF))

    def word(self, v: int) -> None:
        self.put(struct.pack("<H", v & 0xFFFF))

    def short(self, v: int) -> None:
        self.put(struct.pack("<h", max(-32768, min(32767, v))))

    def dword(self, v: int) -> None:
        self.put(struct.pack("<I", v & 0xFFFFFFFF))

    def string(self, s: str) -> None:
        raw = s.encode("utf-8")[:65535]
        self.word(len(raw))
        self.put(raw)

    def getvalue(self) -> bytes:
        return b"".join(self.parts)


def _chunk(chunk_type: int, body: bytes) -> bytes:
    return struct.pack("<IH", len(body) + 6, chunk_type) + body


def _palette_chunk(palette) -> bytes:
    w = _Writer()
    w.dword(len(palette))
    w.dword(0)
    w.dword(len(palette) - 1)
    w.put(bytes(8))
    for rgba in palette:
        w.word(0)
        w.put(bytes(rgba))
    return _chunk(_CHUNK_PALETTE, w.getvalue())


def _old_palette_chunk(palette) -> bytes:
    w = _Writer()
    w.word(1)
    w.byte(0)
    w.byte(0 if len(palette) == 256 else len(palette))
    for r, g, b, _a in palette:
        w.put(bytes((r, g, b)))
    return _chunk(_CHUNK_OLD_PALETTE_256, w.getvalue())


def _user_data_chunk(ud: UserData) -> bytes:
    w = _Writer()
    flags = (1 if ud.text else 0) | (2 if ud.color and ud.color[3] else 0)
    w.dword(flags)
    if flags & 1:
        w.string(ud.text)
    if flags & 2:
        w.put(bytes(ud.color))
    return _chunk(_CHUNK_USER_DATA, w.getvalue())


def write_bytes(spr: Sprite) -> bytes:
    validate(spr)
    out = _Writer()
    palette = [tuple(int(c) for c in rgba) for rgba in spr.palette]

    frames = []
    for frame in range(spr.frame_count):
        chunks = []
        if frame == 0:
            chunks.append(_palette_chunk(palette))
            chunks.append(_old_palette_chunk(palette))
            for layer in spr.layers:
                w = _Writer()
                w.word(layer.flags)
                w.word(0)  # image layer
                w.word(0)  # child level
                w.word(0)
                w.word(0)
                w.word(0 if layer.is_background else layer.blend_mode)
                w.byte(255 if layer.is_background else layer.opacity)
                w.put(bytes(3))
                w.string(layer.name)
                chunks.append(_chunk(_CHUNK_LAYER, w.getvalue()))
                if layer.user_data and not layer.user_data.is_empty():
                    chunks.append(_user_data_chunk(layer.user_data))
            if spr.tags:
                w = _Writer()
                w.word(len(spr.tags))
                w.put(bytes(8))
                for tag in spr.tags:
                    w.word(tag.from_frame)
                    w.word(tag.to_frame)
                    w.byte(tag.direction)
                    w.put(bytes(8))
                    w.put(bytes(tag.color[:3]))
                    w.byte(0)
                    w.string(tag.name)
                chunks.append(_chunk(_CHUNK_TAGS, w.getvalue()))

        for index, layer in enumerate(spr.layers):
            cel = layer.cels.get(frame)
            if cel is None:
                continue
            w = _Writer()
            w.word(index)
            w.short(cel.x)
            w.short(cel.y)
            w.byte(cel.opacity)
            link = next((f for f in sorted(layer.cels) if f < frame and layer.cels[f] is cel), None)
            if link is not None:
                w.word(_CEL_LINKED)
                w.put(bytes(7))
                w.word(link)
            else:
                w.word(_CEL_COMPRESSED)
                w.put(bytes(7))
                w.word(cel.width)
                w.word(cel.height)
                w.put(zlib.compress(np.ascontiguousarray(cel.image, np.uint8).tobytes()))
            chunks.append(_chunk(_CHUNK_CEL, w.getvalue()))
            if link is None and cel.user_data and not cel.user_data.is_empty():
                chunks.append(_user_data_chunk(cel.user_data))

        body = b"".join(chunks)
        header = struct.pack("<IHHH2sI", 16 + len(body), _FRAME_MAGIC, min(len(chunks), 0xFFFF),
                             spr.durations[frame], bytes(2), len(chunks))
        frames.append(header + body)

    frames_bytes = b"".join(frames)
    header = _Writer()
    header.dword(128 + len(frames_bytes))
    header.word(_MAGIC)
    header.word(spr.frame_count)
    header.word(spr.width)
    header.word(spr.height)
    header.word(_MODE_TO_DEPTH[spr.color_mode])
    header.dword(_HEADER_FLAG_LAYER_OPACITY)
    header.word(spr.durations[0])
    header.dword(0)
    header.dword(0)
    header.byte(spr.transparent_index)
    header.put(bytes(3))
    header.word(len(palette))
    header.byte(1)  # pixel width
    header.byte(1)  # pixel height
    head = header.getvalue()
    out.put(head + bytes(128 - len(head)))
    out.put(frames_bytes)
    return out.getvalue()


def write(spr: Sprite, path: str) -> None:
    """Save atomically: a crash mid-write never leaves a truncated file."""
    data = write_bytes(spr)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".ase")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def validate(spr: Sprite) -> None:
    """Raise ValueError for a sprite LibreSprite could not load correctly."""
    if spr.color_mode not in COLOR_MODES:
        raise ValueError(f"Unknown color mode {spr.color_mode!r}")
    if not (1 <= spr.width <= _MAX_DIMENSION and 1 <= spr.height <= _MAX_DIMENSION):
        raise ValueError("Sprite width and height must be between 1 and 65535")
    if not 1 <= spr.frame_count <= 65535:
        raise ValueError("A sprite needs between 1 and 65535 frames")
    if not 1 <= len(spr.palette) <= 256:
        raise ValueError("The palette must have between 1 and 256 colors")
    if any(not 1 <= d <= 65535 for d in spr.durations):
        raise ValueError("Frame durations must be between 1 and 65535 ms")
    for i, layer in enumerate(spr.layers):
        if layer.is_background and i != 0:
            raise ValueError(f"Background layer '{layer.name}' must be the bottom layer")
        for frame, cel in layer.cels.items():
            if not 0 <= frame < spr.frame_count:
                raise ValueError(f"Layer '{layer.name}' has a cel in missing frame {frame + 1}")
            if cel.image.ndim != 3 or cel.image.shape[2] != spr.channels:
                raise ValueError(f"Cel on '{layer.name}' frame {frame + 1} has the wrong pixel format")
            if cel.width < 1 or cel.height < 1:
                raise ValueError(f"Cel on '{layer.name}' frame {frame + 1} is empty")
    for tag in spr.tags:
        if not 0 <= tag.from_frame <= tag.to_frame < spr.frame_count:
            raise ValueError(f"Tag '{tag.name}' covers frames outside the sprite")
