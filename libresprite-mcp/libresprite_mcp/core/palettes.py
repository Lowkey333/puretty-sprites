"""Well-known retro and pixel-art palettes, as (r, g, b, a) tuples."""
from __future__ import annotations

_HEX = {
    "gameboy": ["#0F380F", "#306230", "#8BAC0F", "#9BBC0F"],
    "monochrome": ["#000000", "#FFFFFF"],
    "grayscale_4": ["#000000", "#555555", "#AAAAAA", "#FFFFFF"],
    "cga": ["#000000", "#55FFFF", "#FF55FF", "#FFFFFF"],
    "pico8": [
        "#000000", "#1D2B53", "#7E2553", "#008751",
        "#AB5236", "#5F574F", "#C2C3C7", "#FFF1E8",
        "#FF004D", "#FFA300", "#FFEC27", "#00E436",
        "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA",
    ],
    "c64": [
        "#000000", "#FFFFFF", "#880000", "#AAFFEE",
        "#CC44CC", "#00CC55", "#0000AA", "#EEEE77",
        "#DD8855", "#664400", "#FF7777", "#333333",
        "#777777", "#AAFF66", "#0088FF", "#BBBBBB",
    ],
    "dawnbringer16": [
        "#140C1C", "#442434", "#30346D", "#4E4A4E",
        "#854C30", "#346524", "#D04648", "#757161",
        "#597DCE", "#D27D2C", "#8595A1", "#6DAA2C",
        "#D2AA99", "#6DC2CA", "#DAD45E", "#DEEED6",
    ],
    "dawnbringer32": [
        "#000000", "#222034", "#45283C", "#663931",
        "#8F563B", "#DF7126", "#D9A066", "#EEC39A",
        "#FBF236", "#99E550", "#6ABE30", "#37946E",
        "#4B692F", "#524B24", "#323C39", "#3F3F74",
        "#306082", "#5B6EE1", "#639BFF", "#5FCDE4",
        "#CBDBFC", "#FFFFFF", "#9BADB7", "#847E87",
        "#696A6A", "#595652", "#76428A", "#AC3232",
        "#D95763", "#D77BBA", "#8F974A", "#8A6F30",
    ],
    "sweetie16": [
        "#1A1C2C", "#5D275D", "#B13E53", "#EF7D57",
        "#FFCD75", "#A7F070", "#38B764", "#257179",
        "#29366F", "#3B5DC9", "#41A6F6", "#73EFF7",
        "#F4F4F4", "#94B0C2", "#566C86", "#333C57",
    ],
    "endesga32": [
        "#BE4A2F", "#D77643", "#EAD4AA", "#E4A672",
        "#B86F50", "#733E39", "#3E2731", "#A22633",
        "#E43B44", "#F77622", "#FEAE34", "#FEE761",
        "#63C74D", "#3E8948", "#265C42", "#193C3E",
        "#124E89", "#0099DB", "#2CE8F5", "#FFFFFF",
        "#C0CBDC", "#8B9BB4", "#5A6988", "#3A4466",
        "#262B44", "#181425", "#FF0044", "#68386C",
        "#B55088", "#F6757A", "#E8B796", "#C28569",
    ],
}

PRESET_HEX = _HEX

PRESETS: dict[str, list[tuple[int, int, int, int]]] = {
    name: [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16), 255) for h in colors]
    for name, colors in _HEX.items()
}
