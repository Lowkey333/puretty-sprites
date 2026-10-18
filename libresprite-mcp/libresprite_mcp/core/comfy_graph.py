"""Convert ComfyUI's UI-format workflows into the API format.

ComfyUI saves and ships workflows (including its bundled templates) in the
UI format: nodes with positions, a flat list of links and positional
"widgets_values". Its HTTP API only accepts the API format,
{node_id: {"class_type", "inputs"}}. The editor converts between them in
JavaScript; this does the same in Python, using the server's /object_info
to learn each node's input names and order:

- widget values are matched, in order, to the node's widget inputs; the
  extra values the editor stores after a seed ("fixed", "randomize") or an
  upload combo ("image") are skipped, a dynamic combo's chosen option is
  followed by that option's own inputs (sent as "combo.input"), and inputs
  newer than the saved workflow get the node's default,
- inputs fed by a link become [source_node_id, output_slot],
- notes and reroutes are dropped (reroutes are followed), muted nodes are
  removed, bypassed nodes are bridged input-to-output by type, and the
  editor-only PrimitiveNode is folded into the inputs it feeds.

Subgraphs are not supported (none of the 3D templates use them).
"""
from __future__ import annotations

_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
_UI_ONLY = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}
_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
_UPLOAD_VALUES = {"image", "video", "audio", "upload"}
_MUTED, _BYPASSED = 2, 4


class ConversionError(Exception):
    pass


def is_ui_format(graph) -> bool:
    return isinstance(graph, dict) and "nodes" in graph and "links" in graph


def _input_specs(info: dict) -> list[tuple[str, object, dict]]:
    specs = []
    for section in ("required", "optional"):
        for name, spec in (info.get("input", {}).get(section) or {}).items():
            kind = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            specs.append((name, kind, opts))
    return specs


def _widget_kinds(object_info: dict) -> set:
    """Input types the editor shows as widgets: the basic ones plus custom
    types no node outputs (e.g. LOAD_3D, the 3D viewer's state)."""
    outputs = set()
    for info in object_info.values():
        outputs.update(o for o in info.get("output", []) if isinstance(o, str))
    custom = set()
    for info in object_info.values():
        for _name, kind, _opts in _input_specs(info):
            if isinstance(kind, str) and kind not in outputs and "," not in kind and kind not in _WIDGET_TYPES:
                custom.add(kind)
    return _WIDGET_TYPES | custom


def ui_to_api(ui: dict, object_info: dict) -> tuple[dict, list[str]]:
    """Return (api_graph, warnings) for a UI-format workflow."""
    object_info = {k: v for k, v in (object_info or {}).items() if isinstance(v, dict)}
    if ui.get("definitions", {}).get("subgraphs"):
        raise ConversionError("workflows with subgraphs are not supported; unpack them in ComfyUI first")
    nodes = {n["id"]: n for n in ui.get("nodes", [])}
    links = {}
    for link in ui.get("links", []):
        if isinstance(link, dict):
            links[link["id"]] = (link["origin_id"], link["origin_slot"], link.get("type"))
        else:
            link_id, src, slot, _dst, _dst_slot, kind = link[:6]
            links[link_id] = (src, slot, kind)
    widget_kinds = _widget_kinds(object_info)
    warnings: list[str] = []

    def input_link(node: dict, name: str):
        for inp in node.get("inputs") or []:
            if inp.get("name") == name and inp.get("link") is not None:
                return inp["link"]
        return None

    def resolve(link_id, depth=0):
        """Follow reroutes, bypassed nodes and primitives to a real source."""
        if link_id is None or link_id not in links or depth > 64:
            return None
        src_id, slot, kind = links[link_id]
        src = nodes.get(src_id)
        if src is None or src.get("mode") == _MUTED:
            return None
        if src["type"] == "Reroute":
            inputs = src.get("inputs") or []
            return resolve(inputs[0].get("link") if inputs else None, depth + 1)
        if src["type"] == "PrimitiveNode":
            values = src.get("widgets_values") or []
            return ("value", values[0] if values else None)
        if src.get("mode") == _BYPASSED:
            for inp in src.get("inputs") or []:
                if inp.get("link") is not None and (inp.get("type") == kind or kind in (None, "*")):
                    return resolve(inp["link"], depth + 1)
            return None
        return ("link", src_id, slot)

    api: dict[str, dict] = {}
    for node_id, node in nodes.items():
        kind = node.get("type")
        if kind in _UI_ONLY or node.get("mode") in (_MUTED, _BYPASSED):
            continue
        info = object_info.get(kind)
        if info is None:
            raise ConversionError(f"node {node_id} has type {kind!r}, which this ComfyUI doesn't know "
                                  "(missing custom node?)")
        values = node.get("widgets_values")
        by_name = values if isinstance(values, dict) else None
        queue = list(values) if isinstance(values, list) else []
        inputs: dict = {}
        defaulted: list[str] = []

        def assign(specs, prefix=""):
            for name, input_kind, opts in specs:
                key = prefix + name
                is_widget = (isinstance(input_kind, list) or input_kind in widget_kinds
                             or input_kind == "COMFY_DYNAMICCOMBO_V3"
                             or "default" in opts) and not opts.get("forceInput")
                value, have_value = None, False
                if is_widget:
                    if by_name is not None:
                        have_value, value = key in by_name, by_name.get(key)
                    elif queue:
                        have_value, value = True, queue.pop(0)
                        if opts.get("control_after_generate") and queue and queue[0] in _CONTROL_VALUES:
                            queue.pop(0)
                        if any(opts.get(k) for k in ("image_upload", "video_upload", "audio_upload", "upload"))                                 and queue and queue[0] in _UPLOAD_VALUES:
                            queue.pop(0)
                    elif "default" in opts:
                        have_value, value = True, opts["default"]
                        defaulted.append(key)
                source = resolve(input_link(node, key))
                if source and source[0] == "link":
                    inputs[key] = [str(source[1]), source[2]]
                elif source and source[0] == "value":
                    inputs[key] = source[1]
                elif have_value:
                    inputs[key] = value
                if input_kind == "COMFY_DYNAMICCOMBO_V3":
                    chosen = inputs.get(key, opts.get("default"))
                    for option in opts.get("options", []):
                        if option.get("key") == chosen:
                            assign(_input_specs({"input": option.get("inputs", {})}), key + ".")

        assign(_input_specs(info))
        if queue:
            warnings.append(f"node {node_id} ({kind}): {len(queue)} unused widget value(s) {queue[:3]}")
        if defaulted:
            warnings.append(f"node {node_id} ({kind}): newer inputs set to their defaults: {', '.join(defaulted)}")
        api[str(node_id)] = {"class_type": kind, "inputs": inputs,
                             "_meta": {"title": node.get("title") or kind}}
    return api, warnings
