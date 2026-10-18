"""krita_* tools against a running Krita with the Puretty Bridge plugin.

These tests need Krita open (they create, export and close their own
temporary documents and leave your other documents alone); they are
skipped when the bridge does not answer. The client error handling is
tested without Krita.
"""
import json
import os

import pytest
from mcp.server.mcpserver import Image
from PIL import Image as PILImage

from conftest import run
from libresprite_mcp.tools import krita


def _bridge_up() -> bool:
    try:
        return bool(krita.request_sync("ping", timeout=5).get("pong"))
    except Exception:
        return False


needs_krita = pytest.mark.skipif(not _bridge_up(), reason="Krita with the Puretty Bridge plugin is not running")


def test_missing_bridge_file_gives_a_clear_message(monkeypatch, tmp_path):
    monkeypatch.setenv("KRITA_BRIDGE_FILE", str(tmp_path / "none.json"))
    out = run(krita.krita_status())
    assert out.startswith("Failed to reach Krita: Krita is not reachable. Start Krita")


def test_stale_bridge_file_gives_a_clear_message(monkeypatch, tmp_path):
    stale = tmp_path / "bridge.json"
    stale.write_text(json.dumps({"port": 9, "token": "x"}))  # nothing listens on port 9
    monkeypatch.setenv("KRITA_BRIDGE_FILE", str(stale))
    assert "Start Krita" in run(krita.krita_list_layers())


@pytest.fixture
def doc(tmp_path):
    """A fresh 32x24 Krita document, closed (unsaved) afterwards."""
    name = f"pytest-{os.getpid()}-{id(tmp_path)}"
    out = run(krita.krita_new_document(32, 24, name))
    assert out.startswith("Created Krita document"), out
    yield name
    run(krita.krita_close_document(name))


@needs_krita
def test_status_lists_documents(doc):
    status = json.loads(run(krita.krita_status()))
    assert status["krita_version"] and status["bridge_version"] == "1.0"
    assert any(d["name"] == doc and d["width"] == 32 and d["height"] == 24 for d in status["documents"])


@needs_krita
def test_add_layer_view_export_and_save(doc, tmp_path):
    src = tmp_path / "sprite.png"
    img = PILImage.new("RGBA", (8, 6), (0, 0, 0, 0))
    for x in range(8):
        img.putpixel((x, 0), (255, 0, 0, 255))
    img.putpixel((3, 3), (10, 200, 30, 128))
    img.save(src)

    out = run(krita.krita_add_image_layer(str(src), "imported", x=4, y=5, document=doc))
    assert out.startswith("Added layer 'imported' (8x6 at 4,5)"), out

    layers = json.loads(run(krita.krita_list_layers(doc)))
    names = [l["name"] for l in layers["layers"]]
    assert names == ["Layer 1", "imported"], names  # transparent base layer, new layer on top
    imported = layers["layers"][-1]
    assert imported["bounds"] == [4, 5, 8, 4]  # rows 0..3 have content

    view = run(krita.krita_view_canvas(document=doc, max_size=256))
    assert "canvas 32x24" in view[0] and "shown at 8x" in view[0]
    assert isinstance(view[1], Image)

    out_png = tmp_path / "export.png"
    out = run(krita.krita_export(str(out_png), document=doc))
    assert out.startswith("Exported 32x24"), out
    exported = PILImage.open(out_png).convert("RGBA")
    assert exported.getpixel((4, 5)) == (255, 0, 0, 255)
    assert exported.getpixel((7, 8)) == (10, 200, 30, 128)  # straight alpha survives
    assert exported.getpixel((0, 0))[3] == 0

    layer_png = tmp_path / "layer.png"
    out = run(krita.krita_export(str(layer_png), layer_name="imported", scale=3, document=doc))
    assert out.startswith("Exported 96x72"), out
    assert PILImage.open(layer_png).convert("RGBA").getpixel((4 * 3 + 1, 5 * 3 + 1)) == (255, 0, 0, 255)

    kra = tmp_path / "saved.kra"
    out = run(krita.krita_save_document(str(kra), document=doc))
    assert out == f"Saved {kra}" and kra.stat().st_size > 0


@needs_krita
def test_open_existing_image(tmp_path):
    src = tmp_path / "concept.png"
    PILImage.new("RGBA", (40, 30), (1, 2, 3, 255)).save(src)
    out = run(krita.krita_open(str(src)))
    assert out.startswith(f"Opened {src}") and "40x30" in out, out
    assert run(krita.krita_close_document(str(src))).startswith("Closed")


@needs_krita
def test_run_python(doc):
    out = run(krita.krita_run_python(
        "print('hello from krita')\nresult = (doc.width(), doc.height(), app.version() != '')"))
    assert "hello from krita" in out and "result = (32, 24, True)" in out
    err = run(krita.krita_run_python("raise ValueError('boom')"))
    assert err.startswith("Failed to run Python in Krita") and "ValueError: boom" in err


@needs_krita
def test_errors(doc, tmp_path):
    assert "No layer named 'nope'" in run(krita.krita_export(str(tmp_path / "x.png"), layer_name="nope",
                                                            document=doc))
    assert "No open document matches" in run(krita.krita_list_layers("no-such-doc"))
    assert run(krita.krita_add_image_layer(str(tmp_path / "missing.png"))).startswith("File ")
    assert run(krita.krita_new_document(0, 5)) == "Width and height must be > 0"
