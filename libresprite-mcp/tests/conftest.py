"""Shared fixtures for the tool tests.

Each test calls a tool coroutine directly and checks the file it wrote.
Tests within a file form a sequence on a module-scoped sprite; files are
independent. Tests marked `libresprite` also run the real LibreSprite
(found like the server finds it) and are skipped when it is missing.
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import libresprite_mcp.tools  # noqa: E402,F401  (registers all tools)
from libresprite_mcp.core import libresprite  # noqa: E402
from libresprite_mcp.tools import canvas, drawing  # noqa: E402

ERROR_PREFIXES = ("Failed", "ERROR", "Invalid", "Script failed", "File ", "Unknown", "Layer not found")


def run(coro):
    """Execute an async tool call from a sync test."""
    return asyncio.run(coro)


def ok(result):
    """Assert a tool call did not return an error message."""
    text = result if isinstance(result, str) else str(result)
    assert not text.startswith(ERROR_PREFIXES), text
    return result


def fails(result, fragment: str = ""):
    """Assert a tool call returned an error mentioning fragment."""
    text = result if isinstance(result, str) else str(result)
    assert fragment.lower() in text.lower(), text
    assert not text.startswith(("Canvas created", "Layer '", "Pixels drawn")), text
    return result


@pytest.fixture(scope="session")
def base_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("ls-pytest"))


@pytest.fixture(scope="module")
def sprite(request, base_dir):
    """A fresh 32x32 sprite per test module with a painted 'body' layer."""
    name = request.module.__name__.rsplit(".", 1)[-1].removeprefix("test_")
    path = os.path.join(base_dir, f"{name}.ase")
    ok(run(canvas.create_canvas(32, 32, path)))
    ok(run(canvas.add_layer(path, "body")))
    ok(run(drawing.draw_rectangle_at(path, "body", 1, 8, 8, 16, 16, "#D04648", True)))
    return path


@pytest.fixture(scope="session")
def libresprite_exe():
    exe = libresprite.find_executable()
    if not exe:
        pytest.skip("LibreSprite not found (set LIBRESPRITE_PATH)")
    # LibreSprite keeps a small settings file per opened document; don't
    # leave one behind for every temporary test sprite.
    files_dir = os.path.join(os.getenv("APPDATA", ""), "LibreSprite", "files")
    before = set(os.listdir(files_dir)) if os.path.isdir(files_dir) else set()
    yield exe
    if os.path.isdir(files_dir):
        for name in set(os.listdir(files_dir)) - before:
            if "pytest-of-" in name:
                os.remove(os.path.join(files_dir, name))
