"""Find and run LibreSprite: headless scripts, CLI exports and the editor.

The tools edit .ase files directly (see core/ase.py), so LibreSprite is
only needed for what it does itself: opening files in the editor, running
JavaScript in its scripting engine, its native sprite-sheet exporter and
confirming a file loads the way we expect.

LibreSprite is found through, in order: LIBRESPRITE_PATH (the executable
or its folder, also read from a .env file next to pyproject.toml), PATH,
the usual install folders, and a portable copy in ../tools/libresprite of
the Puretty-sprites checkout.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

import dotenv

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(_PACKAGE_DIR)  # libresprite-mcp/
_CHECKOUT_DIR = os.path.dirname(_PROJECT_DIR)  # Puretty-sprites/
dotenv.load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

_EXE = "libresprite.exe" if os.name == "nt" else "libresprite"

# Lines LibreSprite prints for every run; dropped from script output.
_NOISE = re.compile(r"^(Loading file |Using loader for format |Reading file |Invalid default: )")
_JS_ERROR = re.compile(r"^[A-Za-z]*Error: ")


def _candidates(env_var: str, extra_dirs: list[str]) -> list[str]:
    found = []
    configured = os.getenv(env_var, "").strip().strip('"')
    if configured:
        found.append(os.path.join(configured, _EXE) if os.path.isdir(configured) else configured)
    on_path = shutil.which("libresprite")
    if on_path:
        found.append(on_path)
    found += [os.path.join(d, _EXE) for d in extra_dirs]
    if sys.platform == "darwin":
        found.append("/Applications/LibreSprite.app/Contents/MacOS/libresprite")
    elif os.name != "nt":
        found += ["/usr/bin/libresprite", "/usr/local/bin/libresprite"]
    return found


def _install_dirs() -> list[str]:
    dirs = [os.path.join(_CHECKOUT_DIR, "tools", "libresprite")]
    if os.name == "nt":
        for base in (os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)"),
                     os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs")):
            if base:
                dirs.append(os.path.join(base, "LibreSprite"))
        dirs.append(os.path.join(os.path.expanduser("~"), "scoop", "apps", "libresprite", "current"))
    return dirs


def find_executable(for_scripts: bool = False) -> str | None:
    """Path to LibreSprite, or None.

    for_scripts prefers LIBRESPRITE_SCRIPT_PATH or a nightly build in
    ../tools/libresprite-nightly: LibreSprite 1.3 cannot run commands such
    as NewLayer from a headless script, nightly builds since 2026-10-14 can.
    """
    candidates = []
    if for_scripts:
        candidates += _candidates("LIBRESPRITE_SCRIPT_PATH",
                                  [os.path.join(_CHECKOUT_DIR, "tools", "libresprite-nightly")])
    candidates += _candidates("LIBRESPRITE_PATH", _install_dirs())
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def not_found_message() -> str:
    return ("LibreSprite was not found. Install it from https://libresprite.github.io "
            "(or https://github.com/LibreSprite/LibreSprite/releases) and set LIBRESPRITE_PATH "
            "to libresprite.exe or its folder, in the MCP server's environment or in "
            f"{os.path.join(_PROJECT_DIR, '.env')}.")


def clean_output(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not _NOISE.match(l)).strip()


def run(args: list[str], exe: str | None = None, timeout: float = 120) -> tuple[bool, str]:
    """Run LibreSprite with arguments; returns (ok, cleaned stdout+stderr)."""
    exe = exe or find_executable()
    if not exe:
        return False, not_found_message()
    try:
        result = subprocess.run([exe] + args, capture_output=True, text=True, timeout=timeout,
                                encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, f"LibreSprite did not finish within {timeout:g}s"
    except OSError as e:
        return False, f"Could not start LibreSprite ({exe}): {e}"
    output = clean_output((result.stdout or "") + "\n" + (result.stderr or ""))
    return result.returncode == 0, output


def version(exe: str | None = None) -> str | None:
    ok, out = run(["--version"], exe=exe, timeout=30)
    match = re.search(r"LibreSprite\s+(\S+(?:\s+\([0-9a-f]+\))?)", out)
    return match.group(1) if ok and match else None


def is_dev_build(ver: str | None) -> bool:
    return bool(ver and "dev" in ver)


def script_error(output: str) -> str | None:
    """The error LibreSprite printed for a failing script, if any.

    LibreSprite exits 0 even when a script throws; it prints
    "TypeError: message" followed by "    at <file>:<line>".
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if _JS_ERROR.match(line) and i + 1 < len(lines) and lines[i + 1].startswith("    at "):
            return line + " " + lines[i + 1].strip()
        if line.startswith("ERROR:"):
            return line[len("ERROR:"):].strip()
    return None


def run_script(source: str, filename: str | None = None, timeout: float = 120,
               settings: dict[str, str] | None = None) -> tuple[bool, str]:
    """Run JavaScript in headless LibreSprite, optionally on an open file.

    Returns (ok, output). ok is False when LibreSprite could not run or
    the script threw; output is what the script printed with console.log.
    """
    exe = find_executable(for_scripts=True)
    if not exe:
        return False, not_found_message()
    fd, path = tempfile.mkstemp(suffix=".js", prefix="libresprite_mcp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(source)
        args = ["--batch"]
        if filename:
            args.append(os.path.abspath(filename))
        for key, value in (settings or {}).items():
            args += ["--set", f"{key}={value}"]
        args += ["--script", path]
        ok, output = run(args, exe=exe, timeout=timeout)
    finally:
        os.remove(path)
    error = script_error(output)
    if error:
        return False, output
    return ok, output


def open_in_editor(filename: str) -> tuple[bool, str]:
    """Open a file in the LibreSprite editor without waiting for it."""
    exe = find_executable()
    if not exe:
        return False, not_found_message()
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([exe, os.path.abspath(filename)], cwd=os.path.dirname(exe), **kwargs)
    except OSError as e:
        return False, f"Could not start LibreSprite: {e}"
    return True, exe
