"""Install the Puretty Bridge plugin (and optionally Krita AI Diffusion) into Krita.

    python install.py                      # Puretty Bridge only
    python install.py --with-ai-diffusion  # also the latest Krita AI Diffusion release
    python install.py --uninstall          # remove Puretty Bridge (add --with-ai-diffusion to remove both)

Plugins are copied into Krita's pykrita folder (%APPDATA%\\krita\\pykrita on
Windows) and enabled in kritarc (%LOCALAPPDATA%\\kritarc on Windows) by
setting enable_<module>=true in its [python] section. kritarc is backed up
first and every other setting is left byte-for-byte as it was. Close Krita
before running this: Krita rewrites kritarc when it quits.
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
AI_DIFFUSION_REPO = "Acly/krita-ai-diffusion"


def krita_data_dir() -> str:
    if os.name == "nt":
        return os.path.join(os.environ["APPDATA"], "krita")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/krita")
    return os.path.expanduser("~/.local/share/krita")


def kritarc_path() -> str:
    if os.name == "nt":
        return os.path.join(os.environ["LOCALAPPDATA"], "kritarc")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Preferences/kritarc")
    return os.path.expanduser("~/.config/kritarc")


def krita_running() -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq krita.exe", "/NH"],
                             capture_output=True, text=True).stdout
        return "krita.exe" in out.lower()
    return subprocess.run(["pgrep", "-x", "krita"], capture_output=True).returncode == 0


def set_python_flags(path: str, flags: dict) -> str | None:
    """Set (True/False) or remove (None) enable_<module> keys in kritarc's [python] section.

    Edits lines in place so every other setting, the section order and the
    file's line endings stay as they were. Returns the backup path.
    """
    data = b""
    backup = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, backup)
    newline = "\r\n" if b"\r\n" in data else ("\n" if data else os.linesep)
    text = data.decode("utf-8")
    trailing = text.endswith(newline)
    lines = text.split(newline)
    if trailing:
        lines.pop()

    wanted = {f"enable_{module}": value for module, value in flags.items()}
    try:
        start = lines.index("[python]")
    except ValueError:
        start = None

    if start is None:
        additions = [f"{k}={'true' if v else 'false'}" for k, v in wanted.items() if v is not None]
        if additions:
            if lines and lines[-1].strip():
                lines.append("")
            lines += ["[python]"] + additions
    else:
        end = start + 1
        while end < len(lines) and not lines[end].startswith("["):
            end += 1
        section = []
        handled = set()
        for line in lines[start + 1:end]:
            key = line.split("=", 1)[0]
            if key in wanted:
                handled.add(key)
                if wanted[key] is not None:
                    section.append(f"{key}={'true' if wanted[key] else 'false'}")
            else:
                section.append(line)
        new_keys = [f"{k}={'true' if v else 'false'}" for k, v in wanted.items()
                    if k not in handled and v is not None]
        insert_at = len(section)
        while insert_at > 0 and not section[insert_at - 1].strip():
            insert_at -= 1  # keep the blank line that separates sections
        section[insert_at:insert_at] = new_keys
        lines[start + 1:end] = section

    out = newline.join(lines) + (newline if trailing or start is None else "")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    return backup


def copy_plugin(pykrita: str) -> None:
    os.makedirs(pykrita, exist_ok=True)
    target = os.path.join(pykrita, "puretty_bridge")
    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.copytree(os.path.join(HERE, "puretty_bridge"), target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(os.path.join(HERE, "puretty_bridge.desktop"), pykrita)
    print(f"Installed Puretty Bridge into {pykrita}")


def install_ai_diffusion(pykrita: str, version: str | None) -> str:
    api = f"https://api.github.com/repos/{AI_DIFFUSION_REPO}/releases/"
    api += f"tags/v{version.lstrip('v')}" if version else "latest"
    with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "puretty-sprites"})) as r:
        release = json.load(r)
    asset = next(a for a in release["assets"] if a["name"].endswith(".zip"))
    print(f"Downloading {asset['name']} ...")
    with urllib.request.urlopen(asset["browser_download_url"]) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    members = [m for m in archive.namelist()
               if (m.startswith("ai_diffusion/") or m == "ai_diffusion.desktop")
               and not os.path.isabs(m) and ".." not in m.split("/")]
    if "ai_diffusion.desktop" not in members:
        raise SystemExit("Unexpected archive layout: no ai_diffusion.desktop")
    target = os.path.join(pykrita, "ai_diffusion")
    if os.path.isdir(target):
        shutil.rmtree(target)
    for member in members:
        archive.extract(member, pykrita)
    print(f"Installed Krita AI Diffusion {release['tag_name']} into {pykrita}")
    return release["tag_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-ai-diffusion", action="store_true", help="also install/remove Krita AI Diffusion")
    parser.add_argument("--ai-diffusion-version", help="release to install, e.g. 1.53.0 (default: latest)")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()

    if krita_running():
        raise SystemExit("Krita is running. Close it first: it rewrites kritarc when it quits.")

    pykrita = os.path.join(krita_data_dir(), "pykrita")
    flags = {}
    if args.uninstall:
        for name in ["puretty_bridge"] + (["ai_diffusion"] if args.with_ai_diffusion else []):
            shutil.rmtree(os.path.join(pykrita, name), ignore_errors=True)
            desktop = os.path.join(pykrita, f"{name}.desktop")
            if os.path.exists(desktop):
                os.remove(desktop)
            flags[name] = None
            print(f"Removed {name} from {pykrita}")
    else:
        copy_plugin(pykrita)
        flags["puretty_bridge"] = True
        if args.with_ai_diffusion:
            install_ai_diffusion(pykrita, args.ai_diffusion_version)
            flags["ai_diffusion"] = True

    backup = set_python_flags(kritarc_path(), flags)
    print(f"Updated [python] in {kritarc_path()}" + (f" (backup: {backup})" if backup else ""))


if __name__ == "__main__":
    main()
