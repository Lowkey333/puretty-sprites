"""comfyui_* tools against an in-process fake of ComfyUI's HTTP API.

The fake implements the endpoints the tools use (/system_stats, /queue,
/object_info/<node>, /prompt, /history/<id>, /view, /upload/image) and
records what it receives, so the tests check the graphs the tools build,
the polling and downloading, and how errors are reported.
"""
import io
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from mcp.server.mcpserver import Image
from PIL import Image as PILImage

from conftest import run
from libresprite_mcp.tools import comfyui

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKFLOWS = os.path.join(REPO, "integrations", "comfyui", "workflows")


def _png(color=(200, 40, 40, 255), size=(64, 64)) -> bytes:
    out = io.BytesIO()
    PILImage.new("RGBA", size, color).save(out, "PNG")
    return out.getvalue()


class FakeComfy(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type="application/json"):
        data = json.dumps(body).encode() if content_type == "application/json" else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        s = self.state
        if url.path == "/system_stats":
            self._send(200, {"system": {"comfyui_version": "0.9.9", "python_version": "3.12.7 (main)"},
                             "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 5060 Ti", "type": "cuda",
                                          "vram_total": 16 * 2**30, "vram_free": 12 * 2**30}]})
        elif url.path == "/queue":
            self._send(200, {"queue_running": [], "queue_pending": [[1]]})
        elif url.path == "/object_info/CheckpointLoaderSimple":
            # old combo format: [[options], {...}]
            self._send(200, {"CheckpointLoaderSimple": {"input": {"required": {
                "ckpt_name": [s["checkpoints"], {}]}}}})
        elif url.path == "/object_info/LoraLoader":
            # new combo format: ["COMBO", {"options": [...]}]
            self._send(200, {"LoraLoader": {"input": {"required": {
                "lora_name": ["COMBO", {"options": ["pixel-art-xl.safetensors", "other_sd15.safetensors"]}]}}}})
        elif url.path == "/object_info/DWPreprocessor" and s.get("dwpose", True):
            self._send(200, {"DWPreprocessor": {"input": {"required": {"image": ["IMAGE"]}}}})
        elif url.path.startswith("/object_info/"):
            self._send(200, {})
        elif url.path.startswith("/history/"):
            pid = url.path.rsplit("/", 1)[-1]
            s["history_calls"] = s.get("history_calls", 0) + 1
            if s["history_calls"] % 2 == 1:  # first poll: still running
                self._send(200, {})
            else:
                self._send(200, {pid: s["entries"][pid]})
        elif url.path == "/view":
            name = q["filename"][0]
            s["views"].append((name, q.get("subfolder", [""])[0], q.get("type", [""])[0]))
            if name.endswith(".json"):  # SavePoseKpsAsJsonFile output
                return self._send(200, s["pose"]) if "pose" in s else self._send(404, {"error": "missing"})
            body = b"glTF-fake-mesh" if name.endswith(".glb") else _png()
            self._send(200, body, "application/octet-stream")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        s = self.state
        if self.path == "/prompt":
            graph = json.loads(body)["prompt"]
            s["prompts"].append(graph)
            classes = {n["class_type"] for n in graph.values()}
            if "BadNode" in classes:
                return self._send(400, {
                    "error": {"type": "prompt_outputs_failed_validation",
                              "message": "Prompt outputs failed validation", "details": ""},
                    "node_errors": {"99": {"class_type": "BadNode", "errors": [
                        {"message": "Value not in list", "details": "ckpt_name: 'nope' not in []"}]}}})
            pid = f"p{len(s['prompts'])}"
            outputs, status = {}, {"status_str": "success", "completed": True, "messages": []}
            for node_id, node in graph.items():
                if node["class_type"] == "SaveImage":
                    batch = next((n["inputs"].get("batch_size", 1) for n in graph.values()
                                  if n["class_type"] == "EmptyLatentImage"), 1)
                    prefix = node["inputs"]["filename_prefix"]
                    outputs[node_id] = {"images": [{"filename": f"{prefix}_{i:05d}_.png", "subfolder": "",
                                                    "type": "output"} for i in range(1, batch + 1)]}
                elif node["class_type"] == "SaveGLB":
                    outputs[node_id] = {"3d": [{"filename": "mesh.glb", "subfolder": "3d", "type": "output"}]}
                elif node["class_type"] == "FailNode":
                    status = {"status_str": "error", "completed": False, "messages": [
                        ["execution_error", {"node_type": "FailNode", "exception_message": "CUDA out of memory"}]]}
            s["entries"][pid] = {"outputs": outputs, "status": status}
            self._send(200, {"prompt_id": pid, "number": len(s["prompts"]), "node_errors": {}})
        elif self.path == "/upload/image":
            name = re.search(rb'filename="([^"]+)"', body).group(1).decode()
            s["uploads"].append(name)
            self._send(200, {"name": name, "subfolder": "", "type": "input"})
        else:
            self._send(404, {"error": "not found"})


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def fake(server, monkeypatch):
    FakeComfy.state = {"prompts": [], "uploads": [], "views": [], "entries": {},
                       "checkpoints": ["v1-5-pruned-emaonly.safetensors", "sd_xl_base_1.0.safetensors"]}
    monkeypatch.setenv("COMFYUI_URL", server)
    return FakeComfy.state


def test_status(fake):
    out = run(comfyui.comfyui_status())
    assert "RTX 5060 Ti" in out and "12.0 GB free of 16.0 GB" in out
    assert "0 running, 1 pending" in out and "0.9.9" in out


def test_list_models_reads_old_and_new_combo_formats(fake):
    models = json.loads(run(comfyui.comfyui_list_models()))
    assert models["checkpoints"] == fake["checkpoints"]
    assert models["loras"] == ["pixel-art-xl.safetensors", "other_sd15.safetensors"]


def test_generate_with_pixel_art_lora(fake, tmp_path):
    result = run(comfyui.comfyui_generate("a slime monster", lora="pixel-art-xl.safetensors",
                                          lora_strength=0.8, seed=123, batch_size=2,
                                          output_dir=str(tmp_path)))
    text = result[0]
    assert text.startswith("Generated 2 image(s) with sd_xl_base_1.0.safetensors"), text
    assert "seed 123, 1024x1024" in text  # SDXL picked for an XL LoRA, SDXL default size
    assert isinstance(result[1], Image)
    graph = fake["prompts"][-1]
    assert graph["12"]["class_type"] == "LoraLoader"
    assert graph["12"]["inputs"]["lora_name"] == "pixel-art-xl.safetensors"
    assert graph["12"]["inputs"]["strength_model"] == 0.8
    assert graph["3"]["inputs"]["model"] == ["12", 0]
    assert graph["6"]["inputs"]["clip"] == ["12", 1] and graph["7"]["inputs"]["clip"] == ["12", 1]
    assert graph["6"]["inputs"]["text"].startswith("pixel art, a slime monster")
    assert "anti-aliasing" in graph["7"]["inputs"]["text"]
    assert graph["5"]["inputs"]["batch_size"] == 2 and graph["3"]["inputs"]["seed"] == 123
    saved = sorted(os.listdir(tmp_path))
    assert saved == ["puretty_00001_.png", "puretty_00002_.png"]
    assert PILImage.open(tmp_path / saved[0]).size == (64, 64)
    # a second run does not overwrite earlier results
    run(comfyui.comfyui_generate("a slime monster", seed=5, output_dir=str(tmp_path), preview=False))
    assert "puretty_00001__1.png" in os.listdir(tmp_path)


def test_generate_defaults_without_lora(fake, tmp_path):
    result = run(comfyui.comfyui_generate("a tree", pixel_art=False, output_dir=str(tmp_path), preview=False))
    assert len(result) == 1 and "512x512" in result[0]  # first checkpoint is SD 1.5
    graph = fake["prompts"][-1]
    assert "12" not in graph and graph["3"]["inputs"]["model"] == ["4", 0]
    assert graph["6"]["inputs"]["text"] == "a tree"
    assert 0 <= graph["3"]["inputs"]["seed"] < 2**32


def test_generate_checkpoint_matching(fake, tmp_path):
    out = run(comfyui.comfyui_generate("x", checkpoint="sd_xl", output_dir=str(tmp_path), preview=False))
    assert "with sd_xl_base_1.0.safetensors" in out[0]
    out = run(comfyui.comfyui_generate("x", checkpoint="missing", output_dir=str(tmp_path)))
    assert out[0].startswith("Failed to generate: Checkpoint 'missing' not found")


def test_img2img_uploads_and_encodes(fake, tmp_path):
    src = tmp_path / "sketch.png"
    src.write_bytes(_png((10, 200, 10, 255), (48, 48)))
    result = run(comfyui.comfyui_img2img(str(src), "a goblin", denoise=0.4, seed=7,
                                         output_dir=str(tmp_path / "out")))
    assert result[0].startswith("img2img from sketch.png"), result[0]
    assert fake["uploads"] == ["sketch.png"]
    graph = fake["prompts"][-1]
    assert graph["10"] == {"class_type": "LoadImage", "inputs": {"image": "sketch.png"}}
    assert graph["11"]["inputs"]["pixels"] == ["10", 0]
    assert graph["3"]["inputs"]["latent_image"] == ["11", 0] and graph["3"]["inputs"]["denoise"] == 0.4
    assert "5" not in graph
    assert os.listdir(tmp_path / "out") == ["puretty_i2i_00001_.png"]


def test_run_workflow_overrides_images_and_seed(fake, tmp_path):
    src = tmp_path / "concept.png"
    src.write_bytes(_png())
    wf = os.path.join(WORKFLOWS, "img2img_sdxl.json")
    result = run(comfyui.comfyui_run_workflow(
        wf, overrides={"6.text": "pixel art, a wizard", "3.steps": 30},
        images={"10.image": str(src)}, randomize_seed=True, output_dir=str(tmp_path / "wf")))
    assert result[0].startswith("Ran img2img_sdxl.json"), result[0]
    graph = fake["prompts"][-1]
    assert graph["6"]["inputs"]["text"] == "pixel art, a wizard" and graph["3"]["inputs"]["steps"] == 30
    assert graph["10"]["inputs"]["image"] == "concept.png"
    assert graph["3"]["inputs"]["seed"] != 42  # the template's seed was replaced
    assert len(os.listdir(tmp_path / "wf")) == 1


def test_run_workflow_downloads_3d_outputs(fake, tmp_path):
    wf = tmp_path / "mesh.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
                              "2": {"class_type": "SaveGLB", "inputs": {"mesh": ["1", 0]}}}))
    result = run(comfyui.comfyui_run_workflow(str(wf), output_dir=str(tmp_path / "out")))
    assert "mesh.glb" in result[0] and len(result) == 1  # no image preview for a mesh
    assert (tmp_path / "out" / "mesh.glb").read_bytes() == b"glTF-fake-mesh"
    assert ("mesh.glb", "3d", "output") in fake["views"]


def test_run_workflow_converts_ui_format_and_rejects_bad_targets(fake, tmp_path):
    # UI-format graphs are converted with /object_info; an unknown node type
    # (the fake server knows none) is reported instead of being sent.
    ui = tmp_path / "ui.json"
    ui.write_text(json.dumps({"nodes": [{"id": 1, "type": "Mystery", "mode": 0}], "links": []}))
    assert "doesn't know" in run(comfyui.comfyui_run_workflow(str(ui)))[0]
    wf = os.path.join(WORKFLOWS, "txt2img_sd15.json")
    out = run(comfyui.comfyui_run_workflow(wf, overrides={"77.text": "x"}))[0]
    assert out.startswith("Failed to run workflow: Unknown override target '77.text'")
    assert run(comfyui.comfyui_run_workflow(str(tmp_path / "nope.json")))[0].startswith("File ")


def test_node_errors_are_reported(fake, tmp_path):
    wf = tmp_path / "bad.json"
    wf.write_text(json.dumps({"99": {"class_type": "BadNode", "inputs": {}}}))
    out = run(comfyui.comfyui_run_workflow(str(wf)))[0]
    assert out.startswith("Failed to run workflow: ComfyUI rejected the workflow")
    assert "node 99 (BadNode): Value not in list" in out


def test_execution_errors_are_reported(fake, tmp_path):
    wf = tmp_path / "fail.json"
    wf.write_text(json.dumps({"1": {"class_type": "FailNode", "inputs": {}}}))
    out = run(comfyui.comfyui_run_workflow(str(wf)))[0]
    assert out == "Failed to run workflow: Workflow failed in ComfyUI: FailNode: CUDA out of memory"


def test_validation_messages(fake, tmp_path):
    assert run(comfyui.comfyui_generate(" "))[0] == "Prompt cannot be empty"
    assert "multiples of 8" in run(comfyui.comfyui_generate("x", width=100))[0]
    assert run(comfyui.comfyui_img2img(str(tmp_path / "none.png"), "x"))[0].startswith("File ")
    assert "denoise" in run(comfyui.comfyui_img2img(__file__, "x", denoise=0))[0]


def _pose_json(found: dict, size=(96, 128), normalized=False) -> list:
    flat = []
    for name in comfyui.POSE_NAMES:
        x, y = found.get(name, (0, 0))
        if normalized:
            x, y = x / size[0], y / size[1]
        flat += [x, y, 1.0 if name in found else 0.0]
    return [{"people": [{"pose_keypoints_2d": flat}], "canvas_width": size[0], "canvas_height": size[1]}]


def test_detect_pose(fake, tmp_path):
    src = tmp_path / "hero.png"
    src.write_bytes(_png(size=(96, 128)))
    found = {name: (10 + 4 * i, 5 + 6 * i) for i, name in enumerate(comfyui.POSE_NAMES) if "ear" not in name}
    fake["pose"] = _pose_json(found)
    result = run(comfyui.comfyui_detect_pose(str(src), output_dir=str(tmp_path / "out")))
    text = result[0]
    assert text.startswith("DWPose found 16 of 18 body keypoints in hero.png"), text
    assert "  neck: x=14, y=11" in text and "Not found: right ear, left ear" in text
    assert isinstance(result[1], Image)
    graph = fake["prompts"][-1]
    assert graph["2"]["class_type"] == "DWPreprocessor"
    assert graph["2"]["inputs"]["bbox_detector"] == "None" and graph["2"]["inputs"]["image"] == ["1", 0]
    assert graph["3"]["inputs"]["pose_kps"] == ["2", 1]
    prefix = graph["3"]["inputs"]["filename_prefix"]
    assert (f"{prefix}_00001.json", "", "output") in fake["views"]
    with open(tmp_path / "out" / "hero_pose.json", encoding="utf-8") as f:
        assert json.load(f) == fake["pose"]
    assert (tmp_path / "out" / f"{prefix}_skeleton_00001_.png").is_file()


def test_detect_pose_failures(fake, tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(_png())
    fake["pose"] = [{"people": [], "canvas_width": 64, "canvas_height": 64}]
    assert run(comfyui.comfyui_detect_pose(str(src), output_dir=str(tmp_path)))[0].startswith(
        "DWPose found no figure in a.png")
    fake["dwpose"] = False
    out = run(comfyui.comfyui_detect_pose(str(src), output_dir=str(tmp_path)))[0]
    assert out.startswith("Failed to detect a pose: ComfyUI has no DWPose node") and "controlnet_aux" in out


def test_pose_points_pixels_and_normalized():
    found = {"neck": (48, 30), "left ankle": (60, 120)}
    for normalized in (False, True):
        points = {p[0]: p[1:] for p in comfyui.pose_points(_pose_json(found, normalized=normalized))}
        assert points["neck"] == pytest.approx((48, 30, 1.0))
        assert points["left ankle"] == pytest.approx((60, 120, 1.0))
        assert points["nose"][2] == 0
    assert comfyui.pose_points({"people": []}) == []


def test_not_running(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://127.0.0.1:9")
    assert "not reachable at http://127.0.0.1:9" in run(comfyui.comfyui_status())


def test_templates_match_the_tool_graphs():
    sys.path.insert(0, os.path.join(REPO, "integrations", "comfyui"))
    import make_templates
    for name, graph in make_templates.templates().items():
        with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as f:
            assert json.load(f) == graph, f"{name} is stale: run integrations/comfyui/make_templates.py"
