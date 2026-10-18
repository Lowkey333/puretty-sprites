"""Puretty Bridge: lets local tools drive Krita over a localhost socket.

The libresprite-mcp server (tools/krita.py) uses it so an AI agent can
open and create documents, push images in as layers, read the canvas back
and export it, while you paint in the same window.

Protocol: one JSON object per line over TCP on 127.0.0.1.

    request:  {"id": 1, "token": "...", "command": "status", "params": {}}
    response: {"id": 1, "ok": true, "result": {...}}
              {"id": 1, "ok": false, "error": "message"}

The server listens on 127.0.0.1 only, on PURETTY_BRIDGE_PORT (default
8765, or the next free port up to 8784). On start it writes the port and a
random per-session token to puretty_bridge.json in Krita's app-data folder
(%APPDATA%\\krita on Windows); requests must carry that token, so only
programs that can read your profile can talk to it. The file is removed
when Krita quits.

Handlers run on Krita's GUI thread (QTcpServer delivers its signals
there), which is where the Krita API must be called.
"""
import base64
import contextlib
import io
import json
import os
import secrets
import sys
import traceback

from krita import Extension, InfoObject, Krita

try:
    from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QObject, QStandardPaths, Qt
    from PyQt5.QtGui import QImage
    from PyQt5.QtNetwork import QHostAddress, QTcpServer
    from PyQt5.QtWidgets import QApplication, QMessageBox
    _LOCALHOST = QHostAddress.LocalHost
    _ARGB32 = QImage.Format_ARGB32
    _READ_WRITE = QIODevice.ReadWrite
    _APP_DATA = QStandardPaths.AppDataLocation
    _FAST, _SMOOTH, _KEEP_ASPECT = Qt.FastTransformation, Qt.SmoothTransformation, Qt.KeepAspectRatio
    _IGNORE_ASPECT = Qt.IgnoreAspectRatio
except ImportError:  # Krita 6 (Qt6)
    from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QStandardPaths, Qt
    from PyQt6.QtGui import QImage
    from PyQt6.QtNetwork import QHostAddress, QTcpServer
    from PyQt6.QtWidgets import QApplication, QMessageBox
    _LOCALHOST = QHostAddress.SpecialAddress.LocalHost
    _ARGB32 = QImage.Format.Format_ARGB32
    _READ_WRITE = QIODevice.OpenModeFlag.ReadWrite
    _APP_DATA = QStandardPaths.StandardLocation.AppDataLocation
    _FAST = Qt.TransformationMode.FastTransformation
    _SMOOTH = Qt.TransformationMode.SmoothTransformation
    _KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    _IGNORE_ASPECT = Qt.AspectRatioMode.IgnoreAspectRatio

VERSION = "1.0"
DEFAULT_PORT = 8765
PORT_ATTEMPTS = 20
MAX_REQUEST_BYTES = 64 * 1024 * 1024
RASTER_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


class BridgeError(Exception):
    """A request failed in a way the caller should hear about."""


def discovery_path():
    folder = QStandardPaths.writableLocation(_APP_DATA) or os.path.expanduser("~")
    return os.path.join(folder, "puretty_bridge.json")


# ------------------------------------------------------------ helpers

def _app():
    return Krita.instance()


def _doc_info(doc):
    return {
        "name": doc.name(),
        "path": doc.fileName(),
        "width": doc.width(),
        "height": doc.height(),
        "color_model": doc.colorModel(),
        "color_depth": doc.colorDepth(),
        "resolution": doc.resolution(),
        "modified": doc.modified(),
        "active": _app().activeDocument() is not None and _same_doc(_app().activeDocument(), doc),
    }


def _same_doc(a, b):
    # Wrappers are recreated on every call; compare what identifies a document.
    return (a.fileName(), a.name(), a.width(), a.height()) == (b.fileName(), b.name(), b.width(), b.height())


def _document(params):
    """The document named by params["document"] (path or name), else the active one."""
    wanted = params.get("document") or ""
    docs = _app().documents()
    if wanted:
        norm = os.path.normcase(os.path.abspath(wanted)) if os.path.sep in wanted or "/" in wanted else None
        for doc in docs:
            if doc.name() == wanted or (norm and doc.fileName()
                                        and os.path.normcase(os.path.abspath(doc.fileName())) == norm):
                return doc
        raise BridgeError("No open document matches %r" % wanted)
    doc = _app().activeDocument()
    if doc is None:
        if docs:
            return docs[-1]
        raise BridgeError("No document is open in Krita")
    return doc


def _window():
    app = _app()
    win = app.activeWindow()
    if win is None and app.windows():
        win = app.windows()[0]
    if win is None:
        win = app.openWindow()
        win.qwindow().show()
    return win


def _show(doc):
    _window().addView(doc)
    _app().setActiveDocument(doc)


def _find_node(doc, name):
    node = doc.nodeByName(name)
    if node is None:
        raise BridgeError("No layer named %r in %s" % (name, doc.name()))
    return node


def _require_rgba8(doc, what):
    if doc.colorModel() != "RGBA" or doc.colorDepth() != "U8":
        raise BridgeError("%s needs an RGBA 8-bit document; %s is %s %s"
                          % (what, doc.name(), doc.colorModel(), doc.colorDepth()))


def _qimage_from_bgra(data, width, height):
    # Krita's RGBA U8 pixels are stored B, G, R, A: QImage's ARGB32 layout on
    # little-endian machines. copy() detaches the image from the Python buffer.
    raw = bytes(data)
    return QImage(raw, width, height, width * 4, _ARGB32).copy()


def _canvas_image(doc, layer):
    """The flattened canvas, or one layer at canvas size, as a QImage."""
    _require_rgba8(doc, "Reading pixels")
    doc.refreshProjection()
    doc.waitForDone()
    w, h = doc.width(), doc.height()
    if layer:
        data = _find_node(doc, layer).projectionPixelData(0, 0, w, h)
    else:
        data = doc.pixelData(0, 0, w, h)
    return _qimage_from_bgra(data, w, h)


def _png_bytes(image):
    buffer = QBuffer()
    buffer.open(_READ_WRITE)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


def _node_info(node):
    b = node.bounds()
    info = {
        "name": node.name(),
        "type": node.type(),
        "visible": node.visible(),
        "opacity": node.opacity(),
        "blending_mode": node.blendingMode(),
        "locked": node.locked(),
        "bounds": [b.x(), b.y(), b.width(), b.height()],
    }
    children = node.childNodes()
    if children:
        info["children"] = [_node_info(c) for c in children]
    return info


# ------------------------------------------------------------ commands

def cmd_ping(params):
    return {"pong": True, "bridge_version": VERSION}


def cmd_status(params):
    app = _app()
    active = app.activeDocument()
    return {
        "krita_version": app.version(),
        "bridge_version": VERSION,
        "python": sys.version.split()[0],
        "documents": [_doc_info(d) for d in app.documents()],
        "active_document": (active.fileName() or active.name()) if active else None,
    }


def cmd_new_document(params):
    width = int(params.get("width", 0))
    height = int(params.get("height", 0))
    if width <= 0 or height <= 0:
        raise BridgeError("width and height must be > 0")
    name = params.get("name") or "Untitled"
    doc = _app().createDocument(width, height, name, params.get("color_model", "RGBA"),
                                params.get("color_depth", "U8"), "", float(params.get("resolution", 72)))
    if doc is None:
        raise BridgeError("Krita could not create the document")
    if params.get("background", "transparent") == "transparent":
        # createDocument fills its "Background" layer with opaque white;
        # swap it for an empty layer so sprites start transparent.
        root = doc.rootNode()
        old = root.childNodes()
        layer = doc.createNode("Layer 1", "paintlayer")
        root.addChildNode(layer, None)
        for node in old:
            node.remove()
        doc.setActiveNode(layer)
        doc.refreshProjection()
    _show(doc)
    return _doc_info(doc)


def cmd_open_document(params):
    path = params.get("path") or ""
    if not os.path.isfile(path):
        raise BridgeError("File not found: %s" % path)
    doc = _app().openDocument(path)
    if doc is None:
        raise BridgeError("Krita could not open %s" % path)
    _show(doc)
    return _doc_info(doc)


def cmd_list_layers(params):
    doc = _document(params)
    return {
        "document": doc.fileName() or doc.name(),
        "order": "bottom_to_top",
        "layers": [_node_info(n) for n in doc.rootNode().childNodes()],
    }


def cmd_add_image_layer(params):
    path = params.get("path") or ""
    image = QImage(path)
    if image.isNull():
        raise BridgeError("Could not load image %s" % path)
    doc = _document(params)
    _require_rgba8(doc, "Adding an image layer")
    image = image.convertToFormat(_ARGB32)
    w, h = image.width(), image.height()
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes() if hasattr(image, "sizeInBytes") else image.byteCount())
    data = bytes(ptr)
    stride = image.bytesPerLine()
    if stride != w * 4:  # drop any row padding
        data = b"".join(data[y * stride:y * stride + w * 4] for y in range(h))
    name = params.get("name") or os.path.splitext(os.path.basename(path))[0]
    node = doc.createNode(name, "paintlayer")
    root = doc.rootNode()
    siblings = root.childNodes()
    root.addChildNode(node, siblings[-1] if siblings else None)  # on top
    x, y = int(params.get("x", 0)), int(params.get("y", 0))
    node.setPixelData(QByteArray(data), x, y, w, h)
    opacity = params.get("opacity")
    if opacity is not None:
        node.setOpacity(int(opacity))
    doc.refreshProjection()
    return {"layer": node.name(), "x": x, "y": y, "width": w, "height": h,
            "document": doc.fileName() or doc.name()}


def cmd_export_document(params):
    path = params.get("path") or ""
    if not path:
        raise BridgeError("path is required")
    doc = _document(params)
    layer = params.get("layer") or ""
    scale = int(params.get("scale", 1))
    if scale < 1:
        raise BridgeError("scale must be >= 1")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in RASTER_FORMATS and doc.colorModel() == "RGBA" and doc.colorDepth() == "U8":
        image = _canvas_image(doc, layer)
        if scale > 1:
            image = image.scaled(image.width() * scale, image.height() * scale, _IGNORE_ASPECT, _FAST)
        if not image.save(path):
            raise BridgeError("Could not write %s" % path)
        return {"path": path, "width": image.width(), "height": image.height()}
    if layer or scale != 1:
        raise BridgeError("layer/scale export needs an RGBA 8-bit document and a PNG/JPG/BMP/WEBP/TIFF path")
    doc.setBatchmode(True)
    try:
        ok = doc.exportImage(path, InfoObject())
    finally:
        doc.setBatchmode(False)
    if not ok:
        raise BridgeError("Krita could not export %s" % path)
    return {"path": path, "width": doc.width(), "height": doc.height()}


def cmd_get_canvas(params):
    doc = _document(params)
    image = _canvas_image(doc, params.get("layer") or "")
    full_w, full_h = image.width(), image.height()
    max_size = int(params.get("max_size", 0) or 0)
    if max_size > 0 and max(full_w, full_h) > max_size:
        image = image.scaled(max_size, max_size, _KEEP_ASPECT, _SMOOTH)
    return {"png_base64": base64.b64encode(_png_bytes(image)).decode("ascii"),
            "width": image.width(), "height": image.height(),
            "canvas_width": full_w, "canvas_height": full_h,
            "document": doc.fileName() or doc.name()}


def cmd_save_document(params):
    doc = _document(params)
    path = params.get("path") or ""
    doc.setBatchmode(True)
    try:
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            ok = doc.saveAs(path)
        else:
            if not doc.fileName():
                raise BridgeError("The document has never been saved; pass a path")
            ok = doc.save()
    finally:
        doc.setBatchmode(False)
    if not ok:
        raise BridgeError("Krita could not save %s" % (path or doc.fileName()))
    return {"path": path or doc.fileName()}


def cmd_close_document(params):
    doc = _document(params)
    name = doc.fileName() or doc.name()
    doc.setModified(False)
    doc.close()
    return {"closed": name}


def cmd_run_python(params):
    code = params.get("code") or ""
    output = io.StringIO()
    namespace = {"Krita": Krita, "InfoObject": InfoObject, "app": _app(),
                 "doc": _app().activeDocument(), "result": None}
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            exec(compile(code, "<puretty_bridge>", "exec"), namespace)
        except Exception:
            raise BridgeError(output.getvalue() + traceback.format_exc())
    result = namespace.get("result")
    return {"stdout": output.getvalue(), "result": None if result is None else repr(result)}


COMMANDS = {
    "ping": cmd_ping,
    "status": cmd_status,
    "new_document": cmd_new_document,
    "open_document": cmd_open_document,
    "list_layers": cmd_list_layers,
    "add_image_layer": cmd_add_image_layer,
    "export_document": cmd_export_document,
    "get_canvas": cmd_get_canvas,
    "save_document": cmd_save_document,
    "close_document": cmd_close_document,
    "run_python": cmd_run_python,
}


# ------------------------------------------------------------ server

class BridgeServer(QObject):
    def __init__(self):
        super().__init__()
        self.token = secrets.token_hex(16)
        self.port = None
        self.error = None
        self.path = None  # resolved once: app paths may differ during shutdown
        self._server = QTcpServer(self)
        self._server.newConnection.connect(self._accept)
        self._buffers = {}

    def start(self):
        first = int(os.environ.get("PURETTY_BRIDGE_PORT", DEFAULT_PORT))
        for port in range(first, first + PORT_ATTEMPTS):
            if self._server.listen(QHostAddress(_LOCALHOST), port):
                self.port = port
                break
        if self.port is None:
            self.error = "no free port in %d-%d: %s" % (first, first + PORT_ATTEMPTS - 1,
                                                        self._server.errorString())
            print("Puretty Bridge: " + self.error)
            return
        self.path = discovery_path()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"port": self.port, "token": self.token, "pid": os.getpid(),
                       "bridge_version": VERSION}, f)
        print("Puretty Bridge: listening on 127.0.0.1:%d" % self.port)

    def stop(self):
        self._server.close()
        if not self.path:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                mine = json.load(f).get("token") == self.token
            if mine:  # another Krita instance may have replaced it
                os.remove(self.path)
        except (OSError, ValueError):
            pass

    def _accept(self):
        while self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            self._buffers[id(sock)] = bytearray()
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda s=sock: self._drop(s))

    def _drop(self, sock):
        self._buffers.pop(id(sock), None)
        sock.deleteLater()

    def _read(self, sock):
        buf = self._buffers.get(id(sock))
        if buf is None:
            return
        buf += bytes(sock.readAll())
        if len(buf) > MAX_REQUEST_BYTES:
            sock.write(json.dumps({"ok": False, "error": "request too large"}).encode() + b"\n")
            sock.disconnectFromHost()
            return
        while b"\n" in buf:
            line, _, rest = bytes(buf).partition(b"\n")
            buf[:] = rest
            if line.strip():
                sock.write(json.dumps(self._handle(line)).encode("utf-8") + b"\n")
                sock.flush()

    def _handle(self, line):
        req_id = None
        try:
            request = json.loads(line.decode("utf-8"))
            req_id = request.get("id")
            if not secrets.compare_digest(str(request.get("token", "")), self.token):
                return {"id": req_id, "ok": False, "error": "bad or missing token"}
            handler = COMMANDS.get(request.get("command"))
            if handler is None:
                return {"id": req_id, "ok": False,
                        "error": "unknown command %r (known: %s)" % (request.get("command"), ", ".join(COMMANDS))}
            return {"id": req_id, "ok": True, "result": handler(request.get("params") or {})}
        except BridgeError as e:
            return {"id": req_id, "ok": False, "error": str(e)}
        except Exception as e:
            return {"id": req_id, "ok": False, "error": "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc())}


class PurettyBridgeExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)
        self.server = None

    def setup(self):
        self.server = BridgeServer()
        self.server.start()
        # Krita shuts Python down before QApplication.aboutToQuit, so remove
        # the connection file on its own applicationClosing signal.
        notifier = Krita.instance().notifier()
        notifier.setActive(True)
        notifier.applicationClosing.connect(self.server.stop)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.server.stop)

    def createActions(self, window):
        action = window.createAction("puretty_bridge_status", "Puretty Bridge Status", "tools/scripts")
        action.triggered.connect(self.show_status)

    def show_status(self):
        if self.server and self.server.port:
            text = ("Listening on 127.0.0.1:%d\nConnection file: %s"
                    % (self.server.port, discovery_path()))
        else:
            text = "Not running: %s" % (self.server.error if self.server else "not started")
        QMessageBox.information(None, "Puretty Bridge", text)
