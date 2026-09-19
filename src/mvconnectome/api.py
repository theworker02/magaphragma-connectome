"""Local-only read API for evidence-backed connectome data."""

from __future__ import annotations

import json
import io
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np
from PIL import Image

from .repository import ConnectomeRepository


class ApiHandler(SimpleHTTPRequestHandler):
    """Serve serialized records and the static explorer without mutation routes."""
    repository_root: Path
    viewer_root: Path

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(self.viewer_root), **kwargs)

    def _send_json(self, status: HTTPStatus, value: object) -> None:
        payload = json.dumps(value, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # Internal boundary helper; serializes images only for the local read API.
    def _send_png(self, array: np.ndarray) -> None:
        buffer = io.BytesIO(); Image.fromarray(array.astype(np.uint8)).save(buffer, format="PNG"); payload = buffer.getvalue()
        self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        repository = ConnectomeRepository(self.repository_root)
        if parsed.path == "/api/status":
            self._send_json(HTTPStatus.OK, repository.status())
            return
        if parsed.path.startswith("/api/datasets/"):
            value = repository.source_dataset(parsed.path.rsplit("/", 1)[-1])
            self._send_json(HTTPStatus.OK if value else HTTPStatus.NOT_FOUND, value or {"error": "dataset not found"})
            return
        if parsed.path.startswith("/api/evidence/"):
            value = repository.evidence(parsed.path.rsplit("/", 1)[-1])
            self._send_json(HTTPStatus.OK if value else HTTPStatus.NOT_FOUND, value or {"error": "evidence not found"})
            return
        if parsed.path == "/api/segments":
            self._send_json(HTTPStatus.OK, {"segments": repository.derived_segments(), "release_eligible": False})
            return
        if parsed.path.startswith("/api/segments/"):
            value = repository.derived_segment(parsed.path.rsplit("/", 1)[-1])
            self._send_json(HTTPStatus.OK if value else HTTPStatus.NOT_FOUND, value or {"error": "segment not found"})
            return
        if parsed.path == "/api/volumes/MV-FIBSEM-WASP5-YURI-4C/slice":
            z = int(parse_qs(parsed.query).get("z", ["6032"])[0]); array_path = self.repository_root / "datasets" / "cache" / "MV-FIBSEM-WASP5-YURI-4C" / "MV-REG-PHASE2-001" / "raw_zyx.npy"
            if not array_path.exists() or not 6000 <= z < 6064:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "volume or source z outside locally ingested bounds"}); return
            self._send_png(np.load(array_path, mmap_mode="r")[z-6000]); return
        if parsed.path.startswith("/api/neurons/"):
            value = repository.neuron_with_evidence(parsed.path.rsplit("/", 1)[-1])
            self._send_json(HTTPStatus.OK if value else HTTPStatus.NOT_FOUND, value or {"error": "neuron not found"})
            return
        if parsed.path == "/api/connections/trace":
            query = parse_qs(parsed.query)
            pre, post = query.get("pre", [""])[0], query.get("post", [""])[0]
            value = repository.connection_trace(pre, post)
            self._send_json(HTTPStatus.OK if value else HTTPStatus.NOT_FOUND, value or {"error": "connection not found"})
            return
        super().do_GET()


def serve(repository_root: Path, viewer_root: Path, port: int) -> None:
    """Start the explicit local explorer against one repository and viewer root."""
    handler = type("VigiliaApiHandler", (ApiHandler,), {"repository_root": repository_root, "viewer_root": viewer_root})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Vigilia explorer: http://127.0.0.1:{port}")
    server.serve_forever()
