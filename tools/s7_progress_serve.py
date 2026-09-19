#!/usr/bin/env python3
"""Serve a live Affinity S7 hybrid-drain progress dashboard.

Polls queue_state on each /api/progress request and auto-refreshes the UI.
Default bind: 127.0.0.1:8743
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from s7_progress_snapshot import DEFAULT_CHUNKS, default_out, snapshot, write_snapshot  # noqa: E402

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Affinity S7 · Progress</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Sora:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --ink: #14201c;
      --ink-soft: #2a3a34;
      --paper: #e8efe8;
      --mist: #d3ddd6;
      --line: rgba(20, 32, 28, 0.14);
      --accent: #c45c26;
      --accent-deep: #8f3d14;
      --local: #1f6f5b;
      --vast: #b45309;
      --bar-track: rgba(20, 32, 28, 0.1);
      --glow: rgba(196, 92, 38, 0.22);
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      min-height: 100%;
      color: var(--ink);
      font-family: "Sora", system-ui, sans-serif;
      background:
        radial-gradient(1200px 700px at 12% -10%, rgba(196, 92, 38, 0.16), transparent 55%),
        radial-gradient(900px 600px at 95% 10%, rgba(31, 111, 91, 0.14), transparent 50%),
        linear-gradient(165deg, #f4f7f3 0%, #dfe8e1 42%, #cfdad2 100%);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.35;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='0.35'/%3E%3C/svg%3E");
      mix-blend-mode: multiply;
    }
    main {
      position: relative;
      max-width: 920px;
      margin: 0 auto;
      padding: clamp(2rem, 6vw, 4.5rem) 1.5rem 3rem;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 1.75rem;
    }
    .brand {
      font-family: "Instrument Serif", Georgia, serif;
      font-size: clamp(3rem, 9vw, 5.5rem);
      line-height: 0.92;
      letter-spacing: -0.02em;
      margin: 0;
      animation: rise 0.7s ease-out both;
    }
    .brand em {
      font-style: italic;
      color: var(--accent-deep);
    }
    .lede {
      margin: 0;
      max-width: 34rem;
      font-size: 1.05rem;
      color: var(--ink-soft);
      animation: rise 0.8s ease-out 0.05s both;
    }
    .panel {
      background: rgba(248, 251, 248, 0.72);
      border: 1px solid var(--line);
      backdrop-filter: blur(10px);
      padding: clamp(1.25rem, 3vw, 2rem);
      animation: rise 0.85s ease-out 0.1s both;
    }
    .row {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }
    .pct {
      font-family: "Instrument Serif", Georgia, serif;
      font-size: clamp(3.5rem, 12vw, 6.5rem);
      line-height: 0.9;
      letter-spacing: -0.03em;
      margin: 0;
    }
    .pct span {
      font-size: 0.42em;
      color: var(--ink-soft);
      margin-left: 0.08em;
    }
    .meta {
      text-align: right;
      font-size: 0.92rem;
      color: var(--ink-soft);
    }
    .meta strong {
      display: block;
      color: var(--ink);
      font-weight: 600;
      font-size: 1.05rem;
    }
    .track {
      position: relative;
      height: clamp(2.4rem, 5vw, 3.2rem);
      background: var(--bar-track);
      overflow: hidden;
    }
    .fill {
      height: 100%;
      width: 0%;
      background:
        linear-gradient(90deg, var(--local) 0%, #2f8f74 45%, var(--accent) 100%);
      box-shadow: 0 0 28px var(--glow);
      transition: width 0.85s cubic-bezier(0.22, 1, 0.36, 1);
      position: relative;
    }
    .fill::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,0.28) 48%, transparent 62%);
      background-size: 220% 100%;
      animation: sheen 2.8s ease-in-out infinite;
    }
    .shards {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.9rem;
      margin-top: 1.25rem;
    }
    @media (max-width: 640px) {
      .shards { grid-template-columns: 1fr; }
      .meta { text-align: left; }
    }
    .shard {
      border-top: 3px solid var(--local);
      padding-top: 0.7rem;
    }
    .shard.vast { border-top-color: var(--vast); }
    .shard .label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--ink-soft);
      margin-bottom: 0.35rem;
    }
    .shard .val {
      font-family: "Instrument Serif", Georgia, serif;
      font-size: 2rem;
      line-height: 1;
    }
    .shard .sub {
      margin-top: 0.35rem;
      font-size: 0.85rem;
      color: var(--ink-soft);
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 0.65rem 1.4rem;
      margin-top: 1.2rem;
      font-size: 0.9rem;
      color: var(--ink-soft);
    }
    .stats b { color: var(--ink); font-weight: 600; }
    footer {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
      font-size: 0.82rem;
      color: var(--ink-soft);
      animation: rise 0.9s ease-out 0.15s both;
    }
    .pulse {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }
    .pulse i {
      width: 0.55rem;
      height: 0.55rem;
      border-radius: 50%;
      background: var(--local);
      animation: blink 1.6s ease-in-out infinite;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(14px); }
      to { opacity: 1; transform: none; }
    }
    @keyframes sheen {
      0%, 100% { background-position: 120% 0; }
      50% { background-position: -20% 0; }
    }
    @keyframes blink {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.35; transform: scale(0.85); }
    }
    .err {
      color: #8a1f1f;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <main>
    <h1 class="brand">Affinity <em>S7</em></h1>
    <p class="lede">Hybrid drain across local even shards and Vast odd shards. Live progress from the shared queue.</p>
    <section class="panel" aria-live="polite">
      <div class="row">
        <p class="pct" id="pct">—<span>%</span></p>
        <div class="meta">
          <strong id="counts">— / — chunks</strong>
          <div id="updated">Waiting for first snapshot…</div>
        </div>
      </div>
      <div class="track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="bar">
        <div class="fill" id="fill"></div>
      </div>
      <div class="shards">
        <div class="shard">
          <div class="label">Shard 0 · Local · even</div>
          <div class="val" id="local">—</div>
          <div class="sub" id="local-sub">RX 7800 XT / ROCm</div>
        </div>
        <div class="shard vast">
          <div class="label">Shard 1 · Vast · odd</div>
          <div class="val" id="vast">—</div>
          <div class="sub" id="vast-sub">RTX 3060 Ti offload</div>
        </div>
      </div>
      <div class="stats">
        <span>Running <b id="running">—</b></span>
        <span>Pending <b id="pending">—</b></span>
        <span>Failed <b id="failed">—</b></span>
        <span>Affinity files <b id="aff">—</b></span>
      </div>
      <p class="err" id="err" hidden></p>
    </section>
    <footer>
      <div class="pulse"><i></i> Auto-refresh every 4s</div>
      <div id="path">—</div>
    </footer>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    function fmt(n) {
      return Number(n).toLocaleString("en-US");
    }
    function fmtPct(p) {
      const n = Number(p);
      if (!Number.isFinite(n)) return "—";
      return n >= 10 ? n.toFixed(1) : n.toFixed(2);
    }
    function apply(d) {
      const pct = Number(d.pct) || 0;
      $("pct").innerHTML = fmtPct(pct) + "<span>%</span>";
      $("counts").textContent = fmt(d.completed) + " / " + fmt(d.total) + " chunks";
      $("updated").textContent = "Updated " + (d.updated_at || "—");
      $("fill").style.width = Math.max(0, Math.min(100, pct)) + "%";
      $("bar").setAttribute("aria-valuenow", String(pct));
      $("local").textContent = fmt(d.local_even);
      $("vast").textContent = fmt(d.vast_odd);
      $("running").textContent = fmt(d.running);
      $("pending").textContent = fmt(d.pending);
      $("failed").textContent = fmt(d.failed);
      const af = d.affinity_files || {};
      $("aff").textContent = fmt(af.total || 0);
      $("path").textContent = d.out || "";
      $("err").hidden = true;
    }
    async function tick() {
      try {
        const r = await fetch("/api/progress?_=" + Date.now(), { cache: "no-store" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        apply(await r.json());
      } catch (e) {
        $("err").hidden = false;
        $("err").textContent = "Could not load progress: " + e.message;
      }
    }
    tick();
    setInterval(tick, 4000);
  </script>
</body>
</html>
"""


class ProgressHandler(BaseHTTPRequestHandler):
    out: Path
    chunks: Path
    write_disk: bool

    def log_message(self, fmt: str, *args) -> None:  # quieter
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path in {"/api/progress", "/api/progress.json", "/progress.json"}:
            try:
                if self.write_disk:
                    snap = write_snapshot(self.out, self.chunks)
                else:
                    snap = snapshot(self.out, self.chunks)
                body = (json.dumps(snap) + "\n").encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except Exception as exc:
                err = {"error": str(exc), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                self._send(500, (json.dumps(err) + "\n").encode("utf-8"), "application/json; charset=utf-8")
            return
        self._send(404, b'{"error":"not found"}\n', "application/json; charset=utf-8")


def background_writer(out: Path, chunks: Path, interval: float) -> None:
    while True:
        try:
            write_snapshot(out, chunks)
        except Exception as exc:
            sys.stderr.write(f"snapshot write failed: {exc}\n")
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("S7_PROGRESS_PORT", "8743")))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--write-interval", type=float, default=4.0, help="Also write progress_ui.json on this interval")
    ap.add_argument("--no-disk-write", action="store_true")
    args = ap.parse_args()

    out = args.out or default_out()
    ProgressHandler.out = out
    ProgressHandler.chunks = args.chunks
    ProgressHandler.write_disk = not args.no_disk_write

    # Prime first snapshot
    try:
        snap = write_snapshot(out, args.chunks) if ProgressHandler.write_disk else snapshot(out, args.chunks)
        print(json.dumps({k: snap[k] for k in ("pct", "completed", "total", "local_even", "vast_odd", "updated_at")}, indent=2))
    except Exception as exc:
        print(f"warning: initial snapshot failed: {exc}", file=sys.stderr)

    if ProgressHandler.write_disk and args.write_interval > 0:
        t = threading.Thread(target=background_writer, args=(out, args.chunks, args.write_interval), daemon=True)
        t.start()

    server = ThreadingHTTPServer((args.host, args.port), ProgressHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Affinity S7 progress UI → {url}", flush=True)
    print(f"S7_OUT={out}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
