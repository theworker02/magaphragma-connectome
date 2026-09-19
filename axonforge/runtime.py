"""AxonForge runtime — submit volumes, run adaptive inference, telemetry."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from axonforge.adaptive import AdaptiveReport, run_adaptive as run_adaptive_pass
from axonforge.graph import SpatialWorkGraph
from axonforge.ledger import CompletenessLedger
from axonforge.__version__ import PROJECT, SUBSYSTEM, __version__
from axonforge.switch import SWITCH
from axonforge.bridge import apply_hints_to_graph, record_trace_run, post_run_gaphound, post_run_toolchain, load_hints

DATA = Path(__file__).resolve().parents[1] / "receipts" / "axonforge"
DATA.mkdir(parents=True, exist_ok=True)


@dataclass
class Telemetry:
    total_volume: int = 0
    validated: int = 0
    unresolved: int = 0
    uncertain_regions: int = 0
    tiles_per_sec: float = 0.0
    cache_hit_rate: float = 0.0
    queue_depth: int = 0
    failed_tasks: int = 0
    adaptive_pass: str = ""
    voxels_per_sec: float = 0.0
    io_mb_s: float = 0.0
    gpu_util: float = 0.0
    vram_mb: float = 0.0
    est_remaining_s: float = 0.0
    review_queue: list[dict] = field(default_factory=list)
    last_error: str = ""
    running: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class Runtime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.graph: SpatialWorkGraph | None = None
        self.ledger = CompletenessLedger()
        self.telemetry = Telemetry()
        self.last_report: AdaptiveReport | None = None

    def clear(self) -> None:
        with self.lock:
            self.graph = None
            self.ledger = CompletenessLedger()
            self.telemetry = Telemetry()
            self.last_report = None
            for p in DATA.glob("work_graph*.json"):
                p.unlink(missing_ok=True)

    def submit_volume(
        self,
        shape_zyx: tuple[int, int, int] = (100, 100, 100),
        tile_zyx: tuple[int, int, int] = (32, 64, 64),
        *,
        reset: bool = False,
    ) -> dict:
        with self.lock:
            if not self._powered():
                return {"ok": False, "error": "AxonForge is powered off", "switch": SWITCH.as_dict()}
            if reset:
                self.graph = None
                self.ledger = CompletenessLedger()
                self.telemetry = Telemetry()
                self.last_report = None
            path = DATA / "work_graph.json"
            self.graph = SpatialWorkGraph.build(
                shape_zyx=shape_zyx,
                tile_zyx=tile_zyx,
                path=path,
                load_existing=not reset,
            )
            self.ledger.total_voxels = int(shape_zyx[0] * shape_zyx[1] * shape_zyx[2])
            self.telemetry.total_volume = self.ledger.total_voxels
            self.telemetry.queue_depth = sum(1 for _ in self.graph.iter_pending())
            hint_info = apply_hints_to_graph(self.graph)
            self.graph.save()
            return {
                "ok": True,
                "tiles": len(self.graph.tasks),
                "total_voxels": self.ledger.total_voxels,
                "queue_depth": self.telemetry.queue_depth,
                "voxscout_hints": hint_info,
            }

    def request_inference(self, *, run_adaptive: bool = True, use_cascade: bool = False) -> dict:
        with self.lock:
            if not self._powered():
                return {"ok": False, "error": "AxonForge is powered off", "switch": SWITCH.as_dict()}
            if self.graph is None:
                return {"ok": False, "error": "no volume submitted"}
            self.telemetry.running = True
            self.telemetry.last_error = ""
            graph = self.graph
            ledger = self.ledger
        try:
            t0 = time.perf_counter()
            report = run_adaptive_pass(graph, ledger, use_cascade=use_cascade)
            wall = time.perf_counter() - t0
            with self.lock:
                self.last_report = report
                self.graph = graph
                self.ledger = ledger
                pending = sum(1 for t in graph.tasks.values() if t.status == "PENDING")
                failed = sum(1 for t in graph.tasks.values() if t.status == "REJECTED")
                n_tiles = max(1, report.tiles_done + report.tiles_rejected)
                self.telemetry = Telemetry(
                    total_volume=ledger.total_voxels,
                    validated=ledger.validated_voxels,
                    unresolved=ledger.unaccounted_volume,
                    uncertain_regions=ledger.uncertain_regions,
                    tiles_per_sec=n_tiles / report.wall_seconds if report.wall_seconds > 0 else 0.0,
                    cache_hit_rate=report.cache_hit_rate,
                    queue_depth=pending,
                    failed_tasks=failed,
                    adaptive_pass=report.pass_name,
                    voxels_per_sec=report.voxels_per_sec,
                    io_mb_s=(ledger.total_voxels / 1e6) / max(wall, 1e-6),
                    gpu_util=0.0,
                    vram_mb=0.0,
                    est_remaining_s=0.0 if pending == 0 else pending * 0.01,
                    review_queue=report.review_queue,
                    running=False,
                )
                graph.save()
                self._write_receipts(report)
                shape = graph.shape_zyx
            record_trace_run(shape_zyx=shape, tiles_done=report.tiles_done, pass_name=report.pass_name)
            toolchain = post_run_toolchain(shape, tiles_done=report.tiles_done)
            gaps = toolchain.get("gaphound") or post_run_gaphound(shape)
            return {"ok": True, "report": {
                "pass_name": report.pass_name,
                "tiles_done": report.tiles_done,
                "tiles_rejected": report.tiles_rejected,
                "wall_seconds": report.wall_seconds,
                "cache_hit_rate": report.cache_hit_rate,
                "ledger": report.ledger,
                "claims": report.claims,
                "gaphound": gaps,
                "toolchain_report": toolchain,
                "toolchain": toolchain.get("tools_invoked") or [
                    "voxscout_hints", "neurocache", "tracewire", "gaphound",
                ],
            }}
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.telemetry.running = False
                self.telemetry.last_error = str(exc)
            return {"ok": False, "error": str(exc)}

    def _write_receipts(self, report: AdaptiveReport) -> None:
        summary = {
            "project": PROJECT,
            "subsystem": SUBSYSTEM,
            "version": __version__,
            "ledger": report.ledger,
            "claims": report.claims,
            "pass_name": report.pass_name,
            "cache_hit_rate": report.cache_hit_rate,
            "accepted": [k for k, v in report.claims.items() if v],
            "rejected": [k for k, v in report.claims.items() if not v],
        }
        payload = json.dumps(summary, indent=2) + chr(10)
        (DATA / "SUMMARY.json").write_text(payload, encoding="utf-8")
        out = DATA.parents[1] / "receipts" / "SUMMARY.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")

    def status(self) -> dict:
        with self.lock:
            return {
                "project": PROJECT,
                "subsystem": SUBSYSTEM,
                "version": __version__,
                "telemetry": self.telemetry.as_dict(),
                "switch": SWITCH.as_dict(),
                "toolchain": {
                    "hints_available": bool(load_hints()),
                    "connected": [
                        "voxscout", "axonforge", "gaphound", "tilemedic",
                        "branchjudge", "seamsmith", "morphguard", "neurocache",
                        "synapselens", "tracewire", "deltagraph", "runddoctor",
                        "hyperdrain", "regionbench", "pipeline", "edgeprobe",
                        "artifactvet", "registry",
                    ],
                },
            }

    def _powered(self) -> bool:
        return SWITCH.is_on()

    def switch_status(self) -> dict:
        return SWITCH.as_dict()


RUNTIME = Runtime()

def _af_work_probe():
    g = RUNTIME.graph
    pending = False
    if g is not None:
        pending = any(t.status == "PENDING" for t in g.tasks.values())
    return pending, bool(RUNTIME.telemetry.running)

SWITCH.bind_work_probe(_af_work_probe)

