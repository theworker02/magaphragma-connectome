"""GapHound scan + repair-plan against AxonForge manifests."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from _core._schemas import GapKind
from _core._spatial import AABB, TileIndex
from _core._receipts import write_tool_receipt

REPO = Path(__file__).resolve().parents[2]
AF_RECEIPTS = REPO / "receipts" / "axonforge"


@dataclass
class GapRegion:
    kind: str
    box: dict
    tile_key: str
    reason: str
    priority: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GapReport:
    shape_zyx: tuple[int, int, int]
    tile_zyx: tuple[int, int, int]
    coverage: float
    validated: float
    missing_frac: float
    regions: list[GapRegion] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape_zyx": list(self.shape_zyx),
            "tile_zyx": list(self.tile_zyx),
            "coverage": self.coverage,
            "validated": self.validated,
            "missing_frac": self.missing_frac,
            "counts": self.counts,
            "regions": [r.as_dict() for r in self.regions],
            "highest_priority": self.regions[0].as_dict() if self.regions else None,
        }

    def format_summary(self) -> str:
        lines = [
            f"Coverage:        {100 * self.coverage:.2f}%",
            f"Validated:       {100 * self.validated:.2f}%",
            f"Missing:         {100 * self.missing_frac:.2f}%",
            "",
            f"{self.counts.get('missing', 0)} missing regions",
            f"{self.counts.get('failed', 0)} failed tiles",
            f"{self.counts.get('stale', 0)} stale outputs",
            f"{self.counts.get('boundary-incomplete', 0)} boundary gaps",
        ]
        if self.regions:
            top = self.regions[0]
            b = top.box
            lines += [
                "",
                "Highest-priority gap:",
                f"x={b['x0']}:{b['x1']}",
                f"y={b['y0']}:{b['y1']}",
                f"z={b['z0']}:{b['z1']}",
                f"Reason: {top.reason}",
            ]
        return chr(10).join(lines)


class GapHound:
    def __init__(self, tile_zyx: tuple[int, int, int] = (32, 64, 64)) -> None:
        self.tile_zyx = tile_zyx

    def _load_manifest(self, path: Path | None) -> dict:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        # AxonForge work graph
        wg = AF_RECEIPTS / "work_graph.json"
        if wg.exists():
            return json.loads(wg.read_text(encoding="utf-8"))
        return {}

    def scan(
        self,
        shape_zyx: tuple[int, int, int],
        *,
        manifest: Path | None = None,
        max_age_s: float = 86400.0,
    ) -> GapReport:
        raw = self._load_manifest(manifest)
        tasks = raw.get("tasks", {})
        index = TileIndex(shape_zyx, tuple(raw.get("tile_zyx", self.tile_zyx)))
        now = time.time()
        regions: list[GapRegion] = []
        processed = validated = 0
        total = 0
        for key, box in index.iter_tiles():
            total += 1
            t = tasks.get(key)
            if t is None:
                # try alternate key styles from AxonForge (AF-VOLUME-...)
                t = self._fuzzy_task(tasks, box)
            status = (t or {}).get("status", "MISSING")
            reason = (t or {}).get("reason", "")
            updated = float((t or {}).get("updated_s", 0) or 0)
            kind = None
            pri = 0.0
            if t is None or status in ("MISSING", "PENDING") and t is None:
                kind, pri, reason = GapKind.MISSING.value, 0.95, reason or "inference artifact absent"
            elif status == "REJECTED":
                kind, pri, reason = GapKind.FAILED.value, 0.9, reason or "tile rejected by gate"
            elif status in ("PENDING", "RUNNING"):
                kind, pri, reason = GapKind.UNVERIFIED.value, 0.55, reason or "not yet validated"
            elif updated and (now - updated) > max_age_s:
                kind, pri, reason = GapKind.STALE.value, 0.7, "stale output beyond max_age"
                processed += 1
            elif status in ("VALIDATED", "CACHED"):
                processed += 1
                validated += 1
                # boundary incomplete: validated but neighbor missing
                if self._boundary_incomplete(key, index, tasks):
                    kind, pri, reason = GapKind.BOUNDARY_INCOMPLETE.value, 0.65, "neighbor gap at seam"
                else:
                    continue
            else:
                kind, pri, reason = GapKind.UNVERIFIED.value, 0.4, f"status={status}"

            regions.append(GapRegion(kind=kind, box=box.as_dict(), tile_key=key, reason=reason, priority=pri))

        regions.sort(key=lambda r: -r.priority)
        counts: dict[str, int] = {}
        for r in regions:
            counts[r.kind] = counts.get(r.kind, 0) + 1
        coverage = processed / total if total else 0.0
        val = validated / total if total else 0.0
        missing = counts.get(GapKind.MISSING.value, 0) / total if total else 0.0
        report = GapReport(shape_zyx, index.tile_zyx, coverage, val, missing, regions, counts)
        write_tool_receipt("gaphound", report.as_dict())
        out = REPO / "receipts" / "gaphound" / "latest_scan.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2) + chr(10), encoding="utf-8")
        try:
            from _core.artifactvet import vet_gaphound_scan
            vet = vet_gaphound_scan(out)
            if not vet.ok:
                # non-fatal warn — scan still returned
                pass
            (REPO / "receipts" / "gaphound" / "latest_scan.vet.json").write_text(
                json.dumps(vet.as_dict(), indent=2) + chr(10), encoding="utf-8"
            )
        except Exception:
            pass
        return report

    def _fuzzy_task(self, tasks: dict, box: AABB) -> dict | None:
        # Match AF-VOLUME style by grid indices
        tz, ty, tx = self.tile_zyx
        iz, iy, ix = box.z0 // tz, box.y0 // ty, box.x0 // tx
        for k, v in tasks.items():
            tile = v.get("tile") or {}
            if tile.get("z") == iz and tile.get("y") == iy and tile.get("x") == ix:
                return v
            if f"Z{iz:03d}" in k and f"Y{iy:03d}" in k and f"X{ix:03d}" in k:
                return v
        return None

    def _boundary_incomplete(self, key: str, index: TileIndex, tasks: dict) -> bool:
        for nb in index.neighbors(key, halo=1):
            if nb not in tasks and self._fuzzy_task(tasks, AABB(0, 0, 0, 1, 1, 1)) is None:
                # neighbor key absent entirely
                found = False
                for k in tasks:
                    if nb.split("-")[-1] in k or nb in k:
                        found = True
                        break
                # simpler: if tasks empty of neighbor grid
                if not any(nb.replace("T-", "") in k or k.endswith(nb) for k in tasks):
                    # check AF keys for neighbor indices
                    parts = nb.replace("T-", "").split("-")
                    iz, iy, ix = (int(p[1:]) for p in parts)
                    if not any(
                        (tasks[k].get("tile") or {}).get("z") == iz
                        and (tasks[k].get("tile") or {}).get("y") == iy
                        and (tasks[k].get("tile") or {}).get("x") == ix
                        for k in tasks
                    ):
                        return True
        return False

    def repair_plan(self, report: GapReport | None = None) -> dict[str, Any]:
        if report is None:
            path = REPO / "receipts" / "gaphound" / "latest_scan.json"
            if not path.exists():
                raise FileNotFoundError("No scan report — run gaphound scan first")
            raw = json.loads(path.read_text(encoding="utf-8"))
            report = GapReport(
                tuple(raw["shape_zyx"]),
                tuple(raw["tile_zyx"]),
                raw["coverage"],
                raw["validated"],
                raw["missing_frac"],
                [GapRegion(**r) for r in raw["regions"]],
                raw.get("counts", {}),
            )
        # Minimum worklist: only missing/failed/stale/boundary-incomplete, sorted by priority.
        actionable = [
            r for r in report.regions
            if r.kind in {
                GapKind.MISSING.value,
                GapKind.FAILED.value,
                GapKind.STALE.value,
                GapKind.BOUNDARY_INCOMPLETE.value,
            }
        ]
        worklist = [
            {
                "tile_key": r.tile_key,
                "box": r.box,
                "action": "recompute" if r.kind != GapKind.FAILED.value else "recover_then_recompute",
                "reason": r.reason,
                "priority": r.priority,
                "kind": r.kind,
            }
            for r in actionable
        ]
        plan = {
            "n_work_items": len(worklist),
            "vs_full_volume_tiles": int(np_prod(report.shape_zyx) / max(np_prod(report.tile_zyx), 1)),
            "worklist": worklist,
            "note": "Minimum AxonForge worklist to close holes — does not rerun full volume.",
        }
        write_tool_receipt("gaphound", {"repair_plan": plan})
        out = REPO / "receipts" / "gaphound" / "repair_plan.json"
        out.write_text(json.dumps(plan, indent=2) + chr(10), encoding="utf-8")
        return plan


def np_prod(shape: Iterable[int]) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return n
