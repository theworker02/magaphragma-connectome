"""Diagnose why Connectome is / is not moving forward."""
from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _core._receipts import write_tool_receipt

REPO = Path(__file__).resolve().parents[2]

# Import module names for each registered tool id
TOOL_IMPORTS: dict[str, str] = {
    "voxscout": "voxscout",
    "axonforge": "axonforge",
    "gaphound": "gaphound",
    "tilemedic": "tilemedic",
    "branchjudge": "branchjudge",
    "seamsmith": "seamsmith",
    "morphguard": "morphguard",
    "neurocache": "neurocache",
    "synapselens": "synapselens",
    "tracewire": "tracewire",
    "deltagraph": "deltagraph",
    "runddoctor": "runddoctor",
    "hyperdrain": "hyperdrain",
    "regionbench": "regionbench",
    "pipeline": "pipeline",
    "edgeprobe": "edgeprobe",
    "artifactvet": "artifactvet",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended: list[str] = field(default_factory=list)
    switch: dict[str, Any] = field(default_factory=dict)
    tool_imports: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "warnings": self.warnings,
            "recommended": self.recommended,
            "switch": self.switch,
            "tool_imports": self.tool_imports,
        }

    def format_summary(self) -> str:
        lines = []
        for c in self.checks:
            mark = "OK" if c.ok else "FAIL"
            lines.append(f"[{mark}] {c.name}: {c.detail}")
        if self.switch:
            lines.append("")
            lines.append(
                f"AxonForge switch: powered_on={self.switch.get('powered_on')} "
                f"mode={self.switch.get('mode')} reason={self.switch.get('last_reason', '')}"
            )
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"! {w}")
        if self.recommended:
            lines.append("")
            lines.append("Recommended next commands:")
            for r in self.recommended:
                lines.append(f"  {r}")
        return chr(10).join(lines)


class RunDoctor:
    def run(self) -> DoctorReport:
        report = DoctorReport()
        datasets = REPO / "datasets"
        report.checks.append(Check("dataset readable", datasets.exists(), str(datasets)))
        af = REPO / "receipts" / "axonforge" / "work_graph.json"
        report.checks.append(Check("manifests valid", True if not af.exists() else self._json_ok(af), str(af)))
        models = REPO / "models"
        report.checks.append(Check("checkpoint available", models.exists(), str(models)))
        gpu = False
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
        report.checks.append(Check("GPU detected", True, "optional" if not gpu else "cuda available"))

        af_ok = False
        try:
            import axonforge  # noqa: F401
            af_ok = True
        except Exception as exc:
            report.checks.append(Check("AxonForge operational", False, str(exc)))
        else:
            report.checks.append(Check("AxonForge operational", af_ok, "import ok"))

        # AxonForge light switch state
        try:
            from axonforge.switch import SWITCH
            report.switch = SWITCH.as_dict()
            report.checks.append(Check(
                "AxonForge switch",
                True,
                f"on={report.switch.get('powered_on')} mode={report.switch.get('mode')}",
            ))
        except Exception as exc:
            report.checks.append(Check("AxonForge switch", False, str(exc)))

        # Each registered tool importable
        try:
            import registry
            tool_ids = [t.id for t in registry.list_tools()]
        except Exception:
            tool_ids = list(TOOL_IMPORTS.keys())
        import_ok = 0
        for tid in tool_ids:
            mod = TOOL_IMPORTS.get(tid, tid)
            try:
                importlib.import_module(mod)
                report.tool_imports[tid] = True
                import_ok += 1
            except Exception as exc:
                report.tool_imports[tid] = False
                report.warnings.append(f"tool import failed: {tid} ({exc})")
        report.checks.append(Check(
            "registered tools importable",
            import_ok == len(tool_ids),
            f"{import_ok}/{len(tool_ids)} ok",
        ))

        cache = REPO / "receipts" / "neurocache"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            probe = cache / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            cache_ok = True
        except Exception as exc:
            cache_ok = False
            report.checks.append(Check("cache writable", False, str(exc)))
        if cache_ok:
            report.checks.append(Check("cache writable", True, str(cache)))

        usage = shutil.disk_usage(str(REPO))
        free_gb = usage.free / (1024 ** 3)
        report.checks.append(Check("storage available", free_gb > 1.0, f"{free_gb:.1f} GiB free"))

        stale = failed = 0
        if af.exists():
            tasks = json.loads(af.read_text(encoding="utf-8")).get("tasks", {})
            for v in tasks.values():
                if v.get("status") == "REJECTED":
                    failed += 1
                if v.get("status") == "PENDING":
                    stale += 1

        gh = REPO / "receipts" / "gaphound" / "latest_scan.json"
        tm = REPO / "receipts" / "tilemedic"
        plan = REPO / "receipts" / "gaphound" / "repair_plan.json"

        if gh.exists():
            counts = json.loads(gh.read_text(encoding="utf-8")).get("counts", {})
            if counts.get("stale", 0):
                report.warnings.append(f"{counts['stale']} stale tiles")
            if counts.get("missing", 0):
                report.warnings.append(f"{counts['missing']} missing regions")
            if counts.get("failed", 0):
                report.warnings.append(f"{counts['failed']} failed inference jobs")
        elif failed:
            report.warnings.append(f"{failed} failed inference jobs")
        if stale:
            report.warnings.append(f"{stale} unfinished tiles in work graph")

        # Specific next commands from GapHound / TileMedic state
        if report.switch and not report.switch.get("powered_on", True):
            report.recommended.append("connectome switch on")
        if failed or any("failed" in w for w in report.warnings):
            report.recommended.append("connectome tilemedic recover --failed")
            report.recommended.append("connectome pipeline close-gaps")
        if any("missing" in w or "stale" in w for w in report.warnings):
            report.recommended.append("connectome gaphound scan && connectome gaphound repair-plan")
            report.recommended.append("connectome pipeline close-gaps --shape 100,100,100")
        if plan.exists() and not any("close-gaps" in r for r in report.recommended):
            n = json.loads(plan.read_text(encoding="utf-8")).get("n_work_items", 0)
            if n:
                report.recommended.append(f"connectome pipeline close-gaps  # repair_plan has {n} items")
        if tm.exists() and any(tm.glob("**/*.json")):
            if not any("tilemedic" in r for r in report.recommended):
                report.recommended.append("connectome tilemedic recover --failed  # prior TileMedic receipts present")
        if not report.recommended:
            report.recommended.append("connectome pipeline plan  # VoxScout -> AxonForge hints")
            report.recommended.append("connectome tools")

        write_tool_receipt("runddoctor", report.as_dict())
        return report

    def _json_ok(self, path: Path) -> bool:
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return True
        except Exception:
            return False
