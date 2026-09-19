"""Two-process determinism check for the exposure audit + -008 snapshot.
Compares substantive content (exposure states, per-source avoid-edge hashes,
available/reviewed source sets), ignoring volatile embedded paths."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(tag: str):
    d = Path(tempfile.mkdtemp())
    audit = d / f"audit_{tag}.json"
    snap = d / f"snap_{tag}.json"
    env = {"PYTHONPATH": "src"}
    subprocess.run([sys.executable, "tools/g3_exposure_audit.py", "--out", str(audit)], cwd=REPO, check=True,
                   capture_output=True, env={**__import__("os").environ, **env})
    subprocess.run([sys.executable, "tools/freeze_g3_008_exclusion_snapshot.py", "--audit", str(audit), "--out", str(snap)],
                   cwd=REPO, check=True, capture_output=True, env={**__import__("os").environ, **env})
    a = json.loads(audit.read_text())
    s = json.loads(snap.read_text())
    audit_content = [(p["package_id"], p["exposure_state"], p["reviewed_edges_hash"], p["queued_edges_hash"]) for p in a["packages"]]
    snap_content = {
        "reviewed": s["reviewed_source_ids"],
        "available": s["fully_available_source_ids"],
        "per_source": [(r["source_id"], r["whole_crop_available"], r["avoid_edges_hash"], r["avoid_exact_edge_count"]) for r in s["per_source"]],
    }
    return audit_content, snap_content


a1, s1 = run("p1")
a2, s2 = run("p2")
print(json.dumps({
    "audit_deterministic": a1 == a2,
    "snapshot_deterministic": s1 == s2,
}, indent=2))
raise SystemExit(0 if (a1 == a2 and s1 == s2) else 1)
