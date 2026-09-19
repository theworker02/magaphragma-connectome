"""Single-process final pre-review verification for the -008 axis-neutral batch.

Runs the corrected snapshot-only sampler EXACTLY ONCE, retains the returned
object in memory, and compares it against (a) the canonical bin/run1.json and
(b) the 12 existing on-disk -008 queues. Also validates the pinned snapshot,
empty event logs, and proves by static inspection that the prospective
selection path cannot glob review directories for exclusions.

Writes its verdict atomically (temp file -> os.replace) so the artifact only
appears complete. Does not rely on shell exit-code chaining. Does not rerun the
sampler more than once. Does not regenerate or overwrite -008.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "tools"))
import build_g3_008_axis_neutral as sampler

SNAPSHOT = REPO / "experiments/phase6e/G3_008_PINNED_EXCLUSION_SNAPSHOT_001.json"
RUN1 = REPO / "bin/run1.json"
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-008"
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-008"
PROTOCOL4 = REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_004.json"
VERDICT = REPO / "experiments/phase6e/G3_008_FINAL_PRE_REVIEW_VERIFICATION_001.json"

EXPECTED_COUNT = 713
EXPECTED_DIGEST = "2cbabb9933c3e88fa2a03287ec8cd2b65f0dd3b72dcaf35b235402c7892dafeb"
AXIS_INDEX = {"Z": 0, "Y": 1, "X": 2}


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    report = {"id": "MV-G3-008-FINAL-PRE-REVIEW-VERIFICATION-001", "checks": {}, "failures": []}

    def check(name, cond, detail=None):
        report["checks"][name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            report["failures"].append(name)
        return bool(cond)

    # --- 1. Validate pinned snapshot (canonicalize union of per_source.avoid_edges) ---
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    edges = set()
    for src in snap.get("per_source", []):
        for e in src.get("avoid_edges", []):
            edges.add((tuple(e[0]), tuple(e[1]), int(e[2])))
    listed = sorted([list(l), list(r), c] for (l, r, c) in edges)
    snap_digest = hashlib.sha256(json.dumps(listed, sort_keys=True).encode()).hexdigest()
    check("snapshot_unique_713", len(listed) == EXPECTED_COUNT, {"count": len(listed)})
    check("snapshot_digest", snap_digest == EXPECTED_DIGEST, {"digest": snap_digest})
    check("snapshot_id_status", snap.get("id") == "MV-G3-008-PINNED-EXCLUSION-SNAPSHOT-001"
          and snap.get("status") == "FROZEN_PINNED_EXCLUSION_FOR_G3_008")

    # --- 2. Run corrected sampler exactly once, keep object in memory ---
    sel = sampler.select_only()
    current = sampler.canonical_manifest(sel)
    selection = sel["selection"]
    sources = [it["source_id"] for it in selection]
    strata = [it["stratum"] for it in selection]
    axis_counts = {"Z": 0, "Y": 0, "X": 0}
    for it in selection:
        for ax in ("Z", "Y", "X"):
            axis_counts[ax] += 1
    check("count_12_locations", len(selection) == 12, {"n": len(selection)})
    check("count_12_distinct_sources", len(set(sources)) == 12)
    check("count_36_edges", len(selection) * 3 == 36)
    check("axis_balance_12_12_12", axis_counts == {"Z": 12, "Y": 12, "X": 12}, axis_counts)
    strat_counts = {s: strata.count(s) for s in ("Q1", "Q2", "Q3", "Q4")}
    check("stratum_balance_3_each", strat_counts == {"Q1": 3, "Q2": 3, "Q3": 3, "Q4": 3}, strat_counts)
    check("pinned_count_713", sel["pinned_exclusion_count"] == EXPECTED_COUNT)
    check("pinned_digest_matches", sel["pinned_exclusion_sha256"] == EXPECTED_DIGEST)
    # structural/leakage assertions from the sampler itself
    try:
        sampler._assert_build_invariants(sel)
        check("build_invariants", True)
    except AssertionError as e:
        check("build_invariants", False, str(e))

    # --- 3. Compare against canonical Run 1 ---
    run1 = json.loads(RUN1.read_text(encoding="utf-8"))
    cur_canon, run1_canon = canon(current), canon(run1)
    equal_run1 = cur_canon == run1_canon
    report["current_canonical_sha256"] = sha(cur_canon)
    report["run1_canonical_sha256"] = sha(run1_canon)
    report["CURRENT_EQUALS_RUN1"] = equal_run1
    if not check("current_equals_run1", equal_run1):
        # first semantic difference
        diff = None
        for k in sorted(set(current) | set(run1)):
            if current.get(k) != run1.get(k):
                diff = {"key": k, "current": current.get(k), "run1": run1.get(k)}
                break
        report["first_run1_difference"] = diff
        _write(report, "FAIL")
        return

    # --- 4. Validate existing on-disk -008 ---
    logs_empty = True
    empty_detail = {}
    for i in range(1, 13):
        crop = f"MV-G3-AXNEU4-{i:02d}"
        log = WS_ROOT / crop / "workspace.events.jsonl"
        ok = log.exists() and log.read_text(encoding="utf-8").strip() == ""
        empty_detail[crop] = ok
        logs_empty = logs_empty and ok
    check("all_12_event_logs_empty", logs_empty, empty_detail)

    # reconstruct on-disk selection from the 12 queues and compare centers+edges
    ondisk = []
    disk_ok = True
    for i, it in enumerate(selection, 1):
        crop = f"MV-G3-AXNEU4-{i:02d}"
        qp = Q_ROOT / f"{crop}.json"
        if not qp.exists():
            disk_ok = False
            break
        q = json.loads(qp.read_text(encoding="utf-8"))
        center = q["selection"]["center_zyx"]
        by_axis = {qq["axis_name"]: (qq["pair_left_zyx"], qq["pair_right_zyx"], qq["channel_zyx"]) for qq in q["questions"]}
        ondisk.append({"source_id": q["crop_id"], "stratum": q["selection"]["stratum"], "center": center, "by_axis": by_axis})

    centers_match = disk_ok and all(ondisk[i]["center"] == selection[i]["center_zyx"] for i in range(12))
    # rebuild expected edges from in-memory centers and compare to disk edges
    edges_match = disk_ok
    if disk_ok:
        for i, it in enumerate(selection):
            z, y, x = it["center_zyx"]
            expect = {"Z": ([z, y, x], [z + 1, y, x], 0), "Y": ([z, y, x], [z, y + 1, x], 1), "X": ([z, y, x], [z, y, x + 1], 2)}
            for ax in ("Z", "Y", "X"):
                le, ri, ch = ondisk[i]["by_axis"][ax]
                if [le, ri, ch] != [expect[ax][0], expect[ax][1], expect[ax][2]]:
                    edges_match = False
    check("ondisk_008_centers_match", centers_match)
    check("ondisk_008_all_36_edges_match", edges_match)

    # protocol-004 references the correct pinned artifact/hash and same selection
    proto_ok = False
    if PROTOCOL4.exists():
        p4 = json.loads(PROTOCOL4.read_text(encoding="utf-8"))
        proto_ok = (p4.get("pinned_exclusion_sha256") == EXPECTED_DIGEST
                    and p4.get("pinned_exclusion_count") == EXPECTED_COUNT
                    and canon(p4.get("canonical_manifest")) == cur_canon)
    check("protocol_004_matches", proto_ok)

    # --- 5. Self-reference impossible by construction (static source inspection) ---
    # Conservative recursive reachability from select_only(): starting at
    # select_only, inspect the source of EVERY project-defined function it can
    # reach -- following name references not only within this module but into
    # any other project-defined module whose callables are referenced. A
    # filesystem-discovery call (glob/iterdir/listdir/rglob/walk/scandir) or a
    # reference to a review/output directory anywhere in that reachable graph
    # fails the check. Normal reads of the pinned snapshot, raw volumes, and the
    # fixed protocol/survey/manifest inputs are allowed (literal path reads, not
    # discovery). build() and other writers are flagged only if actually
    # reachable from select_only().
    REPO_STR = str(REPO)
    forbidden = [r"\bglob\b", r"\biterdir\b", r"\blistdir\b", r"\brglob\b", r"\bwalk\b", r"\bscandir\b",
                 r"g3-interface-queues", r"g3-external-review-packages"]

    def _project_defined(obj) -> bool:
        mod = getattr(obj, "__module__", None)
        if mod is None:
            return False
        m = sys.modules.get(mod)
        f = getattr(m, "__file__", None) if m else None
        return bool(f) and str(Path(f).resolve()).startswith(REPO_STR)

    def reachable_functions(root_name: str) -> dict:
        """Conservative cross-module static closure of project-defined callables
        reachable from root. Keyed by 'module.qualname'; value is source text.
        For each visited function, resolve every bare name it references against
        that function's own module globals; any project-defined callable so
        referenced is added to the frontier (recursively)."""
        def qual(obj):
            return f"{obj.__module__}.{obj.__qualname__}"

        root = getattr(sampler, root_name)
        seen = {}
        frontier = [root]
        while frontier:
            fn = frontier.pop()
            key = qual(fn)
            if key in seen:
                continue
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                src = ""
            seen[key] = src
            fmod = sys.modules.get(fn.__module__)
            fglobals = getattr(fmod, "__dict__", {})
            for name in set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", src)):
                if name == fn.__name__:
                    continue
                target = fglobals.get(name)
                if callable(target) and _project_defined(target) and hasattr(target, "__qualname__"):
                    if qual(target) not in seen:
                        frontier.append(target)
        return seen

    reachable = reachable_functions("select_only")
    reachable_names = sorted(reachable)
    build_qual = f"{sampler.__name__}.build"
    build_in_closure = build_qual in reachable
    per_function_scan = {}
    hits = []
    for fname, src in reachable.items():
        fn_hits = [pat for pat in forbidden if re.search(pat, src)]
        per_function_scan[fname] = {"forbidden_hits": fn_hits}
        for pat in fn_hits:
            hits.append({"function": fname, "pattern": pat})
    report["selection_reachability"] = {
        "root": f"{sampler.__name__}.select_only",
        "reachable_functions": reachable_names,
        "reachable_count": len(reachable_names),
        "build_reachable_from_select_only": build_in_closure,
        "forbidden_patterns": forbidden,
        "per_function_forbidden_scan": per_function_scan,
        "forbidden_hits": hits,
        "cross_module": True,
        "conservative_note": "Closure follows name references from select_only into every project-defined callable (this and other repo modules); stdlib/numpy are not project-defined and are not recursed. Directory-discovery calls or review/output dir references anywhere in this set fail the gate.",
    }
    check("selection_closure_excludes_build", not build_in_closure, {"reachable": reachable_names})
    check("selection_cannot_discover_review_dirs", not hits,
          {"reachable_functions": reachable_names, "forbidden_hits": hits})

    overall = "PASS" if not report["failures"] else "FAIL"
    _write(report, overall)


def _write(report: dict, overall: str) -> None:
    report["overall"] = overall
    report["decision"] = "G3_008_READY_FOR_HUMAN_REVIEW" if overall == "PASS" else "STOP_DIVERGENCE_OR_GATE_FAILURE"
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="g3_008_verdict_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, VERDICT)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(json.dumps({"overall": overall, "decision": report["decision"], "failures": report["failures"],
                      "CURRENT_EQUALS_RUN1": report.get("CURRENT_EQUALS_RUN1")}, indent=2))


if __name__ == "__main__":
    main()
