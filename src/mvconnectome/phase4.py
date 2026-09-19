"""Quarantined Phase 4 materialisation of a CATMAID source-connectome.

All products here are local research artifacts.  This module deliberately has
no public API integration: source-data release rights remain unresolved.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .io import sha256_file, write_json_atomic

FRAME = {
    "id": "MV-FRAME-CATMAID-001", "units": "nm", "axis_order": "XYZ",
    "origin": "CATMAID stack local physical origin", "transform_status": "VERIFIED",
    "note": "CATMAID coordinates are retained verbatim; no cross-volume transform is asserted.",
}


def _load(path: Path, key: str) -> list[dict]:
    """Read a typed local-build collection, treating missing records as defects."""
    return json.loads(path.read_text(encoding="utf-8"))[key]


def _write_parquet(path: Path, rows: list[dict]) -> None:
    """Persist an inspectable tabular local artifact when optional Arrow is present."""
    table = pa.Table.from_pylist(rows) if rows else pa.table({})
    pq.write_table(table, path, compression="zstd")


def _stamp() -> str:
    """Return a UTC timestamp for quarantined local-build provenance."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def freeze_baseline(build: Path, baseline_id: str = "MV-LOCAL-LAMINA-BASELINE-001") -> Path:
    """Write a checksum-only immutable snapshot of the existing local build."""
    tables = ["neurons.json", "synapses.json", "connections.json", "unresolved_connectors.json", "manifest.json"]
    manifest = json.loads((build / "manifest.json").read_text())
    target = build / "baselines" / baseline_id
    target.mkdir(parents=True, exist_ok=True)
    record = {
        "id": baseline_id, "kind": "LOCAL_RESEARCH_BASELINE", "release_eligible": False,
        "generated_at": _stamp(), "code_commit": "UNKNOWN_NO_GIT_METADATA",
        "source_artifact_sha256": manifest["source_artifact_sha256"], "counts": manifest["counts"],
        "coordinate_frame": FRAME, "provenance_rules": ["source IDs retained", "no spatial identity inference", "no synthetic biological records", "rights quarantine"],
        "tables": {name: {"sha256": sha256_file(build / name), "bytes": (build / name).stat().st_size} for name in tables},
    }
    output = target / "manifest.json"
    if output.exists():
        prior = json.loads(output.read_text())
        if prior["tables"] != record["tables"]:
            raise ValueError(f"Refusing to overwrite immutable baseline {baseline_id}: table hashes differ")
        return output
    write_json_atomic(output, record)
    return output


def validate_build(build: Path) -> dict:
    """Validate the internal completeness of a local-only CATMAID build."""
    neurons = _load(build / "neurons.json", "neurons")
    synapses = _load(build / "synapses.json", "synapses")
    conns = _load(build / "connections.json", "connections")
    unresolved = _load(build / "unresolved_connectors.json", "connectors")
    errors: list[str] = []
    ids = [n["id"] for n in neurons]
    if len(ids) != len(set(ids)): errors.append("duplicate MV-N identity")
    neuron_ids = set(ids)
    syn_ids = {s["id"] for s in synapses}
    connector_ids = [s["source_connector_id"] for s in synapses] + [u["connector_id"] for u in unresolved]
    if len(connector_ids) != len(set(connector_ids)): errors.append("connector represented more than once")
    for syn in synapses:
        if syn["status"] != "SOURCE_ANNOTATED": errors.append(f"{syn['id']}: not SOURCE_ANNOTATED")
        if not syn["pre_neurons"] or not syn["post_neurons"]: errors.append(f"{syn['id']}: unresolved partner")
        if not set(syn["pre_neurons"] + syn["post_neurons"]) <= neuron_ids: errors.append(f"{syn['id']}: foreign neuron")
    expected: dict[tuple[str, str], set[str]] = defaultdict(set)
    for syn in synapses:
        for pre in syn["pre_neurons"]:
            for post in syn["post_neurons"]: expected[(pre, post)].add(syn["id"])
    actual = {(c["pre_neuron_id"], c["post_neuron_id"]): set(c["synapse_ids"]) for c in conns}
    if actual != expected: errors.append("connection table does not exactly reproduce synapse aggregation")
    for conn in conns:
        if not set(conn["synapse_ids"]) <= syn_ids: errors.append(f"{conn['id']}: missing supporting synapse")
        if conn["review_state"] not in {"UNREVIEWED", "EXPERT_REVIEWED", "LOCKED_RELEASE", "MACHINE_ONLY"}: errors.append(f"{conn['id']}: invalid review state")
    result = {"status": "passed" if not errors else "failed", "errors": errors,
              "counts": {"neurons": len(neurons), "synapses": len(synapses), "connections": len(conns), "unresolved_connectors": len(unresolved)}}
    write_json_atomic(build / "validation.json", result)
    if errors: raise ValueError("; ".join(errors))
    return result


def _fetch_one(skeleton_id: int, raw_dir: Path, endpoint: str, retries: int) -> dict:
    """Cache one source skeleton response with bounded retry evidence."""
    raw_path = raw_dir / f"{skeleton_id}.json"
    if raw_path.exists() and raw_path.stat().st_size:
        return {"source_skeleton_id": skeleton_id, "status": "cached", "path": raw_path.name,
                "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}
    url = endpoint.rstrip("/") + f"/{skeleton_id}/compact-detail"
    failure = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"Accept": "application/json", "User-Agent": "MVConnectome/0.1 local-research"})
            with urlopen(request, timeout=45) as response:
                content = response.read()
            json.loads(content)  # reject HTML and partial/non-JSON responses
            temp = raw_path.with_suffix(".partial")
            temp.write_bytes(content)
            temp.replace(raw_path)
            return {"source_skeleton_id": skeleton_id, "status": "retrieved", "path": raw_path.name,
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "url": url}
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if attempt < retries: time.sleep(min(12, 1.5 * (2 ** attempt)))
    return {"source_skeleton_id": skeleton_id, "status": "failed", "error": failure, "url": url}


def cache_morphologies(build: Path, cache: Path, endpoint: str, workers: int = 2, retries: int = 2) -> Path:
    """Resume-safe, politely concurrent download of CATMAID compact skeleton details."""
    neurons = _load(build / "neurons.json", "neurons")
    raw_dir = cache / "morphologies"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(n["source_skeleton_id"] for n in neurons)
    results = []
    # Two workers is deliberate: documented public read access is used without bulk hammering.
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as pool:
        futures = {pool.submit(_fetch_one, sid, raw_dir, endpoint, retries): sid for sid in ids}
        for future in as_completed(futures):
            results.append(future.result())
            time.sleep(0.08)
    results.sort(key=lambda x: x["source_skeleton_id"])
    for row in results:
        row.update({"retrieved_at": _stamp(), "endpoint_method": "GET", "response_format": "CATMAID compact-detail JSON"})
    manifest = {"kind": "QUARANTINED_SOURCE_MORPHOLOGY_CACHE", "release_eligible": False,
                "coordinate_frame": FRAME, "source_endpoint_template": endpoint.rstrip("/") + "/{skeleton_id}/compact-detail",
                "records": results, "summary": dict(Counter(r["status"] for r in results))}
    write_json_atomic(cache / "morphology_source_manifest.json", manifest)
    _write_parquet(cache / "morphology_source_manifest.parquet", results)
    return cache / "morphology_source_manifest.json"


def materialize_physical(build: Path, cache: Path, output: Path) -> dict:
    """Normalize cached source morphology without modifying any raw response."""
    neurons = _load(build / "neurons.json", "neurons")
    synapses = _load(build / "synapses.json", "synapses")
    mv_by_source = {n["source_skeleton_id"]: n["id"] for n in neurons}
    nodes: list[dict] = []; edges: list[dict] = []; metrics = {}; failures = []
    raw_dir = cache / "morphologies"
    for sid, mv_id in sorted(mv_by_source.items()):
        path = raw_dir / f"{sid}.json"
        if not path.exists(): failures.append({"mv_neuron_id": mv_id, "source_skeleton_id": sid, "reason": "MORPHOLOGY_NOT_CACHED"}); continue
        try:
            payload = json.loads(path.read_text())
            records = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], list) else payload
            # CATMAID compact skeleton detail rows: id, parent_id, creator, x_nm, y_nm, z_nm, radius, confidence.
            by_id = {int(row[0]): row for row in records if len(row) >= 8}
            if not by_id: raise ValueError("no compact-detail node rows")
            source_hash = sha256_file(path)
            node_rows = []
            for row in by_id.values():
                node = {"mv_neuron_id": mv_id, "source_skeleton_id": sid, "source_treenode_id": int(row[0]),
                        "parent_treenode_id": None if row[1] is None or int(row[1]) < 0 else int(row[1]),
                        "x_nm": float(row[3]), "y_nm": float(row[4]), "z_nm": float(row[5]), "radius_nm": float(row[6]),
                        "confidence": row[7], "source_sha256": source_hash, "coordinate_frame": FRAME["id"], "provenance": "CATMAID compact-detail"}
                node_rows.append(node); nodes.append(node)
            total = 0.0; children = Counter(); edge_count = 0
            for node in node_rows:
                parent = node["parent_treenode_id"]
                if parent is None: continue
                if parent not in by_id: continue
                pr = by_id[parent]; length = math.dist((node["x_nm"], node["y_nm"], node["z_nm"]), (float(pr[3]), float(pr[4]), float(pr[5])))
                edges.append({"mv_neuron_id": mv_id, "source_skeleton_id": sid, "child_treenode_id": node["source_treenode_id"], "parent_treenode_id": parent, "length_nm": length, "source_sha256": node["source_sha256"], "coordinate_frame": FRAME["id"]})
                total += length; edge_count += 1; children[parent] += 1
            xyz = [(n["x_nm"], n["y_nm"], n["z_nm"]) for n in node_rows]
            metrics[mv_id] = {"node_count": len(node_rows), "edge_count": edge_count, "branch_point_count": sum(c > 1 for c in children.values()),
                              "endpoint_count": sum(n["source_treenode_id"] not in children for n in node_rows), "cable_length_nm": total,
                              "bounds_nm_xyz": [[min(p[i] for p in xyz), max(p[i] for p in xyz)] for i in range(3)]}
        except (ValueError, TypeError, IndexError, json.JSONDecodeError) as exc: failures.append({"mv_neuron_id": mv_id, "source_skeleton_id": sid, "reason": str(exc)})
    output.mkdir(parents=True, exist_ok=True)
    _write_parquet(output / "skeleton_nodes.parquet", nodes); _write_parquet(output / "skeleton_edges.parquet", edges)
    # Upgrade only those source records whose independently cached raw morphology parsed successfully.
    for neuron in neurons:
        if neuron["id"] in metrics:
            neuron["morphology"] = "SOURCE_RECONSTRUCTED_PHYSICAL"
            neuron["morphology_metrics"] = metrics[neuron["id"]]
            neuron["morphology_cache"] = f"morphologies/{neuron['source_skeleton_id']}.json"
    write_json_atomic(output / "physical_neurons.json", {"neurons": neurons, "failures": failures, "coordinate_frame": FRAME})
    # Project source synapses in physical CATMAID coordinates exactly as exported.
    projected = [{**s, "coordinate_frame": FRAME["id"], "connector_xyz_nm": s["coordinates_nm_xyz"]} for s in synapses]
    _write_parquet(output / "synapses.parquet", projected)
    return {"neurons": len(metrics), "nodes": len(nodes), "edges": len(edges), "failures": failures}


def build_graph_products(build: Path, output: Path) -> dict:
    """Create local graph derivatives without reclassifying source biology as Vigilia."""
    neurons = _load(build / "neurons.json", "neurons"); conns = _load(build / "connections.json", "connections")
    ids = [n["id"] for n in neurons]; index = {n: i for i, n in enumerate(ids)}; matrix = np.zeros((len(ids), len(ids)), dtype=np.uint32)
    for c in conns: matrix[index[c["pre_neuron_id"]], index[c["post_neuron_id"]]] = len(c["synapse_ids"])
    np.savez_compressed(output / "connectivity_matrix.npz", source=matrix, neuron_ids=np.array(ids))
    with (output / "connectivity_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["pre\\post", *ids]); writer.writerows([[ids[i], *row] for i, row in enumerate(matrix)])
    graphml = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"><graph edgedefault=\"directed\">"]
    graphml += [f'<node id="{n}"/>' for n in ids]
    graphml += [f'<edge id="{c["id"]}" source="{c["pre_neuron_id"]}" target="{c["post_neuron_id"]}"><data key="synapse_count">{len(c["synapse_ids"])}</data></edge>' for c in conns]
    graphml.append("</graph></graphml>"); (output / "connectivity.graphml").write_text("\n".join(graphml), encoding="utf-8")
    return {"nodes": len(ids), "edges": len(conns), "matrix_nonzero": int(np.count_nonzero(matrix))}


def status_report(build: Path, output: Path, physical: dict, graph: dict) -> Path:
    """Write a transparent local-build status report from verified artifact counts."""
    manifest = json.loads((build / "manifest.json").read_text()); c = manifest["counts"]
    text = "# Local Connectome Status\n\n" + "\n".join([
        "**LOCAL RESEARCH BUILD — NOT FOR PUBLIC REDISTRIBUTION**", "",
        f"- Source MV-N: {c['neurons']}", f"- Morphologies cached/physical: {physical['neurons']}",
        f"- Morphology nodes: {physical['nodes']}", f"- Morphology edges: {physical['edges']}",
        f"- Connectors: {c['connectors']}", f"- Resolved source synapses: {c['synapses']}",
        "- Newly resolved connectors: 0 (no spatial identity inference)", f"- Unresolved connectors: {c['unresolved_connectors']}",
        f"- MV-CONN / graph edges: {c['connections']} / {graph['edges']}", f"- Graph nodes: {graph['nodes']}",
        "- Fragments / candidate project neurons / project extensions: 0 / 0 / 0", "- EM-mapped neurons: 0 (no verified CATMAID-to-EM transform registered)",
        "- Regions: 0 (no source-region annotation imported)", "- Rights/release blockers: source-data export and derivative release scope unresolved.",
    ]) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text, encoding="utf-8"); return output


def create_local_release(build: Path, rows_path: Path) -> Path:
    """Package derived local products only under a hard non-redistribution marker."""
    release = build / "MV-LOCAL-CONNECTOME-0.1"
    release.mkdir(parents=True, exist_ok=True)
    for name in ("skeleton_nodes.parquet", "skeleton_edges.parquet", "synapses.parquet", "connectivity.graphml", "connectivity_matrix.npz", "connectivity_matrix.csv", "unresolved_connectors.json", "physical_neurons.json"):
        src = build / name
        if src.exists(): (release / name).write_bytes(src.read_bytes())
    neurons = _load(build / "neurons.json", "neurons"); synapses = _load(build / "synapses.json", "synapses"); conns = _load(build / "connections.json", "connections")
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    _write_parquet(release / "neurons.parquet", neurons); _write_parquet(release / "connections.parquet", conns)
    _write_parquet(release / "connectors.parquet", [{"source_connector_id": str(r[0]), "x_nm": r[1], "y_nm": r[2], "z_nm": r[3], "source_skeleton_id": r[4], "source_treenode_id": r[7], "relation_id": r[10], "source_artifact_sha256": sha256_file(rows_path)} for r in rows])
    _write_parquet(release / "unresolved_connectors.parquet", _load(build / "unresolved_connectors.json", "connectors"))
    checksums = []
    for path in sorted(release.glob("*")):
        if path.is_file(): checksums.append(f"{sha256_file(path)}  {path.name}")
    (release / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    known = "# Known issues\n\nThis is a quarantined local research build, not a public release. CATMAID source-data/export and derivative-release rights remain unresolved. No cross-volume EM transform, region labels, project extensions, or candidate neurons are asserted.\n"
    (release / "KNOWN_ISSUES.md").write_text(known, encoding="utf-8")
    write_json_atomic(release / "manifest.json", {"id": "MV-LOCAL-CONNECTOME-0.1", "kind": "LOCAL_RESEARCH_BUILD", "release_eligible": False, "rights_quarantine": True, "source_artifact_sha256": sha256_file(rows_path), "counts": {"neurons":len(neurons), "synapses":len(synapses), "connections":len(conns), "connector_link_rows":len(rows)}, "coordinate_frame": FRAME})
    return release
