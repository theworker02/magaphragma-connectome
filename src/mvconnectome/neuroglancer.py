"""Portable Neuroglancer state exports for local connectome proofreading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .io import sha256_file, write_json_atomic


NEUROGLANCER_VIEWER = "https://neuroglancer-demo.appspot.com"


def _line_annotation(edge: dict[str, Any], nodes: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    """Convert one valid CATMAID edge into an editable Neuroglancer line."""
    parent = nodes.get(edge["parent_treenode_id"]); child = nodes.get(edge["child_treenode_id"])
    if parent is None or child is None:
        return None
    return {"id": f"{edge['mv_neuron_id']}:{edge['child_treenode_id']}", "type": "line",
            "pointA": [parent["x_nm"], parent["y_nm"], parent["z_nm"]], "pointB": [child["x_nm"], child["y_nm"], child["z_nm"]],
            "description": f"CATMAID source morphology; {edge['mv_neuron_id']}; {edge['coordinate_frame']}"}


def export_state(nodes_path: Path, edges_path: Path, output: Path, image_source: str | None = None, segmentation_source: str | None = None, ffn_validation: Path | None = None) -> Path:
    """Export source morphology as editable local line annotations.

    URLs are caller-supplied because Neuroglancer is client-side and requires
    actual HTTP-accessible volumes; the exporter never invents one for a local
    NPY cache.
    """
    nodes = pq.read_table(nodes_path).to_pylist(); edges = pq.read_table(edges_path).to_pylist()
    index = {int(node["source_treenode_id"]): node for node in nodes}
    annotations = [annotation for edge in edges if (annotation := _line_annotation(edge, index)) is not None]
    layers: list[dict[str, Any]] = []
    if image_source:
        layers.append({"type": "image", "name": "Megaphragma FIB-SEM", "source": image_source})
    if segmentation_source:
        layers.append({"type": "segmentation", "name": "FFN candidate segmentation", "source": segmentation_source, "segments": []})
    layers.append({"type": "annotation", "name": "CATMAID source skeletons", "annotationColor": "#e9b44c", "source": "local://annotations", "annotations": annotations})
    state: dict[str, Any] = {"kind": "NEUROGLANCER_STATE", "neuroglancer_viewer": NEUROGLANCER_VIEWER, "dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"]},
                             "layers": layers, "navigation": {"pose": {"position": {"voxelCoordinates": [0, 0, 0]}}},
                             "description": "Local proofreading state. CATMAID morphology is source-annotated; FFN layers are machine-only candidates until review.",
                             "provenance": {"nodes": {"path": str(nodes_path), "sha256": sha256_file(nodes_path)}, "edges": {"path": str(edges_path), "sha256": sha256_file(edges_path)}}}
    if ffn_validation:
        validation = json.loads(ffn_validation.read_text(encoding="utf-8")); state["ffn_validation"] = {"path": str(ffn_validation), "sha256": sha256_file(ffn_validation), "candidate_extension_gate": validation["candidate_extension_gate"]}
    output.parent.mkdir(parents=True, exist_ok=True); write_json_atomic(output, state)
    return output
