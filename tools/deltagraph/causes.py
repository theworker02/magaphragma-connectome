"""Map structural deltas to likely pipeline causes."""
from __future__ import annotations

from typing import Any

from .diff import DeltaResult

CAUSE_KEYS = (
    "checkpoint_changed",
    "segmentation_config_changed",
    "boundary_reconciliation_changed",
    "synapse_threshold_changed",
    "source_data_changed",
)


def infer(
    delta: DeltaResult | dict[str, Any],
    meta_a: dict[str, Any] | None = None,
    meta_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer likely pipeline causes from a structural delta + build metadata."""
    d = delta.to_dict() if isinstance(delta, DeltaResult) else dict(delta)
    meta_a = dict(meta_a or {})
    meta_b = dict(meta_b or {})

    evidence: dict[str, list[str]] = {k: [] for k in CAUSE_KEYS}
    scores: dict[str, float] = {k: 0.0 for k in CAUSE_KEYS}

    if meta_a.get("checkpoint") != meta_b.get("checkpoint") and (
        meta_a.get("checkpoint") is not None or meta_b.get("checkpoint") is not None
    ):
        scores["checkpoint_changed"] += 3.0
        evidence["checkpoint_changed"].append("meta.checkpoint differs")

    for k in ("segmentation_config", "seg_config", "watershed", "affinity_threshold"):
        if meta_a.get(k) != meta_b.get(k) and (k in meta_a or k in meta_b):
            scores["segmentation_config_changed"] += 2.5
            evidence["segmentation_config_changed"].append(f"meta.{k} differs")

    for k in ("boundary_reconciliation", "boundary_policy", "merge_threshold"):
        if meta_a.get(k) != meta_b.get(k) and (k in meta_a or k in meta_b):
            scores["boundary_reconciliation_changed"] += 2.5
            evidence["boundary_reconciliation_changed"].append(f"meta.{k} differs")

    for k in ("synapse_threshold", "synapse_conf_threshold", "detection_threshold"):
        if meta_a.get(k) != meta_b.get(k) and (k in meta_a or k in meta_b):
            scores["synapse_threshold_changed"] += 2.5
            evidence["synapse_threshold_changed"].append(f"meta.{k} differs")

    for k in ("source_volume_hash", "volume_hash", "source_uri"):
        if meta_a.get(k) != meta_b.get(k) and (k in meta_a or k in meta_b):
            scores["source_data_changed"] += 3.0
            evidence["source_data_changed"].append(f"meta.{k} differs")

    splits = d.get("objects_split") or []
    merges = d.get("objects_merged") or []
    if splits or merges:
        scores["boundary_reconciliation_changed"] += 1.5 + 0.1 * (len(splits) + len(merges))
        evidence["boundary_reconciliation_changed"].append(
            f"split={len(splits)} merge={len(merges)}"
        )
        scores["segmentation_config_changed"] += 1.0
        evidence["segmentation_config_changed"].append("object topology changed")

    syn_added = d.get("synapses_added") or []
    syn_removed = d.get("synapses_removed") or []
    conf_changes = d.get("confidence_changes") or []
    syn_conf = [c for c in conf_changes if c.get("kind") == "synapse"]
    if syn_added or syn_removed or syn_conf:
        scores["synapse_threshold_changed"] += 1.2 + 0.05 * (len(syn_added) + len(syn_removed))
        evidence["synapse_threshold_changed"].append(
            f"synapses +{len(syn_added)} -{len(syn_removed)} conf={len(syn_conf)}"
        )

    spatial = d.get("spatial_changes") or []
    if spatial:
        scores["source_data_changed"] += 0.8 + 0.05 * len(spatial)
        evidence["source_data_changed"].append(f"spatial_shifts={len(spatial)}")
        scores["checkpoint_changed"] += 0.5
        evidence["checkpoint_changed"].append("spatial drift on shared neurons")

    neurons_added = d.get("neurons_added") or []
    neurons_removed = d.get("neurons_removed") or []
    if (len(neurons_added) + len(neurons_removed)) >= 3 and not (splits or merges):
        scores["segmentation_config_changed"] += 1.0
        evidence["segmentation_config_changed"].append("large neuron set churn")

    edges = d.get("edges_changed") or {}
    if (edges.get("added") or edges.get("removed")) and (syn_added or syn_removed):
        scores["synapse_threshold_changed"] += 0.5

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    likely = [
        {"cause": name, "score": round(score, 4), "evidence": evidence[name]}
        for name, score in ranked
        if score > 0.0
    ]
    return {
        "likely_causes": likely,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "primary": likely[0]["cause"] if likely else None,
    }
