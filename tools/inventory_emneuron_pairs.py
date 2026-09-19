"""Inventory immutable EMNeuron raw/MaskIns pairs without repairing source data."""
from __future__ import annotations

import argparse, hashlib, json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def modality(name: str) -> str:
    lower = name.lower()
    if "fib" in lower or "hemi-brain" in lower: return "FIB_SEM"
    if "sstem" in lower or "cremi" in lower: return "ssTEM"
    if "atum" in lower: return "ATUM_SEM"
    if "sbem" in lower: return "SBEM"
    return "UNKNOWN"


def inspect(raw: Path, label: Path) -> dict:
    import numpy as np
    import tifffile
    with tifffile.TiffFile(raw) as image:
        series = image.series[0]
        raw_meta = {"shape": list(series.shape), "axes": series.axes, "dtype": str(series.dtype)}
    with tifffile.TiffFile(label) as image:
        series = image.series[0]
        label_meta = {"shape": list(series.shape), "axes": series.axes, "dtype": str(series.dtype)}
        labels = series.asarray()
    finite = bool(np.isfinite(labels).all())
    integer_valued = finite and bool(np.all(labels == np.floor(labels)))
    unique = np.unique(labels) if finite else np.array([])
    reasons = []
    if raw_meta["shape"] != label_meta["shape"]: reasons.append("SHAPE_MISMATCH")
    if raw_meta["axes"] != label_meta["axes"]: reasons.append("AXIS_MISMATCH")
    if not finite: reasons.append("NONFINITE_LABELS")
    if not integer_valued: reasons.append("NONINTEGER_LABEL_VALUES")
    if finite and labels.min() < 0: reasons.append("NEGATIVE_LABEL_ID")
    if unique.size <= 1: reasons.append("EMPTY_OR_SINGLE_VALUE_LABEL")
    return {
        "raw": {"path": str(raw.resolve()), "sha256": sha256(raw), **raw_meta},
        "label": {"path": str(label.resolve()), "sha256": sha256(label), **label_meta,
                  "value_min": float(labels.min()) if finite else None, "value_max": float(labels.max()) if finite else None,
                  "unique_value_count": int(unique.size), "background_id": 0 if finite and bool((labels == 0).any()) else None,
                  "integer_valued": integer_valued},
        "alignment": "ALIGNED" if not reasons else "QUARANTINED", "quarantine_reasons": reasons,
    }


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a=p.parse_args()
    records=[]
    for mask in sorted(a.root.rglob("*_MaskIns.tif")):
        raw=mask.with_name(mask.name.replace("_MaskIns.tif", ".tif"))
        dataset=mask.parent.name
        record={"dataset":dataset,"modality":modality(dataset),"source_pair_id":f"{dataset}/{raw.stem}"}
        if not raw.is_file():
            record.update({"alignment":"QUARANTINED","quarantine_reasons":["MISSING_RAW"],"raw":None,"label":{"path":str(mask.resolve()),"sha256":sha256(mask)}})
        else: record.update(inspect(raw,mask))
        records.append(record)
    report={"kind":"EMNEURON_SOURCE_PAIR_INVENTORY_V1","created_at":datetime.now(UTC).isoformat().replace('+00:00','Z'),"root":str(a.root.resolve()),"pairs":records,
            "summary":{"pairs":len(records),"aligned":sum(x["alignment"]=="ALIGNED" for x in records),"quarantined":sum(x["alignment"]=="QUARANTINED" for x in records),"fib_sem_aligned":sum(x["alignment"]=="ALIGNED" and x["modality"]=="FIB_SEM" for x in records)}}
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report["summary"]))
    return 0
if __name__=="__main__": raise SystemExit(main())
