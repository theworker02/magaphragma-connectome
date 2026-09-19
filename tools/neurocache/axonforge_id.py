"""Identity helper for AxonForge tile inference results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .identity import ScientificIdentity
from .cache import NeuroCache


def tile_inference_identity(
    *,
    volume_hash: str,
    tile_key: str,
    box: Mapping[str, int] | None = None,
    model: str = "axonforge",
    checkpoint: str = "default",
    precision: str = "fp32",
    tile_dimensions: tuple[int, ...] = (32, 64, 64),
    halo: int | tuple[int, ...] = 8,
    algorithm_version: str = "1.0",
    extra: Mapping[str, Any] | None = None,
) -> ScientificIdentity:
    box = dict(box or {})
    coords = (
        float(box.get("z0", 0)),
        float(box.get("y0", 0)),
        float(box.get("x0", 0)),
        float(box.get("z1", tile_dimensions[0])),
        float(box.get("y1", tile_dimensions[1])),
        float(box.get("x1", tile_dimensions[2])),
    )
    sci = {
        "tile_key": tile_key,
        "stage": "tile_inference",
        **dict(extra or {}),
    }
    return ScientificIdentity(
        source_volume_hash=str(volume_hash),
        coordinates=coords,
        resolution=(1.0, 1.0, 1.0),
        preprocessing={"normalize": "uint8_scale"},
        model=model,
        checkpoint=checkpoint,
        precision=precision,
        tile_dimensions=tuple(int(x) for x in tile_dimensions),
        halo=halo,
        scientific_config=sci,
        algorithm_version=algorithm_version,
    )


def put_tile_result(
    identity: ScientificIdentity,
    payload: Any,
    *,
    cache: NeuroCache | None = None,
) -> dict[str, Any]:
    cache = cache or NeuroCache()
    # Optional ArtifactVet on identity dict
    try:
        from _core.artifactvet import ArtifactVet
        tmp = Path(cache.root) / ".vet_identity.json"
        tmp.write_text(json.dumps(identity.to_dict(), indent=2) + "\\n", encoding="utf-8")
        ArtifactVet().vet_json(tmp, required_keys=("source_volume_hash", "model", "checkpoint")).raise_if_failed()
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return cache.put(identity, payload)


def lookup_tile_result(identity: ScientificIdentity, *, cache: NeuroCache | None = None) -> Any:
    cache = cache or NeuroCache()
    return cache.get(identity)


def demo_roundtrip(tile_key: str = "T-Z000-Y000-X000") -> dict[str, Any]:
    vol_hash = hashlib.sha256(b"regionbench-demo-volume").hexdigest()
    ident = tile_inference_identity(volume_hash=vol_hash, tile_key=tile_key, box={"z0": 0, "y0": 0, "x0": 0, "z1": 32, "y1": 64, "x1": 64})
    payload = {"tile_key": tile_key, "status": "VALIDATED", "scores": {"mean": 0.42}}
    put_manifest = put_tile_result(ident, payload)
    hit = lookup_tile_result(ident)
    return {
        "identity_hash": ident.identity_hash(),
        "put": put_manifest,
        "lookup_hit": hit is not None,
        "payload": hit,
    }
