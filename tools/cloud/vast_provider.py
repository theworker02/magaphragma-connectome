"""Vast.ai REST provider for Affinity (no hardcoded marketplace offers)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import CloudProvider, Instance, Offer

# Repo root: tools/cloud/vast_provider.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_repo_dotenv(repo: Path | None = None) -> None:
    """Load KEY=VALUE lines from repo `.env` into os.environ if unset.

    Does not overwrite existing environment variables. Never logs values.
    """
    env_path = (repo or _REPO_ROOT) / ".env"
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def _resolve_vast_api_key(explicit: str | None = None) -> str:
    """Resolve VAST_API_KEY from arg, env, repo `.env`, or common local key files."""
    load_repo_dotenv()
    key = (explicit or os.environ.get("VAST_API_KEY") or "").strip()
    if key:
        return key
    for candidate in (
        Path.home() / ".vast_api_key",
        Path.home() / ".config" / "vastai" / "vast_api_key",
    ):
        if candidate.is_file():
            try:
                key = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if key:
                os.environ["VAST_API_KEY"] = key
                return key
    return ""


# Load `.env` early so module-level defaults below can see budget/filter vars.
load_repo_dotenv()

API_BASE = os.environ.get("VAST_API_BASE", "https://console.vast.ai/api/v0")
# Instance list/get/destroy moved to v1 (v0 /instances/ returns HTTP 410).
API_BASE_INSTANCES = os.environ.get(
    "VAST_API_BASE_INSTANCES", "https://console.vast.ai/api/v1"
)

# Broad Affinity-compatible search: cheap NVIDIA with enough VRAM, not prestige SKUs.
DEFAULT_MIN_VRAM_GB = float(os.environ.get("AFFINITY_MIN_VRAM_GB", "6"))
DEFAULT_MIN_RELIABILITY = float(os.environ.get("AFFINITY_MIN_RELIABILITY", "0.98"))
DEFAULT_MIN_DISK_GB = float(os.environ.get("AFFINITY_MIN_DISK_GB", "32"))
DEFAULT_MIN_CPU_RAM_GB = float(os.environ.get("AFFINITY_MIN_CPU_RAM_GB", "8"))


class VastAuthError(RuntimeError):
    pass


class VastProvider(CloudProvider):
    name = "vast"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = _resolve_vast_api_key(api_key)
        if not self.api_key:
            raise VastAuthError(
                "VAST_API_KEY is required. Set it in the repo `.env`, environment, "
                "or pass api_key=. Never commit credentials."
            )

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout: float = 60.0,
        base: str | None = None,
    ) -> Any:
        root = (base or API_BASE).rstrip("/")
        url = f"{root}/{path.lstrip('/')}"
        data = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vast API {method} {path} -> HTTP {e.code}: {detail}") from e

    def _instances_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout: float = 60.0,
    ) -> Any:
        return self._request(method, path, body, timeout=timeout, base=API_BASE_INSTANCES)

    def search_offers(
        self,
        *,
        offer_type: str = "on-demand",
        min_vram_gb: float = DEFAULT_MIN_VRAM_GB,
        min_reliability: float = DEFAULT_MIN_RELIABILITY,
        min_disk_gb: float = DEFAULT_MIN_DISK_GB,
        min_cpu_ram_gb: float = DEFAULT_MIN_CPU_RAM_GB,
        verified_only: bool = True,
        num_gpus: int = 1,
        limit: int = 100,
        max_dph: float | None = None,
        gpu_name: str | None = None,
        order_by_price: bool = True,
    ) -> list[Offer]:
        """Query live marketplace offers. Do not hardcode offer IDs."""
        q: dict[str, Any] = {
            "verified": {"eq": True} if verified_only else {"eq": False},
            "rentable": {"eq": True},
            "num_gpus": {"eq": int(num_gpus)},
            "gpu_ram": {"gte": float(min_vram_gb) * 1024.0},  # Vast uses MiB
            "reliability": {"gte": float(min_reliability)},
            "disk_space": {"gte": float(min_disk_gb)},
            "cpu_ram": {"gte": float(min_cpu_ram_gb) * 1024.0},
            "type": offer_type,
            "limit": int(limit),
            "order": [["dph_total", "asc"]] if order_by_price else [["dlperf_per_dphtotal", "desc"]],
        }
        if max_dph is not None:
            q["dph_total"] = {"lte": float(max_dph)}
        if gpu_name:
            q["gpu_name"] = {"eq": gpu_name}
        # Prefer machines with direct ports for SSH worker deploy
        q["direct_port_count"] = {"gte": 1}

        payload = self._request("POST", "bundles/", q)
        offers_raw = payload.get("offers") or payload.get("bundles") or []
        if isinstance(payload, list):
            offers_raw = payload

        out: list[Offer] = []
        for row in offers_raw:
            if not isinstance(row, dict):
                continue
            out.append(self._parse_offer(row, interruptible=(offer_type == "bid")))
        return out

    def search_offers_both_markets(self, **kwargs: Any) -> list[Offer]:
        """Search on-demand and interruptible; tag interruptible flag."""
        od = self.search_offers(offer_type="on-demand", **kwargs)
        try:
            bid = self.search_offers(offer_type="bid", **kwargs)
        except Exception:
            bid = []
        # Deduplicate by offer id preferring lower dph
        by_id: dict[int, Offer] = {}
        for o in od + bid:
            prev = by_id.get(o.offer_id)
            if prev is None or o.dph_total < prev.dph_total:
                by_id[o.offer_id] = o
        return list(by_id.values())

    def _parse_offer(self, row: dict[str, Any], *, interruptible: bool) -> Offer:
        gpu_ram = row.get("gpu_ram") or row.get("gpu_ram_gb") or 0
        # Vast typically reports gpu_ram in MiB
        vram_gb = float(gpu_ram) / 1024.0 if float(gpu_ram) > 64 else float(gpu_ram)
        cpu_ram = row.get("cpu_ram") or 0
        cpu_ram_gb = float(cpu_ram) / 1024.0 if float(cpu_ram) > 256 else float(cpu_ram)
        return Offer(
            offer_id=int(row["id"]),
            gpu_name=str(row.get("gpu_name") or row.get("gpu_name_full") or "unknown"),
            num_gpus=int(row.get("num_gpus") or 1),
            vram_gb=vram_gb,
            dph_total=float(row.get("dph_total") or row.get("dph_base") or 0.0),
            dph_base=float(row["dph_base"]) if row.get("dph_base") is not None else None,
            storage_cost_per_gb_hour=(
                float(row["storage_cost"]) if row.get("storage_cost") is not None else None
            ),
            inet_up_cost=float(row["inet_up_cost"]) if row.get("inet_up_cost") is not None else None,
            inet_down_cost=float(row["inet_down_cost"]) if row.get("inet_down_cost") is not None else None,
            reliability=float(row["reliability"]) if row.get("reliability") is not None else None,
            verified=bool(row["verified"]) if row.get("verified") is not None else None,
            rentable=bool(row["rentable"]) if row.get("rentable") is not None else None,
            interruptible=interruptible or bool(row.get("is_bid") or row.get("type") == "bid"),
            cuda_max_good=float(row["cuda_max_good"]) if row.get("cuda_max_good") is not None else None,
            cpu_cores=float(row["cpu_cores"]) if row.get("cpu_cores") is not None else None,
            cpu_ram_gb=cpu_ram_gb,
            disk_gb=float(row["disk_space"]) if row.get("disk_space") is not None else None,
            geolocation=str(row.get("geolocation") or "") or None,
            machine_id=int(row["machine_id"]) if row.get("machine_id") is not None else None,
            raw=row,
        )

    def rank_offers(
        self,
        offers: list[Offer],
        *,
        tiles_per_second_by_gpu: dict[str, float] | None = None,
        default_tiles_per_second: float = 18.9,
        remaining_chunks: int = 0,
        storage_gb: float = 40.0,
        bandwidth_per_chunk_usd: float = 0.0,
    ) -> list[Offer]:
        """
        Rank by projected remaining cost using measured TPS when available,
        else baseline TPS. Primary key: projected_remaining_cost ascending.
        """
        from .pricing import estimate_chunk_economics

        scored: list[tuple[float, Offer]] = []
        for o in offers:
            tps = default_tiles_per_second
            if tiles_per_second_by_gpu:
                # exact then fuzzy contains
                if o.gpu_name in tiles_per_second_by_gpu:
                    tps = tiles_per_second_by_gpu[o.gpu_name]
                else:
                    for k, v in tiles_per_second_by_gpu.items():
                        if k.lower() in o.gpu_name.lower() or o.gpu_name.lower() in k.lower():
                            tps = v
                            break
            storage_hourly = 0.0
            if o.storage_cost_per_gb_hour:
                storage_hourly = o.storage_cost_per_gb_hour * storage_gb
            model = estimate_chunk_economics(
                tiles_per_second=tps,
                gpu_hourly_usd=o.dph_total,
                remaining_chunks=remaining_chunks,
                storage_hourly_usd=storage_hourly,
                bandwidth_per_chunk_usd=bandwidth_per_chunk_usd,
            )
            # stash for callers
            o.raw = {
                **o.raw,
                "_rank_cost_model": model.to_dict(),
                "_projected_remaining_cost_usd": model.projected_remaining_cost_usd,
                "_effective_cost_per_chunk": model.effective_cost_per_chunk,
                "_assumed_tiles_per_second": tps,
            }
            scored.append((model.projected_remaining_cost_usd, o))
        scored.sort(key=lambda t: (t[0], t[1].dph_total, -t[1].reliability if t[1].reliability else 0))
        return [o for _, o in scored]

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        disk: float = 40.0,
        onstart: str | None = None,
        label: str | None = None,
        price: float | None = None,
        ssh: bool = True,
        env: dict[str, str] | None = None,
    ) -> Instance:
        body: dict[str, Any] = {
            "image": image,
            "disk": float(disk),
        }
        if onstart:
            body["onstart"] = onstart
        if label:
            body["label"] = label
        if price is not None:
            body["price"] = float(price)
        if env:
            # Vast accepts env as string "-e KEY=VAL ..." or dict depending on API version
            body["env"] = " ".join(f"-e {k}={v}" for k, v in env.items())
        # Prefer SSH
        if ssh:
            body["runtype"] = "ssh"
        payload = self._request("PUT", f"asks/{int(offer_id)}/", body)
        iid = int(payload.get("new_contract") or payload.get("id") or 0)
        if not iid:
            raise RuntimeError(f"create_instance failed: {payload}")
        return Instance(
            instance_id=iid,
            status="created",
            actual_status=None,
            label=label,
            raw=payload if isinstance(payload, dict) else {"payload": payload},
        )

    def get_instance(self, instance_id: int) -> Instance:
        payload = self._instances_request("GET", f"instances/{int(instance_id)}/")
        row = payload.get("instances") if isinstance(payload, dict) else None
        if isinstance(row, dict) and "actual_status" in row:
            inst = row
        elif isinstance(row, list) and row:
            inst = row[0]
        elif isinstance(payload, dict) and "actual_status" in payload:
            inst = payload
        else:
            inst = payload if isinstance(payload, dict) else {}
        return self._parse_instance(inst, fallback_id=instance_id)

    def list_instances(self) -> list[Instance]:
        payload = self._instances_request("GET", "instances/")
        rows = payload.get("instances") if isinstance(payload, dict) else payload
        if isinstance(rows, dict):
            # sometimes keyed by id
            rows = list(rows.values()) if rows and isinstance(next(iter(rows.values()), None), dict) else [rows]
        out: list[Instance] = []
        for row in rows or []:
            if isinstance(row, dict):
                out.append(self._parse_instance(row))
        return out

    def _parse_instance(self, row: dict[str, Any], fallback_id: int | None = None) -> Instance:
        iid = int(row.get("id") or row.get("instance_id") or fallback_id or 0)
        ssh_host = str(row.get("ssh_host") or "") or None
        ssh_port = int(row["ssh_port"]) if row.get("ssh_port") is not None else None
        # Prefer direct public IP + mapped SSH port when available (faster than sshN.vast.ai).
        public_ip = str(row.get("public_ipaddr") or "").strip()
        ports = row.get("ports") or {}
        mapped_ssh = None
        if isinstance(ports, dict):
            for entry in ports.get("22/tcp") or []:
                if isinstance(entry, dict) and entry.get("HostPort"):
                    mapped_ssh = int(entry["HostPort"])
                    break
        if public_ip and mapped_ssh:
            ssh_host = public_ip
            ssh_port = mapped_ssh
        return Instance(
            instance_id=iid,
            status=str(row.get("status") or "") or None,
            actual_status=str(row.get("actual_status") or "") or None,
            gpu_name=str(row.get("gpu_name") or "") or None,
            dph_total=float(row["dph_total"]) if row.get("dph_total") is not None else None,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            label=str(row.get("label") or "") or None,
            raw=row,
        )

    def destroy_instance(self, instance_id: int) -> dict[str, Any]:
        # Prefer v1; fall back to v0 if needed.
        try:
            return self._instances_request("DELETE", f"instances/{int(instance_id)}/")
        except RuntimeError:
            return self._request("DELETE", f"instances/{int(instance_id)}/")

    def stop_instance(self, instance_id: int) -> dict[str, Any]:
        """Stop compute; disk charges continue — prefer destroy for Affinity."""
        try:
            return self._instances_request(
                "PUT", f"instances/{int(instance_id)}/", {"state": "stopped"}
            )
        except RuntimeError:
            return self._request("PUT", f"instances/{int(instance_id)}/", {"state": "stopped"})

    def estimate_cost(self, *, hours: float, dph: float, storage_gb: float = 0.0) -> dict[str, float]:
        compute = float(hours) * float(dph)
        # storage often already in dph_total; keep separate line for transparency
        return {
            "hours": float(hours),
            "dph": float(dph),
            "compute_usd": compute,
            "storage_gb": float(storage_gb),
            "total_usd": compute,
        }

    def fleet_status(self) -> dict[str, Any]:
        instances = self.list_instances()
        running = [i for i in instances if (i.actual_status or "").lower() == "running"]
        loading = [i for i in instances if (i.actual_status or "").lower() in {"loading", "none", ""}]
        hourly = sum(float(i.dph_total or 0.0) for i in instances)
        return {
            "provider": self.name,
            "n_instances": len(instances),
            "n_running": len(running),
            "n_loading": len(loading),
            "fleet_hourly_usd": hourly,
            "instances": [i.to_dict() for i in instances],
        }
