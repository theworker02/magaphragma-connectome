"""Hard budget kill-switch for Affinity cloud spend."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BudgetRefused(RuntimeError):
    """Raised when a spend/provision action would violate the hard budget."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class BudgetConfig:
    hard_limit_usd: float = 160.0
    target_usd: float = 100.0
    benchmark_usd: float = 5.0
    warn_frac: float = 0.70
    stop_provision_frac: float = 0.85
    drain_frac: float = 0.95

    @classmethod
    def from_env(cls) -> "BudgetConfig":
        return cls(
            hard_limit_usd=_env_float("AFFINITY_CLOUD_BUDGET_USD", 160.0),
            target_usd=_env_float("AFFINITY_TARGET_BUDGET_USD", 100.0),
            benchmark_usd=_env_float("AFFINITY_BENCHMARK_BUDGET_USD", 5.0),
        )


class BudgetLedger:
    """
    Conservative local ledger for cloud spend.

    Tracks:
      - spent_usd: confirmed/observed spend
      - committed_usd: projected cost of currently rented capacity through expected drain
      - reserved_benchmark_usd: reserved for $5 qualification campaign

    Hard rule: spent + committed + new_commitment must not exceed hard_limit_usd.
    No silent override.
    """

    def __init__(self, path: Path, config: BudgetConfig | None = None) -> None:
        self.path = path
        self.config = config or BudgetConfig.from_env()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(
                {
                    "schema_version": 1,
                    "created_at": _now(),
                    "updated_at": _now(),
                    "spent_usd": 0.0,
                    "committed_usd": 0.0,
                    "reserved_benchmark_usd": 0.0,
                    "events": [],
                    "config": asdict(self.config),
                }
            )

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        data["updated_at"] = _now()
        data["config"] = asdict(self.config)
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            spent = float(data["spent_usd"])
            committed = float(data["committed_usd"])
            reserved = float(data.get("reserved_benchmark_usd", 0.0))
            hard = self.config.hard_limit_usd
            used = spent + committed + reserved
            remaining = hard - used
            frac = used / hard if hard > 0 else 1.0
            return {
                "spent_usd": spent,
                "committed_usd": committed,
                "reserved_benchmark_usd": reserved,
                "used_usd": used,
                "remaining_usd": remaining,
                "hard_limit_usd": hard,
                "target_usd": self.config.target_usd,
                "benchmark_usd": self.config.benchmark_usd,
                "used_fraction": frac,
                "phase": self._phase(frac),
                "may_provision": frac < self.config.stop_provision_frac and remaining > 0,
                "must_drain_fleet": frac >= self.config.drain_frac,
                "absolute_block": remaining <= 0 or frac >= 1.0,
            }

    def _phase(self, frac: float) -> str:
        if frac >= 1.0:
            return "HARD_STOP"
        if frac >= self.config.drain_frac:
            return "DRAIN_AND_DESTROY"
        if frac >= self.config.stop_provision_frac:
            return "STOP_PROVISIONING"
        if frac >= self.config.warn_frac:
            return "WARNING"
        return "OK"

    def _append_event(self, data: dict[str, Any], kind: str, detail: dict[str, Any]) -> None:
        data.setdefault("events", []).append({"at": _now(), "kind": kind, **detail})

    def reserve_benchmark(self, amount: float | None = None) -> dict[str, Any]:
        amount = float(amount if amount is not None else self.config.benchmark_usd)
        with self._lock:
            data = self._read()
            snap_used = float(data["spent_usd"]) + float(data["committed_usd"]) + float(
                data.get("reserved_benchmark_usd", 0.0)
            )
            if snap_used + amount > self.config.hard_limit_usd:
                raise BudgetRefused(
                    f"Cannot reserve benchmark ${amount:.2f}: would exceed hard limit "
                    f"${self.config.hard_limit_usd:.2f} (used=${snap_used:.2f})"
                )
            data["reserved_benchmark_usd"] = float(data.get("reserved_benchmark_usd", 0.0)) + amount
            self._append_event(data, "reserve_benchmark", {"amount_usd": amount})
            self._write(data)
            return self.snapshot()

    def release_benchmark_reserve(self, amount: float | None = None) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            cur = float(data.get("reserved_benchmark_usd", 0.0))
            amount = float(amount if amount is not None else cur)
            data["reserved_benchmark_usd"] = max(0.0, cur - amount)
            self._append_event(data, "release_benchmark_reserve", {"amount_usd": amount})
            self._write(data)
            return self.snapshot()

    def check_projected_total(self, projected_remaining_cost_usd: float, *, context: str) -> None:
        """Refuse if projected remaining workload cost alone exceeds hard budget."""
        if projected_remaining_cost_usd > self.config.hard_limit_usd:
            raise BudgetRefused(
                f"REFUSE TO SCALE ({context}): projected remaining cost "
                f"${projected_remaining_cost_usd:.2f} exceeds AFFINITY_CLOUD_BUDGET_USD="
                f"${self.config.hard_limit_usd:.2f}"
            )
        snap = self.snapshot()
        if snap["absolute_block"]:
            raise BudgetRefused(
                f"REFUSE TO SCALE ({context}): budget absolute block "
                f"(used=${snap['used_usd']:.2f} / ${snap['hard_limit_usd']:.2f})"
            )

    def authorize_provision(
        self,
        *,
        hourly_usd: float,
        hours: float,
        context: str,
        projected_remaining_cost_usd: float | None = None,
    ) -> dict[str, Any]:
        """Authorize a new rental. Commits hours*hourly to the ledger on success."""
        commitment = float(hourly_usd) * float(hours)
        if commitment < 0:
            raise BudgetRefused("negative commitment")
        with self._lock:
            data = self._read()
            spent = float(data["spent_usd"])
            committed = float(data["committed_usd"])
            reserved = float(data.get("reserved_benchmark_usd", 0.0))
            used = spent + committed + reserved
            hard = self.config.hard_limit_usd
            frac = used / hard if hard > 0 else 1.0
            if frac >= self.config.stop_provision_frac:
                raise BudgetRefused(
                    f"REFUSE TO SCALE ({context}): used_fraction={frac:.3f} >= "
                    f"stop_provision_frac={self.config.stop_provision_frac}"
                )
            if projected_remaining_cost_usd is not None and projected_remaining_cost_usd > hard:
                raise BudgetRefused(
                    f"REFUSE TO SCALE ({context}): projected remaining "
                    f"${projected_remaining_cost_usd:.2f} > hard ${hard:.2f}"
                )
            if used + commitment > hard:
                raise BudgetRefused(
                    f"REFUSE TO SCALE ({context}): commitment ${commitment:.2f} would exceed "
                    f"remaining ${hard - used:.2f} (hard=${hard:.2f})"
                )
            data["committed_usd"] = committed + commitment
            self._append_event(
                data,
                "authorize_provision",
                {
                    "context": context,
                    "hourly_usd": hourly_usd,
                    "hours": hours,
                    "commitment_usd": commitment,
                },
            )
            self._write(data)
        return self.snapshot()

    def record_spend(self, amount_usd: float, *, context: str, release_commitment: float = 0.0) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            data["spent_usd"] = float(data["spent_usd"]) + float(amount_usd)
            if release_commitment:
                data["committed_usd"] = max(0.0, float(data["committed_usd"]) - float(release_commitment))
            self._append_event(
                data,
                "record_spend",
                {"context": context, "amount_usd": amount_usd, "release_commitment": release_commitment},
            )
            self._write(data)
            return self.snapshot()

    def release_commitment(self, amount_usd: float, *, context: str) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            data["committed_usd"] = max(0.0, float(data["committed_usd"]) - float(amount_usd))
            self._append_event(data, "release_commitment", {"context": context, "amount_usd": amount_usd})
            self._write(data)
            return self.snapshot()
