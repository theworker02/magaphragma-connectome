"""Abstract cloud provider interface for Affinity fleets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Offer:
    offer_id: int
    gpu_name: str
    num_gpus: int
    vram_gb: float
    dph_total: float  # $/hour total (compute + assumed storage component when present)
    dph_base: float | None = None
    storage_cost_per_gb_hour: float | None = None
    inet_up_cost: float | None = None
    inet_down_cost: float | None = None
    reliability: float | None = None
    verified: bool | None = None
    rentable: bool | None = None
    interruptible: bool = False
    cuda_max_good: float | None = None
    cpu_cores: float | None = None
    cpu_ram_gb: float | None = None
    disk_gb: float | None = None
    geolocation: str | None = None
    machine_id: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Instance:
    instance_id: int
    status: str | None
    actual_status: str | None
    gpu_name: str | None = None
    dph_total: float | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    label: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CloudProvider(ABC):
    name: str

    @abstractmethod
    def search_offers(self, **kwargs: Any) -> list[Offer]:
        ...

    @abstractmethod
    def rank_offers(self, offers: list[Offer], **kwargs: Any) -> list[Offer]:
        ...

    @abstractmethod
    def create_instance(self, offer_id: int, **kwargs: Any) -> Instance:
        ...

    @abstractmethod
    def get_instance(self, instance_id: int) -> Instance:
        ...

    @abstractmethod
    def list_instances(self) -> list[Instance]:
        ...

    @abstractmethod
    def destroy_instance(self, instance_id: int) -> dict[str, Any]:
        ...

    @abstractmethod
    def estimate_cost(self, *, hours: float, dph: float, storage_gb: float = 0.0) -> dict[str, float]:
        ...

    @abstractmethod
    def fleet_status(self) -> dict[str, Any]:
        ...
