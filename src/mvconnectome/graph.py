"""Connectivity construction from concrete, provenance-bearing synapse records only."""

from __future__ import annotations

from collections import defaultdict
from itertools import count

from .ids import stable_id
from .models import Connection, EvidenceStatus, ReviewState, Synapse


def connections_from_synapses(synapses: list[Synapse]) -> list[Connection]:
    groups: dict[tuple[str, str], list[Synapse]] = defaultdict(list)
    for synapse in synapses:
        if synapse.pre_neuron_id is None or synapse.post_neuron_id is None:
            continue
        groups[(synapse.pre_neuron_id, synapse.post_neuron_id)].append(synapse)
    connections: list[Connection] = []
    for serial, ((pre, post), items) in zip(count(1), sorted(groups.items())):
        statuses = {item.status for item in items}
        reviews = {item.review_state for item in items}
        status = EvidenceStatus.MANUALLY_VERIFIED if statuses == {EvidenceStatus.MANUALLY_VERIFIED} else EvidenceStatus.MACHINE_PREDICTED
        review = ReviewState.EXPERT_REVIEWED if reviews <= {ReviewState.EXPERT_REVIEWED, ReviewState.LOCKED_RELEASE} else ReviewState.MACHINE_ONLY
        connections.append(Connection(
            id=stable_id("connection", serial), pre_neuron_id=pre, post_neuron_id=post,
            synapse_ids=tuple(item.id for item in items), evidence_ids=tuple(sorted({item.evidence_id for item in items})),
            status=status, review_state=review,
        ))
    return connections
