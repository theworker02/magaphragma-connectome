# Local scientific query API

Start it with `vigilia serve`. It binds only to `127.0.0.1`.

| Endpoint | Meaning |
| --- | --- |
| `GET /api/status` | Real registry state and zero-safe counts. |
| `GET /api/datasets/{id}` | Audited source dataset record, including access and rights state. This does not imply a biological import. |
| `GET /api/evidence/{id}` | An evidence record. `MV-EV-000001` is the audited, literature-derived source-access evidence for the CATMAID reference, not a biological annotation. |
| `GET /api/neurons/{id}` | A neuron plus its direct evidence and source segments. Returns 404 when no such biological record exists. |
| `GET /api/connections/trace?pre=MV-N-…&post=MV-N-…` | A connection, concrete synapses, source segments, volume registrations, and evidence records. |

The expected trace is `neuron → synapse → segment → volume coordinates → evidence/source`. API payloads must preserve statuses and review state; consumers must not collapse predictions into verified observations.
