# AxonForge acceleration report

CPU simulation harness (not GPU HyperDrain).

| Claim | Result |
|-------|--------|
| Halo cache | Accepted when border fingerprint hits and checksum matches reference |
| Cascade-only | Rejected — fails equivalence vs reference |
| Adaptive e2e speed ≥2× | Not claimed on this harness |

See `receipts/SUMMARY.json` after `python -m axonforge.cli demo`.
