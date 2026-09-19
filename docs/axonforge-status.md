# AxonForge v1.4 — Connectome Project status

**Project:** Connectome Project  
**Subsystem:** AxonForge  
**AxonForge Version:** 1.4.0

## Location in this repo

- Package: `axonforge/`
- Dashboard: `axonforge_dashboard/`
- Receipts: `receipts/SUMMARY.json`
- Tests: `tests/test_axonforge_core.py`, `tests/test_demo_run.py`

## Run

```bash
cd magaphragma-connectome
PYTHONPATH=. python -m axonforge.cli serve --port 8741
# API+dashboard: http://127.0.0.1:8741
# Health:        http://127.0.0.1:8741/health
# Demo:          http://127.0.0.1:8741/demo_run
```

Optional second port for dashboard-only static mirror:

```bash
python -m http.server 8742 --directory axonforge_dashboard
```

## Verified locally (2026-09-18)

- Fixed SyntaxError from mangled `+ "\\n"` string literals in `runtime.py` / `graph.py`.
- Fixed `request_inference(run_adaptive=...)` shadowing the `run_adaptive` import (`'bool' object is not callable`).
- pytest: `tests/test_axonforge_core.py` + `tests/test_demo_run.py` — **9 passed**.
- Live: [AxonForge API](http://127.0.0.1:8741) health ok; dashboard on same port; demo_run 303.
- Optional static mirror: [AxonForge dashboard :8742](http://127.0.0.1:8742).

- CTA on :8742 now uses absolute `/demo_run` + `return_to` (Playwright PASS → 1.00M).

## Acceptance posture

- **Accepted:** halo cache (when hits occur with equivalence)
- **Rejected until measured faster+equiv:** cascade-only, adaptive e2e speed claims
- Completeness: `validated_coverage=1.0`, `unaccounted_volume=0` on adaptive demo
