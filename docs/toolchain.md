# Connectome Project toolchain

Engineering tools around AxonForge — deterministic geometry, graph, stats, and hashing first.

## Pipeline diagram

`
                 RegionBench (frozen synthetic regions)
                              |
                              v
  [plan]  VoxScout.analyze --> priority_map.json --> AxonForge hints (optimize_ok)
                              |
                              v
                         ArtifactVet (schema/shape/dtype/hash)

  [execute] AxonForge (+ light switch) <--> NeuroCache (tile inference identity)

  [close-gaps] GapHound.scan --> repair-plan --> TileMedic (failed only)
                     |
                     v
                ArtifactVet on latest_scan.json

  [reconcile] SeamSmith.analyze --REVIEW--> BranchJudge (evidence assist, never auto-merge)

  [provenance] TraceWire <--> EdgeProbe (pre/post diagnostic bundle)

  [diagnose] RunDoctor (imports + switch + GapHound/TileMedic next commands)
`

## Priority instrumentation

- **VoxScout** — spatial priority map (never discards; hints optimize_ok)
- **AxonForge** — execution / GPU + autonomous light switch
- **GapHound** — completeness hunter + minimum repair-plan worklist
- **TileMedic** — safe failed-tile recovery (never mutates scientific params)
- **BranchJudge** — split/merge evidence assistant for SeamSmith/MorphGuard
- **RunDoctor** — connectome doctor one-command diagnosis

## Broadened tools

- **Pipeline** — plan (VoxScout→hints) and close-gaps (GapHound→repair→TileMedic)
- **RegionBench-lite** — frozen synthetic regions under 
eceipts/regionbench/
- **EdgeProbe-lite** — TraceWire walk + bundle under 
eceipts/edgeprobe/
- **ArtifactVet-lite** — validate JSON/NPY before stage handoff
- **NeuroCache × AxonForge** — tile_inference_identity + put/lookup demo

## CLI

```bash
PYTHONPATH=.:tools python tools/connectome.py tools
PYTHONPATH=.:tools python tools/connectome.py switch status
PYTHONPATH=.:tools python tools/connectome.py doctor

# Cross-tool pipelines
PYTHONPATH=.:tools python tools/connectome.py pipeline plan --shape 64,64,64
PYTHONPATH=.:tools python tools/connectome.py pipeline plan --region rb-blob-064
PYTHONPATH=.:tools python tools/connectome.py pipeline close-gaps --shape 100,100,100

# Completeness / recovery
PYTHONPATH=.:tools python tools/connectome.py voxscout analyze
PYTHONPATH=.:tools python tools/connectome.py gaphound scan --shape 100,100,100
PYTHONPATH=.:tools python tools/connectome.py gaphound repair-plan
PYTHONPATH=.:tools python tools/connectome.py tilemedic recover --failed
PYTHONPATH=.:tools python tools/connectome.py branchjudge judge
PYTHONPATH=.:tools python tools/connectome.py seamsmith analyze

# Benchmarks / gates / cache / edges
PYTHONPATH=.:tools python tools/connectome.py regionbench freeze
PYTHONPATH=.:tools python tools/connectome.py regionbench smoke
PYTHONPATH=.:tools python tools/connectome.py artifactvet json receipts/gaphound/latest_scan.json --require shape_zyx counts
PYTHONPATH=.:tools python tools/connectome.py neurocache axonforge-demo
PYTHONPATH=.:tools python tools/connectome.py edgeprobe preA postB
`

Shared protocol: 	ools/_core/ (schemas, spatial, artifacts, provenance, receipts, artifactvet).

## Also registered

- SynapseLens, TraceWire, DeltaGraph, MorphGuard, HyperDrain

## Project entry (required)

Tools are connected through the main Connectome package — not only `python tools/...`:

```bash
pip install -e .
vigilia toolchain status          # 17/17 importable + connection map
vigilia toolchain doctor
connectome pipeline reconcile    # SeamSmith -> BranchJudge
connectome pipeline audit        # MorphGuard
connectome pipeline connectivity # SynapseLens
connectome pipeline diff         # DeltaGraph
connectome pipeline mass-status  # HyperDrain
```

AxonForge `request_inference` calls `post_run_toolchain` so every registered tool is exercised and receipted under `receipts/pipeline/post_run_toolchain.json`.
