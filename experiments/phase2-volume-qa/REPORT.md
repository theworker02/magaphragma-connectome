# Phase 2 real-volume QA and segmentation run

## Source and bounds

- Dataset: `MV-SRC-FLATIRON-DVID-WASP5` (local-only; attribution and release rights unresolved)
- Volume: `MV-FIBSEM-WASP5-YURI-4C`
- Exact source bounds: X `[4000,4512)`, Y `[6000,6512)`, Z `[6000,6064)`
- Voxel size: 8 × 8 × 8 nm, yielding 4.096 × 4.096 × 0.512 µm
- Raw cache SHA-256: `78e786152f9c473eced1d18456da92fd2329044637ebfd545bd1ff3ca099fbcd`

## Visual inspection

The central source slice at Z=6032 was inspected in the generated raw and overlay images. Membrane-like boundaries and varied intracellular ultrastructure are visible. This is a real source-data observation only; no claim of cell type, neuron identity, or annotation correctness is made. The limited 0.512 µm depth means continuity claims are not supported.

## Baseline execution

- Run: `MV-SEG-RUN-000001`
- Model/baseline: `MV-SEG-MODEL-0001`, gradient-derived membrane proxy + marker-controlled watershed
- Input crop: X `[4000,4512)`, Y `[6000,6512)`, Z `[6016,6048)`
- Runtime: 12.51 seconds, CPU
- Output: 60 candidate `MACHINE_SEGMENTED` objects, no neurons and no connectivity edges
- Output hashes: labels `2cb181ba76e7a39bc69c6ba5a2a1c258b3da1455bef511dff13abb0635f8793d`; membrane proxy `8d5b377f3bb155c627b95093d7f012b8bbee3b5ecac4126354479580b52f7da9`

The raw arrays, label volume, visual overlays, and candidate segment table are excluded from Git and available only in local cache/derived directories. No review action has been recorded; all generated segments remain machine-only.
