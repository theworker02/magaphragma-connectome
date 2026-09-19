# Phase 5C DVID production-domain evidence record

## Authoritative source interrogation

On 2026-09-16, the public DVID image-instance metadata endpoint for
`aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast` reported:

| Property | Recorded value |
| --- | --- |
| Coordinate convention | zero-based XYZ voxel centers; arrays are handled as ZYX |
| Minimum point | `[0, 0, 0]` |
| Maximum point (inclusive) | `[16647, 13543, 15400]` |
| Half-open domain | `x=[0,16648), y=[0,13544), z=[0,15401)` |
| Dimensions | `16,648 × 13,544 × 15,401` voxels |
| Voxel size | `8 × 8 × 8 nm` |
| Physical extent | `133.184 × 108.352 × 123.208 µm` |
| Native DVID block shape | `64 × 64 × 64` voxels |
| Raw uint8 payload estimate | `3,472,625,365,312` bytes (about 3.47 TB decimal) |

The retained parent-metadata receipt has SHA-256
`0400003dae908dd557797f085696bd4962e47563923fe661fccfd7fd59650abb`.
A 64-cube raw read at the inclusive far edge begins at
`[16584, 13480, 15337]`; its retained 262,144-byte response has SHA-256
`146171f08f15b01203d3e22a4df82b72ecb2c8a5cc1e285201847c2826afc8aa`.

This proves public parent metadata and a read at the far boundary. It does
not claim that the full parent volume has been downloaded, validated
block-by-block, processed, or released.

## Production work domain

`MV-DOMAIN-PRODUCTION-001` is the full source-reported contiguous extent.
Its immutable local manifest and a resumable, global work index are generated
by:

```powershell
$env:PYTHONPATH='src'
python -m mvconnectome.cli phase5c-plan-domain
```

The current planner creates 28,798 work items with 1024 × 1024 × 128 voxel
cores and 64 × 64 × 16 voxel provisional read halos. Both values are exact
multiples or subdivisions appropriate to the 64-cube DVID storage blocks.
They are planning parameters, not model requirements: the selected production
model must validate its receptive-field/context needs before any work item is
processed.

Each record starts as `NOT_REQUESTED`/`NOT_STARTED`, stores source and read
bounds, stable neighboring work-item IDs, explicit review state, and retry
counter. A queue item has no cached imagery, segmentation, skeleton, synapse,
or biological identity merely by existing.

## Preserved scientific and release gates

- `CATMAID_DVID_REGISTRATION = INSUFFICIENT_EVIDENCE` remains unchanged.
- CATMAID-derived DVID seeding and identity transfer remain prohibited.
- The original 512 × 512 × 64 crop is retained as
  `MV-DOMAIN-PIPELINE-VALIDATION-001`, not expanded or relabeled.
- `production_segmenter_status = NONE_SELECTED`: the CREMI benchmark model was
  rejected; no production model may be selected from visual plausibility.
- `synapse_backend_status = UNKNOWN_OR_UNAVAILABLE`.
- Parent data and derivative-data redistribution permissions remain unresolved;
  both receipts and all future local processing stay outside release packages.

Consequently, Phase 5C has **zero accepted target `MV-SEG` records** and no
target skeletons, fragments, neurons, synapses, or connections. The next
scientific gate is legitimate target-compatible labels (or a properly
documented compatible pretrained model) with spatially separated held-out
evaluation including merge/split analysis.
