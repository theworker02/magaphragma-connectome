# Target-native review queue

This queue is for bounded neuronal-process labels, not complete-neuron claims.
Every raw cube is real DVID/FIB-SEM in `MV-FRAME-DVID-WASP5-001`; the raw
source cache is immutable. A proposal may help a reviewer begin annotation but
cannot be registered as a DVID label without review/correction.

## Locked partitions

| Role | Regions | Rule |
| --- | --- | --- |
| Training candidates | `MV-GTVOL-000001`, `000002`, `000003` | Review different local structure/contrast regimes after visual QA; do not assume anatomical difficulty from coordinates alone. |
| Validation candidate | `MV-GTVOL-000005` | Independently review and never use to update weights. |
| Regression only | `MV-GTVOL-000004` | No labels, training, or model selection. |
| Test reserve | `MV-GTVOL-000006` | No training or model selection. |

## First proposal receipt

`MV-GTVOL-000001` was inferred from the saved source-trained step-2250 model
using its real raw volume (SHA-256
`a8282dc04a789528a7e83a5fba92a057f1ec23a1d71eec6ff9efa1d40eedcd07`) and
checkpoint SHA-256
`094eabaaaa93ce33349e5c3a9ee1c166bcda8081c307d6a2d51c74a10d8af354`.
The raw prediction receipt is in the ignored local research build. Its simple
3-D affinity-component reviewer aid had one component covering all 7,077,888
voxels and is explicitly flagged `NEAR_FULL_VOLUME_MERGE_CANDIDATE` and
`VOLUME_BOUNDARY`. It is **not** eligible for conversion or registration.

The true-3-D ELF review staging did not execute on this host because the
installed Windows and Ubuntu environments both lack the `elf` Python package.
That is an environment dependency blocker only; no substitute 3-D biological
result was manufactured.

## Reviewer handoff

1. Open a training-candidate raw cube in synchronized XY/XZ/YZ.
2. Use any machine proposal only as an overlay/reference, then trace and
   correct bounded local neuronal instances directly from raw EM.
3. Mark cube-face contacts `GT_VOLUME_TRUNCATED` and uncertain continuity
   `AMBIGUOUS`.
4. Export a same-grid non-negative integer `labels_zyx.npy`.
5. Register only reviewer-supplied labels. The CLI rejects `000004` outright.

Target adaptation unlocks only after at least one independently reviewed train
cube and one distinct independently reviewed validation cube exist.
