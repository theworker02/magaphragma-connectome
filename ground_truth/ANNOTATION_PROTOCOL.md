# DVID-native instance annotation protocol

## Inputs and boundaries

Each `MV-GTVOL-*` source cube is an immutable, local-cache-only 192 × 192 ×
192 voxel DVID extraction at 8 nm isotropic resolution. Open it in a 3-D EM
annotation/proofreading tool capable of synchronized XY/XZ/YZ inspection (for
example, a locally configured Napari or Neuroglancer-compatible workflow).
The project's viewer and manifest provide the authoritative XYZ source bounds;
they do not supply biological identities.

## Required annotation output

Export one `uint32` or other non-negative-integer `labels_zyx.npy` array on
exactly the same ZYX grid as the raw cube. Zero is background. Every positive
value denotes one independently followed neuronal process, not merely
foreground. Preserve model proposals separately if used; they are not labels.

For each process, inspect orthogonal views, record uncertain continuity as
`AMBIGUOUS`, and mark every object meeting a cube face as
`GT_VOLUME_TRUNCATED`. Inspect merges, splits, branches, thin processes and
dense contacts explicitly. A second independent or clearly separated review is
required before `GOLD_STANDARD`.

## Registration

After the qualified reviewer has created a mask, use the fail-closed importer:

```powershell
$env:PYTHONPATH='src'
python -m mvconnectome.cli ground-truth-register MV-GTVOL-000001 <labels_zyx.npy> --reviewer <reviewer-id> --status FIRST_PASS
```

It rejects shape mismatch, non-integer/negative labels, empty labels, unknown
regions, missing reviewer identity, and overwriting of a registered label.
It preserves the supplied mask, records its SHA-256, creates `MV-GT-N-*`
aliases, and does not promote any object to a complete neuron.

## Target-adaptation split lock

`MV-GTVOL-000004` is permanently **REGRESSION_ONLY** for target-domain
adaptation. It must not receive a review label, enter train/validation model
selection, or be used to tune a model. Use `MV-GTVOL-000001` through `000003`
for reviewed/corrected training labels and `MV-GTVOL-000005` for an independent
reviewed validation label. `MV-GTVOL-000006` remains held out.

Machine output is a reviewer aid only:
`MACHINE_PSEUDOLABEL -> REVIEW_REQUIRED -> REVIEWED/SECOND_PASS_REVIEWED -> GOLD_STANDARD`.
It never becomes ground truth solely because it was generated.

## Bounded crop review

`ground_truth/DVID_ANNOTATION_CROP_PLAN.json` freezes three real, spatially
separate training-candidate crops and a separate validation crop. Their
materialized raw arrays and hashes are in `annotation_crops_manifest.json`.
Open a local editable review workspace (Napari installed separately) with:

```powershell
python tools/review_annotation_crop.py MV-DVID-ANN-000001
```

The raw layer is source evidence. The editable labels layer begins empty; save
only a reviewer-created `labels_zyx.npy`. Use `4294967295` for genuinely
uncertain/ignore voxels, zero for background, and positive IDs for locally
followed neuronal instances. Then register it with `annotation-crop-register`.
