# External DVID boundary review protocol

## Scope

The reviewer is not asked to trace a neuron or make a connectome claim. Each
decision asks whether two adjacent raw-EM voxels, presented with orthogonal
context, are within the `SAME_PROCESS` or on opposite sides of a
`DIFFERENT_PROCESS` neuronal boundary. `UNCERTAIN` and `BAD_QUESTION` are
valid outcomes and are excluded from model loss.

The immutable request manifest is
`ground_truth/EXTERNAL_DVID_BOUNDARY_REVIEW_REQUEST_001.json`. It identifies a
spatially separated TRAIN crop and VALIDATION crop by raw-content hash. The
regression cube `MV-GTVOL-000004` is not a review target.

## Launching the local expert workspace

Create one immutable workspace per requested crop, then open it with the
dedicated raw-EM reviewer:

```powershell
$env:PYTHONPATH = 'src'
python -m mvconnectome.cli external-boundary-workspace-create MV-DVID-ANN-000001 --output local_research_build/phase6e/external-review/MV-DVID-ANN-000001-workspace.json
python tools/review_external_boundary_package.py local_research_build/phase6e/external-review/MV-DVID-ANN-000001-workspace.json --reviewer "expert-identifier"
```

Repeat for `MV-DVID-ANN-000004` only after the TRAIN review is frozen. The
reviewer selects A, chooses the adjacent Z/Y/X direction to B, and records a
decision directly from the control panel. The viewer defaults to raw EM and
shows A/B in XY, XZ, and YZ; no machine segmentation is loaded.

For broad, unbiased inspection, create and pass the frozen navigation-only
queue. It is a spatial grid—not a membrane candidate generator—and has no
model, supervoxel, feature-ranking, or prior-decision input:

```powershell
python -m mvconnectome.cli external-boundary-scan-create local_research_build/phase6e/external-review/MV-DVID-ANN-000001-workspace.json --output local_research_build/phase6e/external-review/MV-DVID-ANN-000001-scan.json --count 72
& .\.venv-reviewer\Scripts\python.exe tools\review_external_boundary_package.py local_research_build/phase6e/external-review/MV-DVID-ANN-000001-workspace.json --reviewer "approved-reviewer-id" --scan-queue local_research_build/phase6e/external-review/MV-DVID-ANN-000001-scan.json
```

Replace `expert-identifier` with the actual reviewer identity or a stable,
approved reviewer pseudonym. Literal placeholders (`expert-identifier`,
`reviewer`, `unknown`, `tbd`) are rejected and prior placeholder events are
excluded from supervision.

## Rights boundary

This repository does not infer a right to redistribute the raw DVID EM. Give
the reviewer legitimate source access or let them work in an authorized local
project environment. Do not email, publish, upload, or otherwise export the
raw array until the source terms expressly permit that delivery route.

## Required record for each decision

The expert should record:

- crop ID and raw SHA-256;
- `pair_left_zyx` and `channel_zyx` (`0=Z`, `1=Y`, `2=X`); the paired voxel is
  the positive neighbour along that channel;
- `SAME_PROCESS`, `DIFFERENT_PROCESS`, `UNCERTAIN`, or `BAD_QUESTION`;
- reviewer identity, timestamp, and optional rationale;
- confirmation that XY, XZ, YZ, and nearby slices were inspected.

Only adjacent, axis-aligned pairs map directly to SegNeuron’s affinity
targets. An expert may reject an inadequate prompt rather than guessing.

## Admission

Before G1 adaptation is allowed, each split independently needs at least one
reviewed `SAME_PROCESS` and one reviewed `DIFFERENT_PROCESS` decision. The
importer must then verify raw hashes, coordinate geometry, protected-volume
exclusion, and train/validation separation. These records are target training
evidence—not neuron, synapse, or connection records.

Materialize a split only after its decisions meet that gate:

```powershell
python -m mvconnectome.cli external-boundary-materialize-affinity local_research_build/phase6e/external-review/MV-DVID-ANN-000001-workspace.json --output local_research_build/phase6e/external-review/MV-DVID-ANN-000001-affinity
```

The command fails closed if either reviewed class is absent, the raw hash has
changed, an affinity pair is invalid, or an attempt is made to overwrite an
immutable result.
