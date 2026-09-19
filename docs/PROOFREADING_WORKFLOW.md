# Supervoxel-graph proofreading workflow

Vigilia follows the operational separation used by large EM connectomics
projects: machine segmentation proposes local supervoxels; humans review graph
edits; proofread reconstructions are subsequently associated with independently
detected synapses.  It does **not** treat automated instance predictions as
finished neurons.

## States and prohibitions

`MACHINE_SUPERVOXEL_GRAPH_REVIEW_REQUIRED` contains a raw immutable crop plus
a machine-derived label volume.  It is a review workspace, not an `MV-FRAG`,
`MV-N`, synapse, or connection.  It may be made only from eligible DVID
development crops; `MV-GTVOL-000004` is regression-only and is rejected.

Every MERGE, SPLIT, FLAG, or REJECT action is appended to an immutable JSONL
event log.  Source imagery and the original machine label volume are never
modified.  A SPLIT stores human seed points and does not claim that a generated
cut is reviewed.  A corrected instance mask must separately pass
`annotation-crop-register`, including reviewer identity, alignment, label
integrity, hash, and split-safety checks, before it is `REVIEWED_DVID_LABEL`.

## Practical review loop

1. Generate deliberately oversegmented, true-3-D supervoxels from a model
   prediction. Prefer extra pieces to a false merge.
2. Open raw EM and the segmentation overlay in a local Napari/Neuroglancer
   surface. Inspect XY, XZ, and YZ at each proposed merge or split.
3. Record merge/split/flag/reject operations against supervoxel IDs and local
   coordinates. These events are provenance, not automatic training labels.
4. Convert only inspected corrections to a reviewed DVID label. Train future
   model generations from that reviewed target supervision; retain the machine
   proposal and edit log as lineage.
5. Admit a physical `MV-FRAG` only after the existing reconstruction QA gate.

This makes human work local and error-directed. It does not remove the need
for independent review where biological identity or connectivity is claimed.
