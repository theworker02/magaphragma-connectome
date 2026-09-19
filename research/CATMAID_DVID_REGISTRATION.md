# CATMAID ↔ DVID registration investigation

## Decision

**INSUFFICIENT_EVIDENCE.** No CATMAID-derived seed, physical source-neuron mapping, physical source-synapse mapping, or physical `MV-CONN` mapping may be created for the current DVID crop.

## Established frames

- `MV-FRAME-CATMAID-001`: CATMAID project 1/stack 1; physical XYZ nanometres; 8 nm isotropic; audited dimensions 8000 × 12000 × 12331 voxels.
- `MV-FRAME-DVID-WASP5-001`: DVID node `aa49d16e88424cdbae89f4b8ced2a49b`, instance `five_yuri_4contrast`; zero-based parent XYZ voxels; 8 nm isotropic. The local crop is parent `[4000,4512) × [6000,6512) × [6000,6064)`.

The fact that both sources report 8 nm isotropic voxels is compatible with many datasets and cannot establish common specimen identity, orientation, origin, or a transform.

## Lineage findings

The CATMAID source is the published early visual-system project. The DVID source is publicly named `wasp5-yuri`; its public metadata captured in this repository does not identify acquisition authors, publication, specimen, or an authoritative relationship to CATMAID. No source-provided CATMAID↔DVID transform, shared acquisition identifier, or independently verifiable landmark correspondence has been obtained.

## Validation state

No candidate transform was fitted: fitting an axis/translation model without independent landmarks would manufacture registration evidence. Consequently there are zero fit, validation, and held-out landmarks; residual statistics and seed-safety error are unavailable, not zero.

## Consequence

The active branch is **DVID-native physical reconstruction**. FFN or another registered backend may use only DVID-native seeds until an authoritative transform and held-out landmark validation are available. The CATMAID source connectome remains a separate, quarantined source layer.

Structured records: `registration/coordinate_frames.json` and `registration/decision.json`.
