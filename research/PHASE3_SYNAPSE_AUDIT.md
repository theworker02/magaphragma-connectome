# Phase 3 synapse-data audit

Audit date: 2026-09-16.

## Published lamina source

Chua et al. (2023) report that their public CATMAID service provides the *Megaphragma viggianii* FIB-SEM dataset, neuron reconstructions, and synapse annotations. The public analysis repository identifies the paper/authors and links the service. This is the strongest prospective source for a first evidence-supported connectivity import.

**Current access finding:** the CATMAID project listing and stack metadata are publicly readable. However, the visible tile mirror references a no-longer-resolvable DVID UUID, and this repository has not established an annotation-export endpoint, export terms, artifact checksum, coordinate transform for an export, or data redistribution/derivative-release license. No CATMAID synapse was imported.

## WASPSYN

Li et al. (2024) describe 14 *M. viggianii* FIB-SEM subvolumes at 8 nm isotropic resolution. Five sample-3 training volumes contain pre-/postsynaptic point annotations and one-to-many partner relationships; test labels are withheld by the challenge design. The paper states that the dataset remains public under CC-BY.

**Current access finding:** a stable artifact URL, original file names/checksums, artifact-level license statement, coordinate manifest, and mapping from synapse points to neuronal segments have not been captured. Its documented training volumes do not establish overlap with `MV-FIBSEM-WASP5-YURI-4C`.

## Gate decision

No Phase 3 biological synapse, neuron, partner assignment, connection, matrix, or graph may be created yet. The current Phase 2 DVID crop is local-only because its server metadata also lacks specimen attribution and release rights. Proximity between its machine-segmented regions cannot constitute synaptic evidence.

## Required next acquisition

Obtain one of the following before resuming Phase 3 execution:

1. A rights-confirmed CATMAID export containing source stack identity, neuron/segment identifiers, synapse coordinates, pre/post assignments, and coordinate transform; or
2. A WASPSYN training artifact with its exact license, checksum, coordinate manifest, and enough segmentation/partner mapping to support a source-attributed import.

Then register it as a separate immutable source volume, validate point coordinates against raw imagery, and import it with `SOURCE_ANNOTATED` status. Only a recorded human review may create a verified connection.
