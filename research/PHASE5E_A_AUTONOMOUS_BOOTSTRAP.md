# Phase 5E-A autonomous ground-truth bootstrap

SegNeuron proposals are retained as `MACHINE_PSEUDOLABEL`, never ground truth.
The proposal-only route is:

`real DVID raw → SegNeuron affinity outputs → proposal components → automated QA flags → reviewer adjudication → separately registered instance labels`.

The initial automated QA flags tiny fragments, volume-edge contacts, and high
affinity uncertainty. These are review priorities, not error determinations.
Single-model disagreement is honestly `NOT_AVAILABLE`; a future second model
may create an additional disagreement signal. Automated repair is intentionally
not enabled until its effects can be checked against independently adjudicated
labels.

## Executed first proposal

`MV-SEGNEURON-ZERO-SHOT-000004` ran the official checkpoint on the complete
real `MV-GTVOL-000004` DVID cube: 192³ voxels, 171 CPU tiles, 207.85 seconds.
The retained `AFFINITY_COMPONENT_PROPOSAL_V0` result collapsed 7,077,384 of
7,077,888 voxels into a single component. The append-only severity audit
flagged it `NEAR_FULL_VOLUME_MERGE_CANDIDATE`. It is rejected as an annotation
proposal for direct use, retains `MACHINE_PSEUDOLABEL`/`REVIEW_REQUIRED`, and
is prohibited from ground truth, metrics, `MV-SEG`, or production use.

This design follows the useful part of semi-supervised and automated
proofreading research—reducing annotation work through proposals and targeted
inspection—without circularly treating a model's own predictions as its test
truth. The official SL-SSNS repository describes representative subvolume
selection and consistency training; the official SegNeuron repository presents
coarse segmentation plus correction/fine-tuning as its intended workflow.

## FRMC proposal comparison

After the official Linux/WSL `python-elf=0.8.1` environment was installed,
the checked-out ELF watershed-plus-multicut postprocessor ran on the same real
`MV-GTVOL-000004` SegNeuron affinity and boundary outputs with `beta=0.25`.
It produced a retained uint32 label volume with SHA-256
`dbc332d3701496ce5c78b2a93a3b15ab75d89cadfcd14db1431d36dfc84f67a7`.

Its sole instance occupied all 7,077,888 voxels. The independent proposal
audit flagged `NEAR_FULL_VOLUME_MERGE_CANDIDATE`, so this FRMC result is also
`MACHINE_PSEUDOLABEL` and `REVIEW_REQUIRED`, rejected for direct annotation,
ground truth, held-out evaluation, `MV-SEG`, and production use. This is an
automated error-detection result, not evidence that the cube contains one
neuron or that the postprocessor has been biologically qualified.
