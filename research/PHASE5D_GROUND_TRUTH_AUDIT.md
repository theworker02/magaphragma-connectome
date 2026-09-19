# Phase 5D ground-truth audit

## Acquired authoritative artifacts

The official MIT-licensed EMNeuron dataset repository was inspected at commit
`62df92b933550cc2a5d3849fe8543191c356f2d4`. Local, ignored-cache artifacts:

| Artifact | Bytes | SHA-256 | Status |
| --- | ---: | --- | --- |
| `EMNeuron-labeled.rar` | 2,345,568,740 | `966daa6048a22d2647efbe82f53b2a17fa09047c00a942a4101b724260fc1880` | acquired, archive indexed |
| `EMNeuron-valid.rar` | 593,571,609 | `a3f30ae39deb84488c8a9eafc0edfa539d4ea2628fe7e464a6636b83695fd3c2` | acquired, archive indexed |
| `SegNeuronModel.ckpt` | 161,669,157 | `3767a3793cbbdbf98041b8c4c270b92d2355213d11605568c003322bb759bef5` | acquired and technically executed |

Archive metadata confirms raw TIFF and neuronal instance-mask pair naming
(`*_MaskIns.tif`) in the labeled artifact. The validation artifact contains
FIB25 raw/label pairs (`FIB25-1`, `FIB25-2`). The official repository documents
FIB25 as 8 × 8 × 8 nm FIB-SEM. These facts make EMNeuron relevant external
training material.

## Strict limits

The artifacts are not *Megaphragma*, do not establish a transform to DVID, and
do not transfer biological identity. The model checkpoint may have been trained
using related FIB25 material; archive names alone cannot establish a clean,
independent checkpoint-selection or test split. Therefore no EMNeuron metric
will be represented as DVID performance, and no external score can authorize
DVID production.

WASPSYN remains a source for future synapse-model research only: its published
annotations are pre/post coordinates and partner relations, not neuronal
instance masks.

## Current ground-truth gate

`ground_truth/` intentionally contains no `MV-GT-N-*` object. A qualified
reviewer must supply DVID-native instance masks and review records before
spatially separated train, validation, and immutable test manifests can be
frozen. This is the remaining blocker to production authorization.
