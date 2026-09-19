# G3 Affinity Target Specification 002

Companion to `experiments/phase6e/G3_AFFINITY_TARGET_SPEC_002.json`.

## Why this exists

G3-008 completed 36/36 paired-location reviews. Every triplet was Z=`DIFFERENT_PROCESS`, Y/X=`SAME_PROCESS`. The operational human-review target was declared **`CURRENT_AFFINITY_TARGET_INVALID`**. This document states the **correct biological affinity target** independently of that failed review implementation.

## Four layers (do not collapse)

| Layer | What it is | What it is not |
|-------|------------|----------------|
| **Biological target** `A(v,a)` | Same object id across the face `(v, v+e_a)` under a valid reference segmentation `S` | Not EM brightness; not a button click; not a network logit |
| **Observable EM evidence** | Local raw intensities / textures around the face | Not proof of object identity by itself |
| **Annotation procedure** | How humans or code estimate `A` | Invalid if presentation is axis-asymmetric or if labels equal axis identity |
| **Model prediction** `Â` | Learned approximation to `A` | Not ground truth |

## Formal definition

Array order is **ZYX**. Axes `a ∈ {0,1,2}` are Z, Y, X. Voxels are isotropic **8×8×8 nm**.

For voxel `v` and axis `a`:

\[
A(v,a) = 1 \iff S(v) = S(v+e_a) \neq 0
\]

otherwise `A(v,a) = 0` on the labeled domain (undefined on ignore/out-of-bounds).

This definition is **invariant** under global coordinate permutation when `S` and `A` are transformed together.

## Derivation vs human judgment

- If a validated instance segmentation `S` exists on the same crop/frame, affinities **must** be derived mechanically from `S`, with full provenance.
- Human SAME/DIFFERENT judgment is an annotation procedure estimating `A` from EM evidence. It is allowed only when the **entire** presentation (pair, context, planes, camera, normalization, overlays, text) is permutation-equivariant—or when humans correct an existing `S`, not invent face labels in isolation.

## Relation to G3-008

G3-008 tested whether historical orientation-collinear labels survive axis-neutral paired sampling. They did (12/12). Forensic presentation tests show the Napari path is **not** permutation-equivariant (XY co-visibility and focus asymmetries for Z). Therefore G3-008 labels are **not** instances of this specification and must not be used as training targets.

## Production rule

No affinity model may be trained for Megaphragma DVID production until labels are either:

1. derived from validated `S` on target crops, or  
2. collected under a presentation-equivariant procedure that passes the synthetic permutation suite and a small prospective human validation.
