# Phase 6 SegNeuron saturation diagnosis

## Immutable baseline

`MV-SEGNEURON-ZERO-SHOT-000004` remains unchanged. It used real DVID-native
`MV-GTVOL-000004` raw voxels (192³), the recorded official checkpoint, 171 CPU
tiles, and ran for 207.85 seconds. Its pseudolabel and FRMC result remain
rejected by automated QA; neither is a segment, neuron, annotation, or ground
truth.

## Stage-preserving evidence

The raw prediction diagnostic is
`local_research_build/phase5e-b/MV-GTVOL-000004/raw-prediction-diagnostic.json`.
It records affinity-channel means 0.9900, 0.9940, and 0.9941, a mean-affinity
mean of 0.9927, and a foreground/interior-head mean of 0.9861. More than
99.99% of the mean affinity and foreground voxels are at least 0.5.

The exact FRMC fragment settings generate 1,188 watershed fragments on the
same grid. The baseline multicut (`beta=0.25`) merges these to one label.
Two predeclared beta-only comparisons (`beta=0.10`, `beta=0.50`) also retain
1,188 fragments and yield one final label. Thus beta is not a recovery path
for this saturated prediction.

## Activation audit

`MV-EXP-000002-activation-audit` retains a real padded DVID patch, normalized
input, pre-sigmoid head logits, activated probabilities, hashes, statistics,
and histograms. The maximum difference between the retained probability and a
single sigmoid of its captured logit is exactly 0.0.

The sampled patch has affinity-logit mean 6.846 and foreground-logit mean
7.193. Saturation therefore appears *before* activation in this sampled patch.
This is evidence against `ACTIVATION_ERROR` and double activation, not evidence
that all patches have been audited.

The patch's normalized intensity distribution is exceptionally narrow and
bright (mean 0.9484, standard deviation 0.0090). The official documentation
specifies uint8 divided by 255 but does not provide a checkpoint training-set
intensity distribution. Consequently `INPUT_NORMALIZATION_MISMATCH` and
`TARGET_DOMAIN_SHIFT` remain hypotheses requiring controlled experiments;
neither has been declared a root cause.

## Current classification

| Classification | Status |
| --- | --- |
| Activation error / double sigmoid | Not supported for sampled patch |
| Output-semantics error | Not supported; source semantics recorded in `OUTPUT_SEMANTICS.json` |
| Patch reconciliation error | Not primary; sampled pre-blend probabilities are already saturated |
| Agglomeration sensitivity | Present but beta-only changes did not recover instances |
| Input normalization mismatch | Open hypothesis |
| Target domain shift / model limitation | Open hypothesis |
| Official reference parity | Not run; no source reference input/output pair has been registered locally |

All current artifacts are `MACHINE_PSEUDOLABEL` diagnostics and are prohibited
from ground truth, held-out scoring, `MV-SEG`, biological identity, and graph
construction.

## Phase 6A baseline reproduction

`MV-EXP-6A-BASELINE` reran the exact pinned source path against the same
immutable raw cube and checkpoint. Its affinity SHA-256 is
`3b7b80ba1fa01452c3b35663a259549f9510d4f57084ecdb9cb715b6df301f34` and its
foreground TIFF SHA-256 is
`f033b6b5204cf80ba50e43c507cc636d0af9a866e5a60013e3ad244c81d40225`, both
identical to the original. `BASELINE_REPRODUCTION = PASS`.
