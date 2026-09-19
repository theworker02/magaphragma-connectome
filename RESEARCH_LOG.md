# Research log

## 2026-09-16 — programme initialization

- Verified that `waspem-lamina.flatironinstitute.org` responds with a CATMAID application and public landing interface. Its terms, export permissions, complete-volume artifact URL, and machine-download interface were not established in this audit.
- Reviewed Chua et al. (2023): it reports a whole-head serial EM acquisition and a complete early-visual-system/lamina reconstruction; the article directs readers to the CATMAID server and analysis code.
- Reviewed Li et al. (2024): it reports WASPSYN, 14 FIB-SEM subvolumes from three *M. viggianii* whole-brain datasets, 8 nm isotropic voxels, public continued availability under CC-BY, and labeled training material. Stable direct archive URLs and checksums remain to be captured before downloading.
- Created software foundation with an empty biological registry. No source imaging was downloaded and no biological claims were generated.

## 2026-09-16 — Phase 3 synapse gate

- Confirmed from published source descriptions that the lamina CATMAID source reports reconstructions and synapse annotations, and that WASPSYN training volumes report pre/post point/partner annotations.
- Did not import either annotation set: source-export terms, immutable artifacts/checksums, coordinate mapping, and overlap with the local Phase 2 DVID crop remain unverified.
- Recorded a hard acquisition gate in `research/PHASE3_SYNAPSE_AUDIT.md`. No biological connection or graph edge has been generated.
