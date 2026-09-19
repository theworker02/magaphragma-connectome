# MV-MODEL-SEGNEURON-0001 target compatibility

## Documented compatibility

The official SegNeuron repository at commit
`ea6f0c963a2f60f0f130244da27e2b2d753f8ee9` documents an MIT-licensed
checkpoint (`3767a3793cbbdbf98041b8c4c270b92d2355213d11605568c003322bb759bef5`)
and EMNeuron training material containing FIB-25, an 8 × 8 × 8 nm FIB-SEM
volume. Its inference contract accepts 3-D uint8 ZYX data, uses an affinity
model plus FRMC instance postprocessing, and states a 5–10 nm x/y target
range. The target DVID data is 8 × 8 × 8 nm uint8 FIB-SEM.

## Unresolved compatibility and mandatory evaluation

- The training specimen is not *Megaphragma* and must never transfer identity.
- The official MIT archives are acquired and indexed. `labeled.rar` contains
  raw TIFF/`*_MaskIns.tif` pairs; `valid.rar` contains FIB25 raw/label volume
  pairs. The archive hashes are recorded in the model registry.
- The archive's relationship to the checkpoint's original train/validation
  history is not independently established. Its filenames must not be treated
  as a clean selection/test split without source lineage confirmation.
- Target contrast, normalization and artifact characteristics remain pending
  visual QA; the target source remains immutable.
- No DVID-native held-out neuronal-instance labels exist yet.
- No affinity or instance inference has executed under this project.

**Decision: `PARTIAL_PENDING_ARTIFACT_INSPECTION_AND_HELDOUT_TARGET_EVALUATION`; not selected for production.**

The model can only become `VALIDATED_FOR_ASSISTED_RECONSTRUCTION` or
`VALIDATED_FOR_FIRST_PASS_PRODUCTION` after frozen validation/test labels,
raw merge/split/miss/false counts, and a retained held-out evaluation.
