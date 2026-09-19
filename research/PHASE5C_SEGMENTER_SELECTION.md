# Phase 5C production-segmenter selection

## Labeled-data acquisition

Official CREMI cropped HDF5 volumes were acquired into the ignored local cache. They contain `volumes/raw` and `volumes/labels/neuron_ids`, with documented ZYX resolution 40 × 4 × 4 nm. The split is volume-separated: A train, B validation, C test. The artifacts remain local-only until their artifact-level reuse terms are captured.

## Executed benchmark

`MV-MODEL-CREMI-BOUNDARY-0001` is a real small 3-D TensorFlow boundary model trained on CREMI A and evaluated only on held-out CREMI C. Its held-out boundary Dice was **0.0002661048896773478** (TP 6, FP 6, FN 45,077) after one fixed-seed bounded CPU epoch.

## Selection decision

**No production segmenter selected.** The executed model is rejected for production because it fails the held-out benchmark and CREMI is 40 × 4 × 4 nm Drosophila ssTEM, whereas the target is 8 × 8 × 8 nm *Megaphragma* FIB-SEM. It must not run on the DVID crop.

The next gate is target-compatible labeled 3-D EM with immutable train/validation/test splits, or a pretrained model with documented 8 nm FIB-SEM-compatible training data and held-out target evaluation. No target `MV-SEG` is created by this phase.
