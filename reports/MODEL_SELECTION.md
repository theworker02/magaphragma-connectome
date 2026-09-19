# Production model selection

**Status: no production model selected.**

The candidate `MV-MODEL-SEGNEURON-0001` is a documented MIT-licensed
pretrained model with training material that includes FIB-25 at 8 nm
isotropic FIB-SEM. Its hash-registered checkpoint has completed a bounded
real-DVID technical affinity-inference run (`MV-SEGNEURON-TECH-0001`) on CPU.
That makes it a technically relevant *candidate*, not a qualified production
segmenter for the DVID target. It must still be evaluated against frozen
same-grid neuronal instance labels. The gate requires raw merge, split,
missed-object, false-object, and variation-of-information results, with
checkpoint selection based on validation only and one retained held-out test.

WASPSYN remains relevant for a future synapse detector. Its published labels
are synaptic pre/post point-and-partner annotations, not neuronal-instance
masks; it cannot satisfy the current segmenter ground-truth gate.
