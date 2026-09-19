# DVID / SegNeuron input comparison

`MV-GTVOL-000004` satisfies the source-required dtype (`uint8`), array order
(`ZYX`), and official `raw / 255` normalization. The baseline is exactly
reproducible: both affinity and foreground output artifacts match the original
SHA-256 byte-for-byte.

The activation audit nevertheless found an unusually bright, narrow sampled
normalized DVID patch (mean 0.9484, standard deviation 0.0090). The source
does not publish a numerical training-intensity distribution, so this is an
observed target characteristic rather than a proven mismatch. It is sufficient
to keep `INPUT_NORMALIZATION_MISMATCH` and `TARGET_DOMAIN_SHIFT` open, but not
to justify unrecorded normalization, clipping, or polarity changes.
