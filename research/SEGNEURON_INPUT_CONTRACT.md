# SegNeuron input contract

## Source-confirmed

The pinned SegNeuron source requires a nonempty grayscale `uint8` volume in
`(z, y, x)` order, with no channel or time axis. Its inference path converts
the input with `float32(raw) / 255.0`; it does not resize, resample, standardize,
clip, invert, or reinterpret axes. It uses overlapping `20 x 128 x 128` ZYX
patches with stride `10 x 64 x 64` and Gaussian blending.

MNet applies one sigmoid internally to the affinity and auxiliary heads.
Affinities are merge probabilities for Z/Y/X-neighbor offsets. The auxiliary
head (historically named `boundaries.tif`) is foreground/interior confidence,
not an inverted membrane probability.

### Channel-to-physical-offset audit

The upstream `mknhood3d(1)` returns the three offsets, in array `ZYX` order,
as `[-1, 0, 0]`, `[0, -1, 0]`, and `[0, 0, -1]`. `seg_to_affgraph` consumes
these against a segmentation with shape `(Z, Y, X)`, and the retained
inference receipt records the same ordered list. Under this project's canonical
`XYZ` evidence convention, the channel-to-physical mapping is therefore:

| CZYX channel | Array offset (Z,Y,X) | Physical voxel offset (X,Y,Z) |
| --- | --- | --- |
| `C0` | `(-1, 0, 0)` | `(0, 0, -1)` |
| `C1` | `(0, -1, 0)` | `(0, -1, 0)` |
| `C2` | `(0, 0, -1)` | `(-1, 0, 0)` |

The RAG adapter must pass the offsets in the `ZYX` order above. This verifies
that the upstream first affinity channel is the inter-slice Z neighbor; it does
not establish that a particular model's predicted values are biologically
accurate.

## Documented

The project documentation describes the target sampling regime as approximately
5–10 nm in X/Y. DVID is 8 nm isotropic and thus is within the nominal X/Y
range, but this is not a validation of transferability.

## Unknown

Neither the checked-out inference source nor checkpoint metadata provides a
training-intensity mean, standard deviation, clipping range, polarity contract,
or a registered reference input/output pair. Therefore an inversion,
standardization, percentile clipping, or alternate axis permutation is not a
source-supported correction at this time.
