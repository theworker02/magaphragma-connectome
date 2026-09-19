# Canonical coordinate system

Vigilia uses **zero-based voxel-center coordinates** in source stack space.

| Name | Convention |
| --- | --- |
| Public API / evidence / viewer | `(x, y, z)` voxel coordinates, zero-based integers at voxel centers |
| In-memory NumPy arrays | `(z, y, x)` indexing, explicitly converted at each boundary |
| Physical coordinates | `(x_nm, y_nm, z_nm) = origin_nm + voxel_xyz × resolution_nm` |
| Bounds | Half-open: `[x0, x1)`, `[y0, y1)`, `[z0, z1)` |
| CATMAID DVID XY tile | tile column `floor(x / tile_width)`, row `floor(y / tile_height)`, depth `z` |

For `MV-FIBSEM-0416-XY`, CATMAID reports an identity stack translation, orientation `XY`, and 8 nm isotropic resolution. Therefore source, internal, and viewer coordinates are equal at level 0. The implementation still records the transform explicitly; consumers must not infer this for another source.

Voxel *boundaries* lie one-half voxel from centers. A level-0 source coordinate `(4000, 6000, 6000)` maps to physical center `(32000, 48000, 48000)` nm.

Round trips are exact for integer coordinates under an identity/integer affine transform: `source → internal → source`. Physical conversions may involve floating-point values and are compared within a tolerance.
