# HyperDrain status — Phase 2 Mass Production

## Per-GPU ceiling (measured)

Pure MNet forward on RX 7800 XT:

| batch | tiles/s | ms/tile |
|------:|--------:|--------:|
| 1 | ~19.7 | ~51 |
| 8 | ~18.1 | ~55 |
| 16 | ~17.9 | ~56 |

Packed production (~18 tiles/s) is already at this ceiling. **There is no 5× or 25× single-GPU kernel left** without changing production tile math or the model.

Historical compile (~20.8 tiles/s) is ~1.1× eager — and currently NaN-unsafe.

## What *can* deliver 5× / 25× / 30×

Fleet scaling with immutable tile math:

```text
T_wall ≈ T_single_GPU / N_effective_GPUs
```

| GPUs | wall speedup vs 1 GPU | role |
|-----:|----------------------:|------|
| 1 | 1× | this AI box |
| 5 | ~5× | small cloud fleet |
| 25 | ~25× | target band |
| 30 | ~30× | target band |

One packed worker per GPU host. Dynamic queue — no static chunk ranges.

```bash
python tools/hyperdrain.py fleet --gpus 30
python tools/hyperdrain.py production   # on each GPU host
```

## Declared so far

- `HYPERDRAIN_MASS_PRODUCTION_SPEEDUP_VALIDATED` — packed-eager pack=8, equiv PASS, ~1.07× vs warm eager on probe (pipeline win, not a new ceiling)
- Spatial dense — still experimental (RF mismatch)
- Decisive metric — verified chunks / wall-clock hour (fleet aggregate)
