# Independent neuronal-EM boundary-model audit 001

Status: `INDEPENDENT_PRETRAINED_MODEL_PATH_BLOCKED_AFTER_BOUNDED_AUDIT`.

This audit started after the raw-EM pilot failed to discover a reviewed
`DIFFERENT_PROCESS` pair. No audited checkpoint is now eligible to produce
independent target-boundary evidence. None may be used as ground truth, a
biological record, or an automatic approval signal.

| Field | Recorded value |
| --- | --- |
| Model | PyTorch Connectomics BANIS+ NISB checkpoint (`base_banis+_seed42.ckpt`) |
| Repository | `https://github.com/PytorchConnectomics/pytorch_connectomics` |
| Pinned source commit | `652b9795fb1b5bc696054d8f75fcdeb1087a9cd0` |
| Task | Six-channel neuronal affinity prediction, then connected-components decoding |
| Training domain | NISB/BANIS neuronal EM |
| Training resolution | 9 × 9 × 20 nm (anisotropic) |
| Target | *Megaphragma* DVID FIB-SEM, 8 × 8 × 8 nm (isotropic) |
| Architecture | MedNeXt-L/k3, per-channel BCE, EMA; documented by the tutorial |
| Public checkpoint | `pytc/tutorial`, `neuron_nisb/base_banis+_seed42.ckpt` |
| Checkpoint SHA-256 | `df4d0b20f4878244c4a92c5306b5351ecf41adcf57e6d427141779633e5dbf29` |
| Checkpoint size | 742,117,621 bytes |
| Model-card license | MIT (`README.md` SHA-256 `d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17`) |
| Code license | MIT |
| Local storage | `local_research_build/phase6e/independent_model_candidates/pytc_banis/` — excluded from releases |
| AMD/ROCm compatibility | Unknown; must be proven by isolated model construction and real tensor inference |

## PyTorch Connectomics outcome

`RUNTIME_QUALIFIED_TARGET_COMPATIBILITY_REJECTED`.

The checkpoint was independently loaded and executed on the AMD ROCm runtime.
The non-protected DVID smoke output was nevertheless near-uniform high
affinity: the interior had essentially no low-affinity boundary evidence and
its low values were border-concentrated. It is preserved as a negative result,
not a source of labels or review candidates.

## Bounded follow-up audit

Bootstrapper was rejected because its FIB-SEM corrector consumes upstream 2-D
affinities/LSDs rather than raw EM, and the declared artifacts remained Git-LFS
pointers. Funke LSD provides method code but no documented released 3-D
pretrained inference checkpoint. MOSS bundles a 15 MB `lsd_mtlsd_checkpoint.pth`
(`2a5a2df513c4dc669fb161f452c1dad98dd96302976b90864c240c6cc2482fac`), but
it is a 2-D Y/X-affinity wrapper with undocumented training data, voxel scale,
and weight license. It was rejected before runtime work.

The next permitted route is a compact, independently reviewed DVID
same/different boundary-decision package. This is not a fallback claim that
machine labels are truth: the external reviewer supplies the missing
independent biological evidence. `MV-GTVOL-000004` remains excluded.

Sources: [official PyTorch Connectomics repository](https://github.com/PytorchConnectomics/pytorch_connectomics), [official NISB tutorial](https://github.com/PytorchConnectomics/pytorch_connectomics/tree/main/tutorials/neuron_nisb).
