# Third-party checkouts (not in this GitHub tree)

These directories are **local nested clones** on the originating machine and are
intentionally excluded from the public upload (multi-GB + nested `.git`).

Affinity S7 / SegNeuron tooling expects at least:

| Path | Role |
| --- | --- |
| `third_party/segneuron` | MNet affinity model code (`Train_and_Inference`) |
| `third_party/ffn` | Optional FFN adapter (production use still blocked) |
| `third_party/pytorch_connectomics` | Optional research reference |
| `third_party/cellpose` / `dinov3` | Optional experiments |

Place your own checkouts under `third_party/<name>/` so existing `sys.path`
inserts in `tools/` resolve. Prefer pinning a known commit if you resume training
or equivalence gates.

Do not commit nested virtualenvs or EM caches here.
