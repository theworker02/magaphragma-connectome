# External data and license policy

This repository's Apache-2.0 license covers its original software and documentation only. It does not sublicense external microscopy, annotations, published figures, or reconstruction data.

| Source | Reported terms | Redistribution position in this repository |
| --- | --- | --- |
| WASPSYN | CC-BY reported by Li et al. (2024) | Do not bundle until the exact artifact, license text/version, and provenance are recorded in the registry. Cite the dataset/paper. |
| Megaphragma lamina CATMAID | Public browser access verified; data terms not independently verified in this audit | Do not mirror or redistribute. Use source-hosted access and request/verify terms before importing. |
| Chua et al. analysis code | CC0-1.0 shown in its public GitHub repository | Code may be studied/reused under its own terms; its source microscopy and reconstructions have separate attribution/rights obligations. |

`datasets/registry.json` is authoritative for operational download permission. A source must have `download_approved: true`, an immutable direct URL, SHA-256, and exact license before the downloader will fetch it.

## Organism gallery figures (`figures/organism/`)

Openly licensed photographs and SEMs of *Megaphragma* redistributed for documentation (README gallery) are tracked separately in [figures/organism/ATTRIBUTION.md](figures/organism/ATTRIBUTION.md). Those figure licenses (CC BY 3.0 / CC BY 4.0) do **not** authorize redistribution of raw EM volumes, CATMAID exports, or WASPSYN archives.
