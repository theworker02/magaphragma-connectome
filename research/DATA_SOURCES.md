# Data sources and access audit

Audit date: 2026-09-16. This document records what was actually established, not what a paper might imply. The operational source registry is [datasets/registry.json](../datasets/registry.json).

## MV-SRC-LAMINA-CATMAID — *Megaphragma viggianii* early visual system

| Field | Audited record |
| --- | --- |
| Dataset name | *Megaphragma viggianii* early visual system CATMAID source |
| Authors / institution | Chua, Makarova, Gunn, Chklovskii and collaborators; Flatiron Institute/source collaboration |
| DOI / publication | [10.1016/j.cub.2023.09.021](https://doi.org/10.1016/j.cub.2023.09.021); Chua et al., *Current Biology* 33(21), 2023 |
| Acquisition | Paper reports whole-head serial electron microscopy; its data-availability statement directs users to a CATMAID server. |
| Voxel resolution / physical volume | Not independently captured from a machine-readable source in this audit; therefore `UNKNOWN` in the operational registry. |
| Format / size | CATMAID-hosted imagery, neuron reconstructions, and synapse annotations reported; export format and size `UNKNOWN`. |
| Access | Public landing endpoint verified: [waspem-lamina.flatironinstitute.org](https://waspem-lamina.flatironinstitute.org/). It served CATMAID’s public interface at audit time. |
| License / redistribution | `UNKNOWN` for the imagery/annotation export. The analysis-code GitHub repository displays CC0-1.0, but that is not evidence of data redistribution rights. |
| Anatomical coverage | Published compound eye and lamina/early visual system. |
| Existing annotation | Published neuron reconstructions and synapse annotations are reported by the article. |
| Limitations | No immutable data artifact URL, export authorization, checksum, coordinate transform, or complete-volume download was established. The source is reference-only until those facts are recorded. |

## MV-SRC-WASPSYN-2024 — WASPSYN

| Field | Audited record |
| --- | --- |
| Dataset name | WASPSYN: microwasp synapse-detection benchmark |
| Authors / institution | Li, Li, Chen, Huang, Zou, Xiao, Shinomiya, Gunn, Gupta, Polilov, Xu, Zhang, Xiong, Pfister, Wei, Wu; multi-institutional collaboration |
| DOI / publication | [10.1109/TMI.2024.3400276](https://doi.org/10.1109/TMI.2024.3400276); *IEEE Transactions on Medical Imaging* 43(11), 3719–3730 (2024) |
| Acquisition | FIB-SEM of heavy-metal-stained, resin-embedded *M. viggianii* whole heads. |
| Voxel resolution | Isotropic 8 × 8 × 8 nm. |
| Physical volume | 14 annotated subvolumes from three whole-brain datasets. The paper tabulates dimensions; for example one 400 × 400 × 400 and several 416 × 416 × 416 voxel volumes. Do not extrapolate this to whole-head coverage. |
| Format / size | Artifact packaging and sizes not independently verified. |
| Access | Paper says challenge data remain publicly available and identifies its CodaLab challenge distribution. A stable direct archive URL has not yet been pinned in this repository. |
| License | The primary paper states CC-BY. Exact artifact-level license text/version must be preserved before downloading or redistributing any copy. |
| Anatomical coverage | Includes mushroom body medial lobe, antennal lobe, posterior lateral protocerebrum, gnathal ganglion, and other annotated brain-region samples. |
| Existing annotation | Five sample-3 training volumes have pre-/postsynapse annotations and one-to-many partner information. Test labels are not public under challenge design. |
| Existing segmentation / synapses | No segmentation claim is made here. Synapse annotation is present only where documented. |
| Limitations | The dataset is a benchmark subvolume collection, not evidence that a full brain connectome is available. No direct URL/checksum is yet authorized in `datasets/registry.json`. |

## Source verification procedure

Before changing a registry record to `download_approved: true`, archive the artifact landing page/license statement, preserve the upstream filename and SHA-256 (or calculate only after comparison to authoritative checksum), record access date, and obtain written clarification if redistribution is unclear. The `vigilia download` command refuses every other record.
