# Novelty audit

Audit date: 2026-09-16. Statuses are deliberately conservative and should be revisited before any manuscript or release claim.

| Region / material | Classification | Evidence and boundaries |
| --- | --- | --- |
| Compound eye and early visual system / lamina | `PUBLISHED_COMPLETE` **for the scope claimed by Chua et al. (2023)** | The article is titled “A complete reconstruction of the early visual system of an adult insect” and reports the compound eye plus lamina connectome. This status does **not** mean complete brain or CNS. |
| Published CATMAID early-visual material beyond article’s stated scope | `PUBLIC_UNPUBLISHED_RECONSTRUCTION` | Public service presence was verified, but its project content and publication relationship have not been exhaustively audited. No claim about unreported extent is made. |
| WASPSYN 14 subvolumes | `ANNOTATED_PARTIALLY` | Synapse benchmark subvolumes; five training volumes with labels under the paper’s challenge scheme. They are not a neuron reconstruction or whole nervous system. |
| Antennal lobe / mushroom body / PLP / GNG in WASPSYN | `RAW_DATA_AVAILABLE` for documented benchmark subvolumes; `ANNOTATED_PARTIALLY` where training labels exist | The dataset’s reported coverage is local subvolumes, not complete region coverage. |
| Medulla, central brain, major nerves, ventral nerve cord | `UNKNOWN` | This audit did not establish a downloadable dense reconstruction or a full proofread connectome. |
| Whole brain / complete CNS | `UNKNOWN` | No verifiable complete brain/CNS reconstruction release was established. Do not use “first connectome,” “complete connectome,” or equivalent terminology. |

## What is already reconstructed

The literature establishes a published reconstruction of the adult wasp’s early visual system, including the compound eye and lamina, and reports public CATMAID-hosted source data/reconstructions. WASPSYN supplies synapse-detection benchmark annotations from 14 regional subvolumes. Neither finding establishes an open, complete *M. viggianii* brain/CNS connectome.

## Overlap and risk

The CATMAID visual-system material and the whole-head imaging discussed in related literature may overlap, but the exact specimen, coordinate spaces, and transforms have not been verified. Treat them as distinct source records until a source-authoritative mapping exists. Unpublished server content must remain `PUBLIC_UNPUBLISHED_RECONSTRUCTION`, never be relabeled as project output.

## Novelty wording permitted today

“Vigilia Connectome provides provenance-enforced, open reconstruction infrastructure and has not released a biological reconstruction.” No stronger novelty claim is supported.
