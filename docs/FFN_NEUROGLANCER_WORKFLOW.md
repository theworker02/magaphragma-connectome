# FFN, Connectomics, and Neuroglancer workflow

This repository integrates Google's [Flood-Filling Networks](https://github.com/google/ffn) as an **external, pinned inference backend**. It does not vendor FFN, retrain it automatically, or turn its output into a biological reconstruction. FFN is TensorFlow-era research software whose upstream documentation reports Ubuntu/Tesla P100 validation and recommends a GPU (12 GB for the documented training configuration). Keep it in a dedicated environment or container and pin the checkout revision in your run notebook or lab record.

The local adapter also uses a stable, dependency-free subvolume-plan contract inspired by the bounding-box and segmentation-range primitives of [Google Research Connectomics](https://github.com/google-research/connectomics). The upstream library describes itself as under active development; it is deliberately not made a required dependency here. [Neuroglancer](https://github.com/google/neuroglancer) is the proofreading surface: its documented client can display volumetric image and segmentation sources plus line-based models, and supports Neuroglancer precomputed, N5, Zarr/OME-Zarr, DVID, and other HTTP-accessible sources.

## Scientific gate

```text
bounded FIB-SEM ingestion
  -> deterministic FFN subvolume plan
  -> VERIFIED CATMAID-nm -> FIB-SEM-voxel registration
  -> CATMAID endpoint seeds + held-out CATMAID morphology paths
  -> separately pinned FFN inference
  -> label import + held-out coverage / split / merge validation
  -> PASSED candidate-extension gate
  -> candidate review in Neuroglancer
  -> human proofreading, merge/split decision, and new evidence record
```

FFN output is always `MACHINE_ONLY`, local research only, and never modifies the source CATMAID build. A passing gate merely permits *candidate triage*; it does not establish an MV-N, MV-SYN, or MV-CONN record.

## Preconditions

1. Materialize source morphology locally with `vigilia phase4-local` (or use the existing local build).
2. Ingest the selected FIB-SEM crop through `vigilia ingest` or use its produced ingestion record.
3. Independently establish landmark-based registration from `MV-FRAME-CATMAID-001` to the specific FIB-SEM volume. The repository currently has **no verified CATMAID-to-EM transform**; this is an intentional blocker, not a missing default.
4. Copy `schemas/ffn-registration.example.json`, replace every placeholder, and provide real validation evidence. Do not use the example identity matrix as a biological transform.
5. Obtain the FFN code and compatible checkpoint separately under its Apache-2.0 terms, pinning both revision and model hash. Do not install it into this project environment by default.

## Commands

All generated artifacts should stay in ignored `derived/` or another local research directory.

```powershell
# Plan chunked, halo-aware inference units for the exact ingested data.
vigilia ffn-plan <ingestion.json> --output derived/ffn/ffn-subvolumes.json

# Refuses to export seeds unless registration status is VERIFIED and evidence-backed.
vigilia ffn-export-seeds --ingestion <ingestion.json> --registration <verified-registration.json>

# Execute the explicit external checkout. Configure the FFN inference pbtxt to write under --output.
vigilia ffn-run --ffn-checkout <path-to-pinned-google-ffn> --inference-request <request.pbtxt> --bounding-box "start { x:4000 y:6000 z:6000 } size { x:256 y:256 z:64 }" --output derived/ffn/run-001

# FFN emits NPZ files; normalize the one carrying 3-D integer segment IDs.
vigilia ffn-import-labels derived/ffn/run-001/seg-0_0_0.npz --output derived/ffn/run-001/labels_zyx.npy

# A failed gate blocks candidate-extension triage; thresholds remain in the receipt.
vigilia ffn-validate derived/ffn/run-001/labels_zyx.npy --heldout derived/ffn/seeds/ffn-heldout-morphology.json --output derived/ffn/run-001/validation.json
```

`ffn-validate` reports, for each held-out source neuron, dominant-label path coverage and the number of labels observed along that sparse morphology path. It also counts label-sharing pairs across source neurons as merge signals. The default requires 80% path coverage for every held-out neuron and zero merge pairs. These are initial operational thresholds, not a published error benchmark; record rationale and tune them only after representative pilot results.

## Neuroglancer proofreading

The exporter writes a standard state JSON with actual CATMAID edges as editable local line annotations. It intentionally will not invent an image URL for a local NPY cache: Neuroglancer is a client-side viewer and needs a real CORS-enabled source.

```powershell
vigilia neuroglancer-export `
  --image-source "precomputed://https://your-cors-enabled-host/megaphragma/raw" `
  --segmentation-source "precomputed://https://your-cors-enabled-host/megaphragma/ffn-run-001" `
  --ffn-validation derived/ffn/run-001/validation.json
```

Import the generated `derived/neuroglancer/megaphragma-proofreading-state.json` into a local or hosted Neuroglancer instance. Do not expose the current local source cache publicly: its rights and derivative-release status remain unresolved. If serving any data, configure CORS deliberately and expose only artifacts you are permitted to share.

## Provenance and review requirements

- Retain the ingestion checksum, registration checksum, external FFN script checksum, inference request checksum, raw FFN NPZ checksum, normalized label checksum, and validation receipt together.
- Record checkpoint identifier/hash and FFN revision in the lab run record before interpreting results.
- Preserve CATMAID source identity; no spatial match alone may establish a new neuronal identity.
- Human review must decide merges, splits, and synapse candidates before promotion. Candidate connector work remains separate from the 4,030 source-annotated synapses and 5,821 source connectors.
- The architecture remains: source morphology and FFN candidates -> reviewed segment evidence -> MV-N physical reconstruction -> source/new synapse evidence -> MV-CONN graph. No stage may skip evidence or review state.
