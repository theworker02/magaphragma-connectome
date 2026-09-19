# Vigilia Connectome

> **Takeover handoff — unfinished but substantial.**  
> Most of the heavy lifting is done (evidence core, Affinity S7 FAST path, Vast/local hybrid tooling, AWS Affinity decommissioned). The full-volume affinity campaign (~28.8k chunks) is **not** finished. If you want to continue the work, start with **[HANDOFF.md](HANDOFF.md)** for what's done, what's left, and how to run key commands. This GitHub tree is intentionally **lean**: no secrets, no multi-GB affinity dumps, no checkpoints.

## Evidence-first infrastructure for the *Megaphragma viggianii* connectome

Vigilia Connectome is local-first research infrastructure for creating a defensible connectome only when every supporting datum, coordinate transform, model output, review decision, and release claim can be traced and independently inspected. It is not a synthetic-connectome demo, a generic graph viewer, or a pipeline that treats a convincing model mask as biology.

The central design decision is simple: a plausible result is not automatically a biological result. Raw imagery, local caches, published source annotations, machine predictions, human-review events, and release-eligible biological entities remain separate artifacts with explicit transitions between them. A missing source, checksum, rights statement, transform, or review is visible evidence of uncertainty—not a blank to be filled with inference.

> **Current status:** No Vigilia biological reconstruction release exists. The repository contains no invented neurons, synapses, connectivity edges, segmentation outputs, or demonstration biological metrics. Local research artifacts are not evidence of redistribution permission, cross-dataset registration, or scientific-release eligibility.

### Affinity S7 cloud compute

| Path | Status |
| --- | --- |
| **Vast.ai** | Production cloud provider — [docs/VAST_AFFINITY.md](docs/VAST_AFFINITY.md), hybrid fleet [docs/HYBRID_FLEET.md](docs/HYBRID_FLEET.md) |
| **Local AMD ROCm** | Free concurrent worker on the **same** shared claim queue |
| **AWS (EC2/ASG/DDB/S3)** | **DECOMMISSIONED** for Affinity production — [docs/AWS_DECOMMISSION.md](docs/AWS_DECOMMISSION.md) |

Hard cloud budget: `$160` (`AFFINITY_CLOUD_BUDGET_USD`). Cost model: [docs/AFFINITY_COST_MODEL.md](docs/AFFINITY_COST_MODEL.md). Historical AWS fleet experiments remain on disk as evidence; they are not an active execution path.

## What has passed, and what remains intentionally constrained

“Passed” below means only that the stated technical or provenance boundary was demonstrated. It never establishes a neighbouring biological claim.

| Boundary or capability | State | Meaning and limit |
| --- | --- | --- |
| DVID parent metadata and far-edge raw-byte read | Passed, local-only | The reported parent extent and a far-boundary raw read were verified. The whole volume was not downloaded or processed. |
| Phase 5C production-domain plan | Passed | A deterministic **28,798-item** resumable queue covers a 16,648 × 13,544 × 15,401 voxel, 8 nm isotropic source domain. Queue entries are work plans, not segmentation results. |
| Bounded real raw-EM inspection | Passed, local-only | Real DVID raw material can be cached and inspected. Redistribution and derivative-release rights remain unresolved. |
| CATMAID-local analysis | Passed, quarantined | The local status record contains 556 source neurons, 275,790 morphology nodes, 5,821 connectors, 4,030 resolved source synapses, and 2,562 local graph edges. These are attributed source derivatives, not a new Vigilia release. |
| CATMAID ↔ DVID registration | Intentionally blocked | No independently supported specimen/frame relationship exists. CATMAID-derived DVID seeding and physical mappings are prohibited. |
| Reviewed DVID-native instance labels | Not yet present | Raw cubes can be materialized; labels require qualified review, same-grid nonempty instances, split safety, and append-only provenance. |
| Production segmenter | None selected | The CREMI candidate was rejected on held-out evaluation. SegNeuron technical output remains machine-only until target-domain qualification succeeds. |
| FFN production use | Blocked | The external FFN environment import passed, but an appropriate pinned checkpoint is unavailable. A future passing FFN validation would only permit candidate triage. |
| Public data or derivative release | Blocked | Public readability is not evidence of export or redistribution rights. |
| G3 interface supervision | Separately documented | G3 already has dedicated code annotations and a contract; this pass deliberately leaves it unchanged. See [G3 interface supervision](research/G3_INTERFACE_SUPERVISION.md). |

The authoritative local inventory is [CONNECTOME_STATUS.md](reports/CONNECTOME_STATUS.md). The machine-readable production boundary is [MV-PRE-PRODUCTION-BASELINE.json](production/MV-PRE-PRODUCTION-BASELINE.json). The capability and blocker ledgers are [CAPABILITY_MATRIX.md](research/CAPABILITY_MATRIX.md) and [BLOCKER_RESOLUTION.md](research/BLOCKER_RESOLUTION.md).

## The non-negotiable evidence chain

~~~text
source and rights evidence
        │
        ▼
immutable source declaration + checksum
        │
        ▼
local raw/cache artifact ──► machine diagnostic or proposal
        │                           │
        │                           └── never a neuron, synapse, or connection by itself
        ▼
coordinate-frame and split-safety validation
        │
        ▼
append-only qualified human review evidence
        │
        ▼
evidence-backed biological candidate
        │
        ▼
release-integrity gate ──► immutable release artifact
~~~

The implementation enforces six rules.

1. **Unknown biology stays UNKNOWN.** Empty scientific registries are legitimate, test-covered results.
2. **Machine output never self-promotes.** Affinities, segments, pseudolabels, fragments, and ranked questions remain machine evidence until an independent recorded review supplies valid evidence.
3. **Coordinates are evidence, not decoration.** The project does not guess XYZ versus ZYX, nanometres versus voxels, or a cross-specimen registration.
4. **Review is append-only.** Workspace edits record who decided what, when, and against which question; they cannot silently rewrite raw imagery or create biological records.
5. **A public response is not automatically redistributable.** Availability through HTTP, a paper, or a website does not establish data-export or derivative-release rights.
6. **Releases are immutable and auditable.** Stable identifiers, evidence joins, checksums, and integrity tests reject synthetic IDs, orphan records, broken provenance, and invalid coordinates.

## Architecture and responsibilities

| Area | Responsibility | Boundary it maintains |
| --- | --- | --- |
| src/mvconnectome/models.py | Typed domain objects, statuses, namespace validation, JSON conversion | Prevents missing evidence or inadequate review from masquerading as biology. |
| ids.py, io.py, coordinates.py | Stable IDs, checksums/atomic JSON, explicit transforms | Makes identity, durable receipts, and frame conversion inspectable. |
| datasets.py, ingestion.py, catmaid.py | Source registry, checksum-gated ingestion, bounded CATMAID/DVID access | Prevents implicit source selection and filename-derived physical metadata. |
| ground_truth.py, annotation_crops.py, spatial_expansion.py | Crop plans, materialization, buffers, region/split isolation | Keeps training, validation, and protected regression material separated. |
| proofreading.py, membrane_pilot.py, external_boundary_review.py, reviewed_affinity.py | Raw-EM questions, append-only decisions, masked targets | Separates review evidence from model output and biological promotion. |
| auto_consensus.py, pair_supervision.py, pseudolabels.py, g2_*.py | Conservative proposal and affinity-supervision utilities | Produces explicit non-ground-truth artifacts only under frozen manifests. |
| segneuron*.py, segmentation.py, native_reconstruction.py, ffn.py | Technical inference adapters and machine-only baselines | Preserves model/run receipts and refuses to equate execution with qualification. |
| phase4.py, phase5_campaign.py, production_domain.py | Local source build, Phase 5 queue, source-domain inspection | Maintains quarantine and the difference between planned and processed work. |
| qualification.py, integrity.py, reporting.py, repository.py, api.py | Metrics, invariants, reports, read model, loopback explorer | Reports what is known and retains honest empty states. |
| tools/ and tests/ | One-purpose operations plus contract/regression coverage | Keeps audit actions explicit and safety failures verified. |

### Data types that must never be collapsed

| Artifact | It is | It is not |
| --- | --- | --- |
| Source declaration | A registry record with URL, terms, checksum, and access metadata | Permission to download arbitrary URLs or redistribute a source |
| Local cache | A bounded local source copy with a receipt | A public mirror or a biological assertion |
| Machine proposal | An affinity, label, fragment, candidate question, or diagnostic | Ground truth, a verified instance, or a Vigilia entity |
| Review event | An append-only human decision tied to a workspace/question | A substitute for source provenance or automatic promotion |
| Masked affinity target | Local SAME/DIFFERENT/IGNORE supervision from reviewed direct pairs | Dense instance ground truth or a neuron segmentation |
| Biological record | An MV-prefixed object with required evidence and review state | A source row, unreviewed prediction, or synthetic fixture |
| Local source-connectome build | Quarantined, attributed CATMAID derivative | A releasable Vigilia reconstruction |

Synthetic fixtures use a distinct namespace and never enter the biological-record path. This prevents test data gaining accidental release eligibility from a renamed field or loosened status check.

## Implemented programme phases

- **Source, ingestion, and Phase 2:** declared sources, checksums, local ingestion receipts, coordinate metadata, and bounded DVID navigation form a trustworthy local starting point. The cache is still non-release-eligible.
- **CATMAID-local / Phase 4:** source records may be cached and analysed locally with their source identity preserved. This does not establish CATMAID-to-DVID correspondence or release rights.
- **Phase 5C production-domain planning:** the reported parent extent is indexed as a resumable denominator for future work. It does not state that multi-terabyte imagery has been acquired.
- **Ground truth, annotation crops, and G2:** DVID-native plans are split-safe and buffer-aware. Review-created affinity targets are sparse direct supervision, not dense instance labels.
- **Model candidates and diagnostics:** FFN, SegNeuron, ELF, classical watershed, and independent-model work produce receipts, diagnostics, or local candidates. No backend is currently qualified as the production segmenter.
- **G3 interface supervision:** this already-documented workflow is outside this annotation pass. Its raw-EM interface groups, direct pair decisions, contradiction rejection, and exclusion of MV-GTVOL-000004 are defined in [its contract](research/G3_INTERFACE_SUPERVISION.md).

## Setup and verification

Python 3.11+ is required. This is a src layout: install the package in editable mode or set PYTHONPATH when running directly from a checkout.

~~~powershell
# Recommended development setup
python -m pip install -e .

# Test the maintained source tree.
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v

# Inspect declared sources and current integrity/provenance state.
vigilia sources
vigilia validate-manifest datasets/registry.json
vigilia verify
vigilia provenance-report
~~~

During this documentation pass, the source-tree unittest command completed **35 tests successfully** in the active environment. Running without editable installation or PYTHONPATH=src cannot import mvconnectome; that is an invocation configuration issue, not a scientific result. Optional workflows may require NumPy, PyArrow, Napari, TensorFlow, or separately pinned model environments; the project intentionally does not hide those requirements behind automatic installation.

Run the read-only local explorer with:

~~~powershell
vigilia serve
~~~

It intentionally retains empty states and must never invent biological metrics to make an interface look complete.

## Practical, safe workflows

### Ingest a legitimately acquired volume

Create an authority-backed sidecar from [volume-metadata.example.json](schemas/volume-metadata.example.json), then run:

~~~powershell
vigilia ingest <path-to-volume> --metadata <metadata.json>
~~~

The command refuses to infer voxel size, origin, coordinate frame, licence, or checksum from the filename. A mismatch or missing field is a deliberate failure.

### Materialize and review planned DVID material

~~~powershell
vigilia ground-truth-materialize
vigilia annotation-crops-materialize
vigilia annotation-crop-register <crop-id> <labels_zyx.npy> --reviewer <reviewer-id>
~~~

Materializing raw cubes creates no labels. Registration rejects protected regression cubes, empty or mismatched arrays, invalid review input, and unsafe split relationships. Read [PROOFREADING_WORKFLOW.md](docs/PROOFREADING_WORKFLOW.md) before using the review flow.

### Create review evidence rather than train on guesses

~~~powershell
vigilia proofread-workspace-create <crop-id> --supervoxels <supervoxels.npy> --source-method <method> --source-run-id <run-id> --output <workspace>
vigilia proofread-rank <workspace> --output <queue.json>
vigilia proofread-event <workspace> --event <review-event.json>
vigilia proofread-materialize-affinity <workspace> --output <targets>
~~~

The queue only orders questions. A human review event supplies the decision. Materialization produces a masked local affinity target, not a biological reconstruction.

### Use FFN only through its external, validation-gated adapter

See [FFN_NEUROGLANCER_WORKFLOW.md](docs/FFN_NEUROGLANCER_WORKFLOW.md). FFN requires a separately pinned checkout, a verified CATMAID-to-volume registration, and a compatible checkpoint. Those prerequisites are not satisfied today; do not substitute an identity transform or an untracked local model path.

## Repository map

~~~text
src/mvconnectome/      implementation and CLI
tests/                 contract/regression tests; G3 stays separately annotated
tools/                 reproducible operational and audit entry points
datasets/              source declarations and ignored local caches
ground_truth/          plans, manifests, and review receipts
production/            immutable pre-production and ground-truth baselines
registration/          coordinate-frame and cross-dataset decisions
research/              capability, blocker, source, method, and phase records
reports/               generated/local status and validation reports
schemas/               human-editable input contracts and examples
docs/                  workflow and API documentation
viewer/                local read-only explorer assets
~~~

Cached raw imagery, generated arrays, model checkpoints, virtual environments, and vendored external research projects are not maintained application code and are neither annotated nor redistributed by this documentation pass. Their provenance belongs in manifests and receipts.

## Documentation and annotation policy

This README is the overview. The evidence-nearest records remain authoritative:

- [START_HERE.md](START_HERE.md) — smallest safe first milestone.
- [DATA_SOURCES.md](research/DATA_SOURCES.md) and [DATA_CITATION.md](DATA_CITATION.md) — source and citation obligations.
- [COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md) and [CATMAID_DVID_REGISTRATION.md](research/CATMAID_DVID_REGISTRATION.md) — frame semantics and the current registration prohibition.
- [PHASE5C_PRODUCTION_DOMAIN.md](research/PHASE5C_PRODUCTION_DOMAIN.md) — work queue and source-domain boundary.
- [PHASE5D_GROUND_TRUTH_AUDIT.md](research/PHASE5D_GROUND_TRUTH_AUDIT.md) — ground-truth qualification state.
- [MODEL_SELECTION.md](reports/MODEL_SELECTION.md) and [PRODUCTION_MODEL_HELDOUT_EVALUATION.md](reports/PRODUCTION_MODEL_HELDOUT_EVALUATION.md) — candidate-model evidence and rejection/qualification status.

Maintained non-G3 source, tools, tests, the Phase 5C benchmark, and the viewer are documented at their module/API boundaries. Annotations focus on the data contract, scientific meaning, and—most importantly—the claim the code deliberately does **not** make. Existing G3 code remains untouched because it already has dedicated in-code and research documentation.

## Collaboration and support

I'm actively looking for an **electron microscopy (EM) partner**. If we can find one to collaborate with directly, we can release more open-sourced bug brains—more connectomes reconstructed and shared openly for anyone to build on. If you work with EM imaging, or know someone who does, please reach out.

Building this also had real out-of-pocket cost (roughly $160 USD in AI credits spent working through blockers). If the project is useful to you and you'd like to help offset that, sponsorship or a small contribution is genuinely appreciated and completely optional. Details are in [SUPPORT.md](SUPPORT.md).

## Contributing safely

1. Identify the exact source, artifact checksum, licence/terms, coordinate frame, and intended evidence status.
2. Keep raw/source artifacts, machine proposals, human review, and biological records in separate files and types.
3. Test both the success path and the gate that must reject unsafe input.
4. Update the closest research/phase record and this overview only when programme status actually changed.
5. Never commit raw imagery, derived large arrays, secrets, credentials, virtual environments, or third-party source copies as a shortcut.

The project’s credibility comes from leaving a boundary intact when evidence is insufficient. A future release should be able to answer not only “what was reconstructed?” but also “from which immutable source, under which rights, in which coordinate frame, by which model and reviewers, and which claims were still deliberately withheld?”

