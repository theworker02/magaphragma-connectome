"""Command-line composition root for the evidence-first local workflow.

The parser wires together individual subsystems but does not weaken their
domain gates. Each command owns its own provenance, eligibility, and
write-boundary validation; this module only exposes those capabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import serve
from .datasets import download_source, load_registry, validate_registry
from .ingestion import ingest_volume
from .integrity import verify
from .reporting import provenance_report
from .catmaid import fetch_dvid_raw_region, fetch_region
from .segmentation import run_watershed_baseline
from .catmaid_local_build import build as build_catmaid
from .phase4 import build_graph_products, cache_morphologies, create_local_release, freeze_baseline, materialize_physical, status_report, validate_build
from .ffn import build_seed_and_holdout_manifests, import_ffn_labels, plan_subvolumes, run_ffn, validate_ffn_run
from .neuroglancer import export_state as export_neuroglancer_state
from .production_domain import inspect_and_plan as inspect_production_domain
from .qualification import amend_baseline_test_count, evaluate_instance_files, freeze_preproduction_baseline
from .segneuron import export_dvid_smoke_input
from .ground_truth import materialize_regions, register_instance_label
from .target_adaptation import adaptation_status
from .annotation_crops import materialize_annotation_crops, register_reviewed_crop_label
from .auto_consensus import build_auto_consensus, extract_parent_affinity_crop
from .pair_supervision import build_pair_supervision
from .pseudolabels import audit_existing_proposal, generate_interior_component_proposals
from .segneuron_recovery import analyze_raw_prediction
from .proofreading import append_review_event, create_workspace, rank_boundary_candidates
from .reviewed_affinity import materialize_reviewed_affinity
from .membrane_pilot import (append_membrane_review_event, create_membrane_review_workspace,
                              generate_raw_membrane_questions, materialize_raw_membrane_pilot)
from .external_boundary_review import (append_external_boundary_event, create_external_review_workspace,
                                       create_external_scan_queue, materialize_external_boundary_affinity)


def repository_root() -> Path:
    """Return the checkout root used for deliberate, reproducible defaults."""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Parse one local command and dispatch it without bypassing safety gates."""
    root = repository_root()
    parser = argparse.ArgumentParser(prog="vigilia", description="Evidence-first Megaphragma connectome tools")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("sources", help="List audited source datasets")
    validate = subcommands.add_parser("validate-manifest", help="Validate a dataset manifest")
    validate.add_argument("path", type=Path, nargs="?", default=root / "datasets" / "registry.json")
    download = subcommands.add_parser("download", help="Resumable, checksum-verified approved download")
    download.add_argument("source_id")
    download.add_argument("destination", type=Path)
    download.add_argument("--registry", type=Path, default=root / "datasets" / "registry.json")
    ingest = subcommands.add_parser("ingest", help="Register a locally acquired microscopy artifact")
    ingest.add_argument("volume", type=Path)
    ingest.add_argument("--metadata", type=Path, required=True)
    ingest.add_argument("--output", type=Path, default=root / "datasets")
    check = subcommands.add_parser("verify", help="Fail on broken provenance or release integrity")
    check.add_argument("--registry", type=Path, default=root / "datasets" / "registry.json")
    report = subcommands.add_parser("provenance-report", help="Write an auditable source and biological-registry report")
    report.add_argument("--registry", type=Path, default=root / "datasets" / "registry.json")
    report.add_argument("--output", type=Path, default=root / "reports" / "provenance-report.json")
    fetch = subcommands.add_parser("fetch-catmaid-region", help="Fetch a documented public CATMAID tile region into ignored local cache")
    fetch.add_argument("region", type=Path, nargs="?", default=root / "datasets" / "megaphragma" / "phase2-region-001.yaml")
    fetch.add_argument("--cache", type=Path, default=root / "datasets" / "cache")
    raw_fetch = subcommands.add_parser("fetch-dvid-region", help="Fetch a bounded raw DVID region into ignored local cache")
    raw_fetch.add_argument("region", type=Path, nargs="?", default=root / "datasets" / "megaphragma" / "phase2-region-001.yaml")
    raw_fetch.add_argument("--cache", type=Path, default=root / "datasets" / "cache")
    segment = subcommands.add_parser("segment-baseline", help="Run provenance-recorded watershed baseline on a real ingested region")
    segment.add_argument("ingestion", type=Path)
    segment.add_argument("--output", type=Path, default=root / "derived" / "segmentation")
    local = subcommands.add_parser("build-catmaid-local", help="Create quarantined local source-connectome build")
    local.add_argument("--rows", type=Path, default=root / "datasets" / "cache" / "catmaid-lamina-export" / "connectors-bounds-nm.json")
    local.add_argument("--output", type=Path, default=root / "local_research_build" / "catmaid-lamina")
    phase4 = subcommands.add_parser("phase4-local", help="Materialize quarantined physical CATMAID source reconstruction")
    phase4.add_argument("--build", type=Path, default=root / "local_research_build" / "catmaid-lamina")
    phase4.add_argument("--cache", type=Path, default=root / "datasets" / "cache" / "catmaid-lamina-export")
    phase4.add_argument("--endpoint", default="https://waspem-lamina.flatironinstitute.org/1/skeletons")
    phase4.add_argument("--workers", type=int, default=2)
    phase4.add_argument("--skip-fetch", action="store_true")
    phase4.add_argument("--release", action="store_true", help="Package quarantined MV-LOCAL-CONNECTOME-0.1")
    ffn_plan = subcommands.add_parser("ffn-plan", help="Plan reproducible halo-aware FFN subvolumes from an ingested EM crop")
    ffn_plan.add_argument("ingestion", type=Path)
    ffn_plan.add_argument("--output", type=Path, default=root / "derived" / "ffn" / "ffn-subvolumes.json")
    ffn_plan.add_argument("--chunk", type=int, nargs=3, default=(256, 256, 64), metavar=("X", "Y", "Z"))
    ffn_plan.add_argument("--halo", type=int, nargs=3, default=(32, 32, 16), metavar=("X", "Y", "Z"))
    ffn_seeds = subcommands.add_parser("ffn-export-seeds", help="Export verified-registered CATMAID endpoints and held-out morphology")
    ffn_seeds.add_argument("--nodes", type=Path, default=root / "local_research_build" / "catmaid-lamina" / "skeleton_nodes.parquet")
    ffn_seeds.add_argument("--physical-neurons", type=Path, default=root / "local_research_build" / "catmaid-lamina" / "physical_neurons.json")
    ffn_seeds.add_argument("--registration", type=Path, required=True, help="Verified CATMAID-nm to FFN-volume-voxel mapping")
    ffn_seeds.add_argument("--ingestion", type=Path, required=True)
    ffn_seeds.add_argument("--output", type=Path, default=root / "derived" / "ffn" / "seeds")
    ffn_seeds.add_argument("--holdout-fraction", type=float, default=0.2)
    ffn_run = subcommands.add_parser("ffn-run", help="Run a separately pinned Google FFN checkout and retain its receipt")
    ffn_run.add_argument("--ffn-checkout", type=Path, required=True)
    ffn_run.add_argument("--inference-request", type=Path, required=True)
    ffn_run.add_argument("--bounding-box", required=True, help="FFN start/size bounding-box expression")
    ffn_run.add_argument("--output", type=Path, required=True, help="Directory configured as FFN output in the request")
    ffn_run.add_argument("--python", default=sys.executable)
    ffn_import = subcommands.add_parser("ffn-import-labels", help="Normalize a 3-D FFN NPZ segmentation to labels_zyx.npy")
    ffn_import.add_argument("source", type=Path)
    ffn_import.add_argument("--output", type=Path, required=True)
    ffn_validate = subcommands.add_parser("ffn-validate", help="Gate FFN candidate expansion on held-out CATMAID morphology")
    ffn_validate.add_argument("labels", type=Path, help="FFN labels_zyx.npy or converted segmentation volume")
    ffn_validate.add_argument("--heldout", type=Path, required=True)
    ffn_validate.add_argument("--output", type=Path, default=root / "derived" / "ffn" / "validation.json")
    ffn_validate.add_argument("--minimum-path-coverage", type=float, default=0.8)
    ffn_validate.add_argument("--maximum-merge-pairs", type=int, default=0)
    ng = subcommands.add_parser("neuroglancer-export", help="Export an importable local Neuroglancer proofreading state")
    ng.add_argument("--nodes", type=Path, default=root / "local_research_build" / "catmaid-lamina" / "skeleton_nodes.parquet")
    ng.add_argument("--edges", type=Path, default=root / "local_research_build" / "catmaid-lamina" / "skeleton_edges.parquet")
    ng.add_argument("--output", type=Path, default=root / "derived" / "neuroglancer" / "megaphragma-proofreading-state.json")
    ng.add_argument("--image-source", help="Actual CORS-enabled Neuroglancer image source (e.g. precomputed:// URL)")
    ng.add_argument("--segmentation-source", help="Actual CORS-enabled Neuroglancer segmentation source")
    ng.add_argument("--ffn-validation", type=Path)
    production = subcommands.add_parser("phase5c-plan-domain", help="Interrogate authoritative DVID parent metadata and create a resumable production-domain index")
    production.add_argument("--output", type=Path, default=root / "local_research_build" / "phase5c-production")
    production.add_argument("--cache", type=Path, default=root / "datasets" / "cache")
    baseline = subcommands.add_parser("phase5d-freeze-baseline", help="Freeze the source-verified pre-production scientific baseline")
    baseline.add_argument("--output", type=Path, default=root / "production" / "MV-PRE-PRODUCTION-BASELINE.json")
    amend = subcommands.add_parser("phase5d-amend-baseline", help="Append a factual correction to the immutable pre-production baseline")
    amend.add_argument("--baseline", type=Path, default=root / "production" / "MV-PRE-PRODUCTION-BASELINE.json")
    amend.add_argument("--output", type=Path, default=root / "production" / "MV-PRE-PRODUCTION-BASELINE-AMENDMENT-001.json")
    amend.add_argument("--observed-tests", type=int, required=True)
    evaluate = subcommands.add_parser("evaluate-instances", help="Evaluate same-grid neuronal instance labels on frozen validation or test data")
    evaluate.add_argument("truth", type=Path)
    evaluate.add_argument("prediction", type=Path)
    evaluate.add_argument("--model-id", required=True)
    evaluate.add_argument("--region-id", required=True)
    evaluate.add_argument("--split", required=True, choices=("validation", "test"))
    evaluate.add_argument("--output", type=Path, required=True)
    smoke = subcommands.add_parser("segneuron-export-smoke-input", help="Export a bounded real-DVID technical inference input with a provenance receipt")
    smoke.add_argument("ingestion", type=Path)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--receipt", type=Path, required=True)
    gt = subcommands.add_parser("ground-truth-materialize", help="Fetch the planned real DVID ground-truth source cubes into the ignored local cache")
    gt.add_argument("--plan", type=Path, default=root / "ground_truth" / "DVID_NATIVE_GT_PLAN.yaml")
    gt.add_argument("--cache", type=Path, default=root / "datasets" / "cache" / "dvid_native_ground_truth")
    gt.add_argument("--manifest", type=Path, default=root / "ground_truth" / "manifest.json")
    gt_register = subcommands.add_parser("ground-truth-register", help="Register an externally annotated, reviewer-supplied DVID-native instance-label volume")
    gt_register.add_argument("region_id")
    gt_register.add_argument("label", type=Path)
    gt_register.add_argument("--reviewer", required=True)
    gt_register.add_argument("--status", required=True)
    gt_register.add_argument("--manifest", type=Path, default=root / "ground_truth" / "manifest.json")
    gt_register.add_argument("--objects", type=Path, default=root / "ground_truth" / "objects.json")
    adaptation = subcommands.add_parser("target-adaptation-status", help="Fail-closed status of reviewed DVID-native labels required for target adaptation")
    adaptation.add_argument("--plan", type=Path, default=root / "ground_truth" / "DVID_TARGET_ADAPTATION_PLAN.json")
    adaptation.add_argument("--manifest", type=Path, default=root / "ground_truth" / "manifest.json")
    crops = subcommands.add_parser("annotation-crops-materialize", help="Materialize immutable, split-safe real-DVID crops for local review")
    crops.add_argument("--plan", type=Path, default=root / "ground_truth" / "DVID_ANNOTATION_CROP_PLAN.json")
    crops.add_argument("--parents", type=Path, default=root / "ground_truth" / "manifest.json")
    crops.add_argument("--output", type=Path, default=root / "ground_truth" / "raw" / "annotation_crops")
    crops.add_argument("--manifest", type=Path, default=root / "ground_truth" / "annotation_crops_manifest.json")
    crop_register = subcommands.add_parser("annotation-crop-register", help="Register a reviewer-corrected local DVID crop label")
    crop_register.add_argument("crop_id")
    crop_register.add_argument("label", type=Path)
    crop_register.add_argument("--reviewer", required=True)
    crop_register.add_argument("--manifest", type=Path, default=root / "ground_truth" / "annotation_crops_manifest.json")
    crop_register.add_argument("--receipt", type=Path, default=root / "ground_truth" / "review_receipts" / "last.json")
    consensus = subcommands.add_parser("auto-consensus-build", help="Build an explicitly experimental, fail-closed target pseudo-label from raw EM and SegNeuron affinity")
    consensus.add_argument("crop_id")
    consensus.add_argument("--affinity", type=Path, required=True)
    consensus.add_argument("--manifest", type=Path, default=root / "ground_truth" / "annotation_crops_manifest.json")
    consensus.add_argument("--output", type=Path, required=True)
    extract_affinity = subcommands.add_parser("auto-consensus-extract-affinity", help="Extract an aligned crop affinity from a retained parent SegNeuron inference")
    extract_affinity.add_argument("crop_id")
    extract_affinity.add_argument("--parent-affinity", type=Path, required=True)
    extract_affinity.add_argument("--manifest", type=Path, default=root / "ground_truth" / "annotation_crops_manifest.json")
    extract_affinity.add_argument("--output", type=Path, required=True)
    pairs = subcommands.add_parser("auto-pair-supervision-build", help="Build conservative three-state DVID affinity-pair supervision")
    pairs.add_argument("crop_id")
    pairs.add_argument("--affinity", type=Path, required=True)
    pairs.add_argument("--manifest", type=Path, default=root / "ground_truth" / "annotation_crops_manifest.json")
    pairs.add_argument("--output", type=Path, required=True)
    pseudo = subcommands.add_parser("pseudolabel-segneuron", help="Generate explicitly non-ground-truth proposal components and automated QA from an executed SegNeuron run")
    pseudo.add_argument("--inference", type=Path, required=True)
    pseudo.add_argument("--input-receipt", type=Path, required=True)
    pseudo.add_argument("--output", type=Path, required=True)
    pseudo.add_argument("--qa", type=Path, required=True)
    pseudo.add_argument("--threshold", type=float, default=0.5)
    pseudo.add_argument("--minimum-voxels", type=int, default=64)
    pseudo_audit = subcommands.add_parser("pseudolabel-audit", help="Run append-only severity QA on an existing machine pseudolabel")
    pseudo_audit.add_argument("proposal", type=Path)
    pseudo_audit.add_argument("--output", type=Path, required=True)
    recovery = subcommands.add_parser("segneuron-recovery-analyze", help="Diagnose immutable raw SegNeuron outputs before fragmenting or agglomeration")
    recovery.add_argument("--inference", type=Path, required=True)
    recovery.add_argument("--output", type=Path, required=True)
    proof_workspace = subcommands.add_parser("proofread-workspace-create", help="Create a machine-only DVID supervoxel proofreading workspace")
    proof_workspace.add_argument("crop_id")
    proof_workspace.add_argument("--supervoxels", type=Path, required=True)
    proof_workspace.add_argument("--source-method", required=True)
    proof_workspace.add_argument("--source-run-id", required=True)
    proof_workspace.add_argument("--manifest", type=Path, default=root / "ground_truth" / "annotation_crops_manifest.json")
    proof_workspace.add_argument("--output", type=Path, required=True)
    proof_event = subcommands.add_parser("proofread-event", help="Append one human proofreading event to a machine-only workspace")
    proof_event.add_argument("workspace", type=Path)
    proof_event.add_argument("--event", type=Path, required=True, help="JSON object containing reviewer, kind, and operation payload")
    proof_queue = subcommands.add_parser("proofread-rank", help="Rank uncertain supervoxel boundaries as review questions")
    proof_queue.add_argument("workspace", type=Path)
    proof_queue.add_argument("--affinity", type=Path)
    proof_queue.add_argument("--output", type=Path, required=True)
    proof_queue.add_argument("--maximum", type=int, default=250)
    proof_queue.add_argument("--strategy", choices=("uncertainty", "low_affinity", "raw_boundary"), default="uncertainty")
    proof_targets = subcommands.add_parser("proofread-materialize-affinity", help="Convert reviewed SAME/DIFFERENT events into immutable masked Z/Y/X targets")
    proof_targets.add_argument("workspace", type=Path)
    proof_targets.add_argument("--output", type=Path, required=True)
    membrane_raw = subcommands.add_parser("membrane-pilot-materialize", help="Materialize frozen raw DVID membrane-pilot crop")
    membrane_raw.add_argument("--amendment", type=Path, default=root / "ground_truth" / "DVID_ANNOTATION_CROP_PLAN_AMENDMENT_001.json")
    membrane_raw.add_argument("--parents", type=Path, default=root / "ground_truth" / "manifest.json")
    membrane_raw.add_argument("--output", type=Path, required=True)
    membrane_queue = subcommands.add_parser("membrane-pilot-generate", help="Generate raw-EM-only membrane crossing review questions")
    membrane_queue.add_argument("raw_receipt", type=Path)
    membrane_queue.add_argument("--output", type=Path, required=True)
    membrane_workspace = subcommands.add_parser("membrane-pilot-workspace-create", help="Create append-only raw-EM membrane review workspace")
    membrane_workspace.add_argument("--raw-receipt", type=Path, required=True)
    membrane_workspace.add_argument("--questions", type=Path, required=True)
    membrane_workspace.add_argument("--output", type=Path, required=True)
    membrane_event = subcommands.add_parser("membrane-pilot-review-event", help="Append one human raw-EM membrane decision")
    membrane_event.add_argument("workspace", type=Path)
    membrane_event.add_argument("--reviewer", required=True)
    membrane_event.add_argument("--question-id", required=True)
    membrane_event.add_argument("--decision", choices=("SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"), required=True)
    external_workspace = subcommands.add_parser("external-boundary-workspace-create", help="Create an immutable raw-EM workspace for an independent expert boundary review")
    external_workspace.add_argument("crop_id")
    external_workspace.add_argument("--request", type=Path, default=root / "ground_truth" / "EXTERNAL_DVID_BOUNDARY_REVIEW_REQUEST_001.json")
    external_workspace.add_argument("--manifest", type=Path, default=root / "ground_truth" / "annotation_crops_manifest.json")
    external_workspace.add_argument("--output", type=Path, required=True)
    external_event = subcommands.add_parser("external-boundary-review-event", help="Append an expert raw-EM SAME/DIFFERENT boundary decision")
    external_event.add_argument("workspace", type=Path)
    external_event.add_argument("--reviewer", required=True)
    external_event.add_argument("--decision", choices=("SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"), required=True)
    external_event.add_argument("--pair-left-zyx", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    external_event.add_argument("--channel-zyx", type=int, choices=(0, 1, 2), required=True)
    external_event.add_argument("--rationale")
    external_event.add_argument("--orthogonal-views-inspected", action="store_true")
    external_targets = subcommands.add_parser("external-boundary-materialize-affinity", help="Convert eligible external expert decisions into immutable masked CZYX supervision")
    external_targets.add_argument("workspace", type=Path)
    external_targets.add_argument("--output", type=Path, required=True)
    external_scan = subcommands.add_parser("external-boundary-scan-create", help="Freeze spatially stratified raw-EM navigation locations for external review")
    external_scan.add_argument("workspace", type=Path)
    external_scan.add_argument("--output", type=Path, required=True)
    external_scan.add_argument("--count", type=int, default=72)
    external_scan.add_argument("--border", type=int, default=8)
    toolchain = subcommands.add_parser("toolchain", help="Engineering toolchain (AxonForge, VoxScout, GapHound, ...)")
    toolchain.add_argument("toolchain_args", nargs=argparse.REMAINDER, help="Forwarded to connectome CLI (try: status, doctor, pipeline plan)")
    server = subcommands.add_parser("serve", help="Start the local no-fabrication explorer and query API")
    server.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args(argv)
    if arguments.command == "sources":
        for source in load_registry(root / "datasets" / "registry.json")["sources"]:
            print(f"{source['id']}\t{source['access_status']}\t{source['name']}")
        return 0
    if arguments.command == "validate-manifest":
        errors = validate_registry(arguments.path)
        print("valid" if not errors else "\n".join(errors))
        return 0 if not errors else 1
    if arguments.command == "download":
        print(download_source(arguments.registry, arguments.source_id, arguments.destination))
        return 0
    if arguments.command == "ingest":
        print(ingest_volume(arguments.volume, arguments.metadata, arguments.output))
        return 0
    if arguments.command == "verify":
        errors = verify(root, arguments.registry)
        if errors:
            print("INTEGRITY CHECK FAILED", file=sys.stderr)
            print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
            return 1
        print(json.dumps({"status": "passed", "biological_release": None, "message": "No biological records are present; empty state is valid."}))
        return 0
    if arguments.command == "provenance-report":
        errors = verify(root, arguments.registry)
        if errors:
            print("Refusing report because integrity verification failed.", file=sys.stderr)
            return 1
        print(provenance_report(root, arguments.registry, arguments.output))
        return 0
    if arguments.command == "fetch-catmaid-region":
        print(fetch_region(arguments.region, arguments.cache))
        return 0
    if arguments.command == "fetch-dvid-region":
        print(fetch_dvid_raw_region(arguments.region, arguments.cache))
        return 0
    if arguments.command == "segment-baseline":
        print(run_watershed_baseline(arguments.ingestion, arguments.output))
        return 0
    if arguments.command == "build-catmaid-local":
        print(build_catmaid(arguments.rows, arguments.output)); return 0
    if arguments.command == "phase4-local":
        baseline = freeze_baseline(arguments.build)
        validation = validate_build(arguments.build)
        if not arguments.skip_fetch:
            cache_morphologies(arguments.build, arguments.cache, arguments.endpoint, arguments.workers)
        physical = materialize_physical(arguments.build, arguments.cache, arguments.build)
        graph = build_graph_products(arguments.build, arguments.build)
        report_path = status_report(arguments.build, root / "reports" / "CONNECTOME_STATUS.md", physical, graph)
        release = create_local_release(arguments.build, root / "datasets" / "cache" / "catmaid-lamina-export" / "connectors-bounds-nm.json") if arguments.release else None
        print(json.dumps({"baseline": str(baseline), "validation": validation, "physical": physical, "graph": graph, "report": str(report_path), "release": str(release) if release else None})); return 0
    if arguments.command == "ffn-plan":
        print(plan_subvolumes(arguments.ingestion, arguments.output, tuple(arguments.chunk), tuple(arguments.halo))); return 0
    if arguments.command == "ffn-export-seeds":
        print(json.dumps({key: str(value) for key, value in build_seed_and_holdout_manifests(arguments.nodes, arguments.physical_neurons, arguments.registration, arguments.ingestion, arguments.output, arguments.holdout_fraction).items()})); return 0
    if arguments.command == "ffn-run":
        print(run_ffn(arguments.ffn_checkout, arguments.inference_request, arguments.bounding_box, arguments.output, arguments.python)); return 0
    if arguments.command == "ffn-import-labels":
        print(import_ffn_labels(arguments.source, arguments.output)); return 0
    if arguments.command == "ffn-validate":
        print(validate_ffn_run(arguments.labels, arguments.heldout, arguments.output, arguments.minimum_path_coverage, arguments.maximum_merge_pairs)); return 0
    if arguments.command == "neuroglancer-export":
        print(export_neuroglancer_state(arguments.nodes, arguments.edges, arguments.output, arguments.image_source, arguments.segmentation_source, arguments.ffn_validation)); return 0
    if arguments.command == "phase5c-plan-domain":
        print(json.dumps(inspect_production_domain(arguments.output, arguments.cache))); return 0
    if arguments.command == "phase5d-freeze-baseline":
        print(json.dumps(freeze_preproduction_baseline(root, arguments.output))); return 0
    if arguments.command == "phase5d-amend-baseline":
        print(json.dumps(amend_baseline_test_count(arguments.baseline, arguments.output, arguments.observed_tests))); return 0
    if arguments.command == "evaluate-instances":
        print(json.dumps(evaluate_instance_files(arguments.truth, arguments.prediction, arguments.output, model_id=arguments.model_id, region_id=arguments.region_id, split=arguments.split))); return 0
    if arguments.command == "segneuron-export-smoke-input":
        print(json.dumps(export_dvid_smoke_input(arguments.ingestion, arguments.output, arguments.receipt))); return 0
    if arguments.command == "ground-truth-materialize":
        print(json.dumps(materialize_regions(arguments.plan, arguments.cache, arguments.manifest))); return 0
    if arguments.command == "ground-truth-register":
        print(json.dumps(register_instance_label(arguments.manifest, arguments.region_id, arguments.label, arguments.reviewer, arguments.status, arguments.objects))); return 0
    if arguments.command == "target-adaptation-status":
        print(json.dumps(adaptation_status(arguments.plan, arguments.manifest))); return 0
    if arguments.command == "annotation-crops-materialize":
        print(json.dumps(materialize_annotation_crops(arguments.plan, arguments.parents, arguments.output, arguments.manifest))); return 0
    if arguments.command == "annotation-crop-register":
        print(json.dumps(register_reviewed_crop_label(arguments.manifest, arguments.crop_id, arguments.label, arguments.reviewer, arguments.receipt))); return 0
    if arguments.command == "auto-consensus-build":
        print(json.dumps(build_auto_consensus(arguments.manifest, arguments.crop_id, arguments.affinity, arguments.output))); return 0
    if arguments.command == "auto-consensus-extract-affinity":
        print(json.dumps(extract_parent_affinity_crop(arguments.manifest, arguments.crop_id, arguments.parent_affinity, arguments.output))); return 0
    if arguments.command == "auto-pair-supervision-build":
        print(json.dumps(build_pair_supervision(arguments.manifest, arguments.crop_id, arguments.affinity, arguments.output))); return 0
    if arguments.command == "pseudolabel-segneuron":
        print(json.dumps(generate_interior_component_proposals(arguments.inference, arguments.input_receipt, arguments.output, arguments.qa, interior_threshold=arguments.threshold, minimum_voxels=arguments.minimum_voxels))); return 0
    if arguments.command == "pseudolabel-audit":
        print(json.dumps(audit_existing_proposal(arguments.proposal, arguments.output))); return 0
    if arguments.command == "segneuron-recovery-analyze":
        print(json.dumps(analyze_raw_prediction(arguments.inference, arguments.output))); return 0
    if arguments.command == "proofread-workspace-create":
        print(json.dumps(create_workspace(crop_manifest_path=arguments.manifest, crop_id=arguments.crop_id, supervoxels_path=arguments.supervoxels, output=arguments.output, source_method=arguments.source_method, source_run_id=arguments.source_run_id))); return 0
    if arguments.command == "proofread-event":
        print(json.dumps(append_review_event(arguments.workspace, event=json.loads(arguments.event.read_text(encoding="utf-8"))))); return 0
    if arguments.command == "proofread-rank":
        print(json.dumps(rank_boundary_candidates(workspace_path=arguments.workspace, affinity_path=arguments.affinity, output=arguments.output, maximum=arguments.maximum, strategy=arguments.strategy))); return 0
    if arguments.command == "proofread-materialize-affinity":
        print(json.dumps(materialize_reviewed_affinity(arguments.workspace, arguments.output))); return 0
    if arguments.command == "membrane-pilot-materialize":
        print(json.dumps(materialize_raw_membrane_pilot(arguments.amendment, arguments.parents, arguments.output))); return 0
    if arguments.command == "membrane-pilot-generate":
        print(json.dumps(generate_raw_membrane_questions(arguments.raw_receipt, arguments.output))); return 0
    if arguments.command == "membrane-pilot-workspace-create":
        print(json.dumps(create_membrane_review_workspace(raw_receipt=arguments.raw_receipt, questions=arguments.questions, output=arguments.output))); return 0
    if arguments.command == "membrane-pilot-review-event":
        print(json.dumps(append_membrane_review_event(arguments.workspace, reviewer=arguments.reviewer, question_id=arguments.question_id, decision=arguments.decision))); return 0
    if arguments.command == "external-boundary-workspace-create":
        print(json.dumps(create_external_review_workspace(request_path=arguments.request, crop_manifest_path=arguments.manifest, crop_id=arguments.crop_id, output=arguments.output))); return 0
    if arguments.command == "external-boundary-review-event":
        print(json.dumps(append_external_boundary_event(arguments.workspace, reviewer=arguments.reviewer, decision=arguments.decision, pair_left_zyx=arguments.pair_left_zyx, channel_zyx=arguments.channel_zyx, rationale=arguments.rationale, orthogonal_views_inspected=arguments.orthogonal_views_inspected))); return 0
    if arguments.command == "external-boundary-materialize-affinity":
        print(json.dumps(materialize_external_boundary_affinity(arguments.workspace, arguments.output))); return 0
    if arguments.command == "external-boundary-scan-create":
        print(json.dumps(create_external_scan_queue(workspace_path=arguments.workspace, output=arguments.output, count=arguments.count, border=arguments.border))); return 0
    if arguments.command == "toolchain":
        from .toolchain import main as toolchain_main
        args = list(arguments.toolchain_args or [])
        # argparse REMAINDER may keep a leading '--'
        if args and args[0] == "--":
            args = args[1:]
        if not args:
            args = ["status"]
        return int(toolchain_main(args))
    if arguments.command == "serve":
        serve(root, root / "viewer", arguments.port)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
