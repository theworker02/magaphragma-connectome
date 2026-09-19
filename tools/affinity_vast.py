#!/usr/bin/env python3
"""Affinity Vast.ai fleet controller — cost-first production cloud path.

Commands:
  offers | benchmark | plan | launch | status | stop | destroy-all | budget | claim-server

Hard budget: AFFINITY_CLOUD_BUDGET_USD (default 160). No silent override.
AWS is decommissioned for Affinity production — do not call AWS from this tool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

from cloud.vast_provider import (  # noqa: E402
    VastAuthError,
    VastProvider,
    load_repo_dotenv,
)

# Prefer repo `.env` so `python tools/affinity_vast.py plan` works without export.
load_repo_dotenv(REPO)

from cloud.benchmark import (  # noqa: E402
    BENCH_DIR,
    remaining_chunks,
    run_local_batch_qualification,
    write_cost_model_receipt,
)
from cloud.budget import BudgetConfig, BudgetLedger, BudgetRefused  # noqa: E402
from cloud.fleet import build_fleet_plan  # noqa: E402
from cloud.pricing import BASELINE_TILES_PER_SEC  # noqa: E402

LEDGER_PATH = BENCH_DIR / "budget_ledger.json"
SELECTED_PATH = BENCH_DIR / "VAST_SELECTED_GPU.json"
FLEET_PLAN_PATH = BENCH_DIR / "VAST_FLEET_PLAN.json"
OFFERS_PATH = BENCH_DIR / "VAST_GPU_OFFERS.json"


def _ledger() -> BudgetLedger:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    return BudgetLedger(LEDGER_PATH, BudgetConfig.from_env())


def _provider() -> VastProvider:
    return VastProvider()


def cmd_offers(args: argparse.Namespace) -> int:
    p = _provider()
    offers = p.search_offers_both_markets(
        min_vram_gb=args.min_vram,
        min_reliability=args.min_reliability,
        max_dph=args.max_dph,
        limit=args.limit,
    )
    rem = remaining_chunks()
    tps = float(args.tps)
    ranked = p.rank_offers(
        offers,
        default_tiles_per_second=tps,
        remaining_chunks=rem,
    )
    rows = []
    for o in ranked[: args.limit]:
        cm = o.raw.get("_rank_cost_model") or {}
        rows.append(
            {
                **{k: getattr(o, k) for k in (
                    "offer_id", "gpu_name", "num_gpus", "vram_gb", "dph_total",
                    "reliability", "interruptible", "verified", "geolocation",
                )},
                "assumed_tiles_per_second": o.raw.get("_assumed_tiles_per_second"),
                "effective_cost_per_chunk": o.raw.get("_effective_cost_per_chunk"),
                "projected_remaining_cost_usd": o.raw.get("_projected_remaining_cost_usd"),
                "cost_model": cm,
            }
        )
    payload = {
        "remaining_chunks": rem,
        "assumed_tiles_per_second": tps,
        "n_offers": len(rows),
        "offers": rows,
    }
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    OFFERS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OFFERS_PATH), "n_offers": len(rows), "top": rows[:5]}, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Local qualification by default. Remote Vast GPU rent requires --rent and budget."""
    ledger = _ledger()
    if args.rent:
        snap = ledger.reserve_benchmark()
        print(json.dumps({"benchmark_reserve": snap}, indent=2))
        print(
            "Remote Vast $5 campaign: use offers + launch with --max-gpus 1 after "
            "confirming a candidate. Automatic remote GPU rent is not performed here "
            "without an explicit launch.",
            flush=True,
        )
        return 0
    report = run_local_batch_qualification(backend=args.backend)
    best = report.get("best") or {}
    if best.get("tiles_per_second"):
        # Cost model with placeholder $0 local; for cloud projection use --dph
        dph = float(args.dph)
        write_cost_model_receipt(
            tiles_per_second=float(best["tiles_per_second"]),
            gpu_hourly_usd=dph,
            gpu_name=str(best.get("device") or "local"),
            extra={"source": "local_batch_qualification", "batch": best.get("batch")},
        )
        SELECTED_PATH.write_text(
            json.dumps(
                {
                    "status": "LOCAL_QUALIFIED" if dph <= 0 else "CANDIDATE",
                    "gpu_name": best.get("device"),
                    "batch": best.get("batch"),
                    "backend": best.get("backend"),
                    "tiles_per_second": best.get("tiles_per_second"),
                    "dph_assumed": dph,
                    "equivalence_pass": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"best": best, "n_runs": len(report.get("runs", []))}, indent=2))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    ledger = _ledger()
    p = _provider()
    tps = float(args.tps)
    if SELECTED_PATH.exists():
        sel = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
        if sel.get("tiles_per_second"):
            tps = float(sel["tiles_per_second"])
    try:
        plan = build_fleet_plan(
            provider=p,
            ledger=ledger,
            remaining_chunks=remaining_chunks(),
            tiles_per_second=tps,
            budget_usd=args.budget,
            target_hours=args.target_hours,
            max_gpus=args.max_gpus,
            max_price_per_hour=args.max_price_per_hour,
            min_reliability=args.min_reliability,
            prefer_interruptible=args.interruptible,
        )
    except BudgetRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    FLEET_PLAN_PATH.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan.to_dict(), indent=2))
    print("\nplan does NOT launch instances.", flush=True)
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    ledger = _ledger()
    p = _provider()
    if not FLEET_PLAN_PATH.exists() and not args.offer_id:
        print("Run `plan` first, or pass --offer-id", file=sys.stderr)
        return 2
    if args.offer_id:
        offer_ids = [int(args.offer_id)]
        dph = float(args.dph or 0.0)
        hours = float(args.hours)
        n = 1
        gpu_name = args.gpu_name or "unknown"
        tps = float(args.tps)
        rem = remaining_chunks()
        from cloud.pricing import estimate_chunk_economics

        model = estimate_chunk_economics(tiles_per_second=tps, gpu_hourly_usd=dph, remaining_chunks=rem)
        projected = model.projected_remaining_cost_usd
        plan_dict = {
            "qualified_gpu": gpu_name,
            "n_gpus": 1,
            "offer_ids": offer_ids,
            "dph_per_gpu": dph,
            "fleet_hourly_usd": dph,
            "tiles_per_sec_per_gpu": tps,
            "aggregate_tiles_per_sec": tps,
            "chunks_per_hour": model.chunks_per_hour,
            "remaining_chunks": rem,
            "estimated_completion_hours": model.projected_remaining_gpu_hours,
            "estimated_compute_cost_usd": projected,
            "estimated_ancillary_cost_usd": 0.0,
            "estimated_total_cost_usd": projected,
            "budget_remaining_usd": ledger.snapshot()["remaining_usd"],
            "cost_per_chunk": model.effective_cost_per_chunk,
        }
    else:
        plan_dict = json.loads(FLEET_PLAN_PATH.read_text(encoding="utf-8"))
        offer_ids = list(plan_dict["offer_ids"])[: int(args.max_gpus)]
        dph = float(plan_dict["dph_per_gpu"])
        hours = float(args.hours or plan_dict.get("estimated_completion_hours") or 24)
        n = len(offer_ids)
        projected = float(plan_dict["estimated_total_cost_usd"])

    print("=== LAUNCH PREVIEW (no instances yet) ===", flush=True)
    print(json.dumps(plan_dict, indent=2), flush=True)
    if not args.yes:
        print("Refusing to provision without --yes", file=sys.stderr)
        return 3

    try:
        ledger.check_projected_total(projected, context="launch")
        snap = ledger.authorize_provision(
            hourly_usd=dph * n,
            hours=min(hours, 24.0),  # commit at most 24h initially; re-auth to extend
            context="launch",
            projected_remaining_cost_usd=projected,
        )
    except BudgetRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    created = []
    errors: list[dict[str, str]] = []
    for oid in offer_ids:
        try:
            inst = p.create_instance(
                oid,
                image=args.image,
                disk=args.disk,
                label=args.label or "affinity-s7",
                onstart=args.onstart,
            )
            created.append(inst.to_dict())
            print(f"created instance={inst.instance_id} from offer={oid}", flush=True)
        except Exception as e:
            # Keep successful creates — do not destroy them on a later offer miss.
            msg = str(e)
            print(f"launch error offer={oid}: {msg}", file=sys.stderr)
            errors.append({"offer_id": oid, "error": msg})

    if not created:
        ledger.release_commitment(dph * n * min(hours, 24.0), context="launch_all_failed")
        print(json.dumps({"created": [], "errors": errors, "budget": snap}, indent=2))
        return 1

    if errors:
        # Release commitment for offers that never became instances.
        failed_n = len(errors)
        ledger.release_commitment(
            dph * failed_n * min(hours, 24.0), context="launch_partial_release"
        )

    (BENCH_DIR / "VAST_LAUNCHED_INSTANCES.json").write_text(
        json.dumps(
            {"budget": snap, "instances": created, "errors": errors},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"created": created, "errors": errors, "budget": snap}, indent=2))
    return 0 if not errors else 0  # partial success still OK for fleet scale-up


def cmd_status(args: argparse.Namespace) -> int:
    ledger = _ledger()
    out = {"budget": ledger.snapshot()}
    try:
        out["fleet"] = _provider().fleet_status()
    except VastAuthError as e:
        out["fleet_error"] = str(e)
    print(json.dumps(out, indent=2))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    p = _provider()
    ids = [int(x) for x in args.instance_ids] if args.instance_ids else [i.instance_id for i in p.list_instances()]
    results = []
    for iid in ids:
        # Prefer destroy for Affinity — stop still bills disk. Default destroy unless --pause.
        if args.pause:
            results.append({"id": iid, "stop": p.stop_instance(iid)})
        else:
            results.append({"id": iid, "destroy": p.destroy_instance(iid)})
    print(json.dumps(results, indent=2))
    return 0


def cmd_destroy_all(args: argparse.Namespace) -> int:
    p = _provider()
    instances = p.list_instances()
    if not args.yes:
        print(json.dumps({"would_destroy": [i.instance_id for i in instances]}, indent=2))
        print("Pass --yes to destroy.", flush=True)
        return 3
    results = []
    for i in instances:
        results.append({"id": i.instance_id, "destroy": p.destroy_instance(i.instance_id)})
    print(json.dumps(results, indent=2))
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    print(json.dumps(_ledger().snapshot(), indent=2))
    return 0


def cmd_claim_server(args: argparse.Namespace) -> int:
    from cloud.claim_http import serve

    serve(args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Affinity Vast.ai cost-first fleet controller")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("offers", help="Discover and rank live Vast offers")
    p.add_argument("--min-vram", type=float, default=6.0)
    p.add_argument("--min-reliability", type=float, default=0.98)
    p.add_argument("--max-dph", type=float, default=None)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--tps", type=float, default=BASELINE_TILES_PER_SEC)
    p.set_defaults(func=cmd_offers)

    p = sub.add_parser("benchmark", help="Qualify batch/backend (local by default)")
    p.add_argument("--backend", default="eager", choices=["eager", "compile"])
    p.add_argument("--dph", type=float, default=0.0, help="Assumed $/GPU-h for cost model receipt")
    p.add_argument("--rent", action="store_true", help="Reserve $5 benchmark budget (does not auto-rent)")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("plan", help="Economic fleet plan (no launch)")
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--target-hours", type=float, default=168.0)
    p.add_argument("--max-gpus", type=int, default=20)
    p.add_argument("--max-price-per-hour", type=float, default=None)
    p.add_argument("--min-reliability", type=float, default=0.98)
    p.add_argument("--tps", type=float, default=BASELINE_TILES_PER_SEC)
    p.add_argument("--interruptible", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("launch", help="Provision from plan or --offer-id (requires --yes)")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--offer-id", type=int, default=None)
    p.add_argument("--dph", type=float, default=None)
    p.add_argument("--gpu-name", default=None)
    p.add_argument("--tps", type=float, default=BASELINE_TILES_PER_SEC)
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--max-gpus", type=int, default=20)
    p.add_argument("--image", default="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime")
    p.add_argument("--disk", type=float, default=40.0)
    p.add_argument("--label", default="affinity-s7")
    p.add_argument("--onstart", default=None)
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("status", help="Budget + fleet status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("stop", help="Destroy (default) or --pause instances")
    p.add_argument("instance_ids", nargs="*")
    p.add_argument("--pause", action="store_true", help="Stop compute only (disk still bills)")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("destroy-all", help="Destroy all Vast instances for this account key")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_destroy_all)

    p = sub.add_parser("budget", help="Show budget ledger")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("claim-server", help="Run HTTP claim coordinator for local+Vast workers")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8787)
    p.set_defaults(func=cmd_claim_server)

    return ap


def main() -> int:
    # Block accidental AWS usage from this entrypoint
    if os.environ.get("S7_CLAIM_BACKEND", "").lower() == "dynamodb":
        print(
            "ERROR: S7_CLAIM_BACKEND=dynamodb is decommissioned for Affinity. "
            "Use local or http (see docs/VAST_AFFINITY.md).",
            file=sys.stderr,
        )
        return 2
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except VastAuthError as e:
        print(f"AUTH: {e}", file=sys.stderr)
        return 4
    except BudgetRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
