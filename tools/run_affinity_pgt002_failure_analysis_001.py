"""Execute AFFINITY_TRAIN_PGT002_FAILURE_ANALYSIS_001 arms A→D (frozen protocol).

Does not weaken GATE_D, promote step10, or run dense segmentation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party/segneuron/Train_and_Inference"))
sys.path.insert(0, str(REPO / "tools"))

from model.Mnet import MNet  # noqa: E402
from train_affinity_pgt002_001 import (  # noqa: E402
    INIT_CKPT,
    PATCH_ZYX,
    load_raw,
    patch_for_edge,
    sha256,
    win_to_wsl,
)

PROTOCOL = REPO / "experiments/phase6e/AFFINITY_TRAIN_PGT002_FAILURE_ANALYSIS_001.json"
SUPERVISION = REPO / "experiments/phase6e/AFFINITY-TRAIN-PGT002-001/supervision/supervision_manifest.json"
FAIL_PERM = REPO / "experiments/phase6e/AFFINITY_TRAIN_PGT002_001_FAIL_PERMANENT.json"
STEP10 = REPO / "experiments/phase6e/AFFINITY-TRAIN-PGT002-001/run/checkpoints/checkpoint-step10.pt"
OUT_ROOT = REPO / "experiments/phase6e/AFFINITY-TRAIN-PGT002-FAILANAL-001"
RESULTS = REPO / "experiments/phase6e/AFFINITY_TRAIN_PGT002_FAILURE_ANALYSIS_001_RESULTS.json"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def unit_step(channel: int) -> list[int]:
    e = [0, 0, 0]
    e[channel] = 1
    return e


def arm_a(edges: list[dict]) -> dict:
    issues = []
    for e in edges:
        left = list(e["pair_left_zyx"])
        right = list(e["pair_right_zyx"])
        ch = int(e["channel_zyx"])
        expected_right = [a + b for a, b in zip(left, unit_step(ch))]
        if right != expected_right:
            issues.append({"id": e["opaque_decision_id"], "issue": "pair_right_not_plus_e_channel", "left": left, "right": right, "channel": ch})
            continue
        raw = load_raw(e["raw_path"])
        if list(raw.shape) != list(e["shape_zyx"]):
            issues.append({"id": e["opaque_decision_id"], "issue": "shape_mismatch"})
            continue
        if sha256(win_to_wsl(e["raw_path"])) != e["raw_sha256"]:
            issues.append({"id": e["opaque_decision_id"], "issue": "raw_sha256_mismatch"})
            continue
        patch, local, origin = patch_for_edge(raw, ch, left)
        if patch.shape != PATCH_ZYX:
            issues.append({"id": e["opaque_decision_id"], "issue": "patch_shape", "got": list(patch.shape)})
            continue
        c, lz, ly, lx = local
        if not (0 <= c < 3 and 0 <= lz < PATCH_ZYX[0] and 0 <= ly < PATCH_ZYX[1] and 0 <= lx < PATCH_ZYX[2]):
            issues.append({"id": e["opaque_decision_id"], "issue": "local_oob", "local": list(local)})
            continue
        # reconstruct absolute from origin+local and compare to left
        abs_zyx = [origin[0] + lz, origin[1] + ly, origin[2] + lx]
        if abs_zyx != left:
            issues.append({"id": e["opaque_decision_id"], "issue": "origin_local_not_left", "abs": abs_zyx, "left": left})
            continue
        if int(e["target"]) not in (0, 1):
            issues.append({"id": e["opaque_decision_id"], "issue": "bad_target"})
            continue
        if (e["decision"] == "SAME_PROCESS") != (int(e["target"]) == 1):
            issues.append({"id": e["opaque_decision_id"], "issue": "target_decision_mismatch"})

    # step10 scalar-index consistency on eval edges (construction path only)
    index_ok = True
    if STEP10.exists() and torch.cuda.is_available():
        model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().eval()
        state = torch.load(win_to_wsl(STEP10), map_location="cuda", weights_only=False)
        model.load_state_dict(state["model_weights"], strict=True)
        with torch.no_grad():
            for e in edges:
                if e["source_id"] not in {"MV-DVID-RAW-SURVEY-078", "MV-DVID-RAW-SURVEY-083"}:
                    # still check all edges for indexability
                    pass
                raw = load_raw(e["raw_path"])
                patch, local, _ = patch_for_edge(raw, int(e["channel_zyx"]), list(e["pair_left_zyx"]))
                aff, _ = model(torch.from_numpy(patch[None, None]).cuda())
                c, lz, ly, lx = (int(x) for x in local)
                scalar = aff[0, c, lz, ly, lx].detach().cpu().reshape(-1)
                if scalar.numel() != 1:
                    issues.append({"id": e["opaque_decision_id"], "issue": "non_scalar_affinity_read"})
                    index_ok = False

    outcome = "CONSTRUCTION_OK" if not issues else "CONSTRUCTION_DEFECT"
    return {
        "arm_id": "A_CONSTRUCTION_AUDIT",
        "outcome": outcome,
        "n_edges_audited": len(edges),
        "n_issues": len(issues),
        "issues": issues[:50],
        "step10_scalar_index_ok": index_ok and not any(i.get("issue") == "non_scalar_affinity_read" for i in issues),
        "supports_F3": outcome == "CONSTRUCTION_DEFECT",
    }


class FeatureMNet(MNet):
    """Return pre-head feature map (up14 sum) plus affinity, for frozen probes."""

    def forward_features(self, x):
        down11 = self.down11(x)
        down12 = self.down12(down11[0])
        down13 = self.down13(down12[0])
        down14 = self.down14(down13[0])
        bottleNeck1 = self.bottleneck1(down14[0])
        down21 = self.down21(down11[1])
        down22 = self.down22([down21[0], down12[1]])
        down23 = self.down23([down22[0], down13[1]])
        bottleNeck2 = self.bottleneck2([down23[0], down14[1]])
        down31 = self.down31(down21[1])
        down32 = self.down32([down31[0], down22[1]])
        bottleNeck3 = self.bottleneck3([down32[0], down23[1]])
        down41 = self.down41(down31[1])
        bottleNeck4 = self.bottleneck4([down41[0], down32[1]])
        bottleNeck5 = self.bottleneck5(down41[1])
        up41 = self.up41([bottleNeck4[0], down41[0], bottleNeck5, down41[1]])
        up31 = self.up31([bottleNeck3[0], down32[0], bottleNeck4[1], down32[1]])
        up32 = self.up32([up31[0], down31[0], up41, down31[1]])
        up21 = self.up21([bottleNeck2[0], down23[0], bottleNeck3[1], down23[1]])
        up22 = self.up22([up21[0], down22[0], up31[1], down22[1]])
        up23 = self.up23([up22[0], down21[0], up32, down21[1]])
        up11 = self.up11([bottleNeck1, down14[0], bottleNeck2[1], down14[1]])
        up12 = self.up12([up11, down13[0], up21[1], down13[1]])
        up13 = self.up13([up12, down12[0], up22[1], down12[1]])
        up14 = self.up14([up13, down11[0], up23, down11[1]])
        feat = up14[0] + up14[1]
        aff = self.sigmoid(self.outputs[0](feat))
        return feat, aff


def extract_voxel_features(model: FeatureMNet, edges: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, ids = [], [], []
    model.eval()
    with torch.no_grad():
        for e in edges:
            raw = load_raw(e["raw_path"])
            patch, local, _ = patch_for_edge(raw, int(e["channel_zyx"]), list(e["pair_left_zyx"]))
            feat, _ = model.forward_features(torch.from_numpy(patch[None, None]).cuda())
            c, lz, ly, lx = (int(v) for v in local)
            # channel-conditioned: concat feature at voxel with one-hot channel
            f = feat[0, :, lz, ly, lx].detach().float().cpu().numpy()
            onehot = np.zeros(3, dtype=np.float32)
            onehot[c] = 1.0
            # also include raw intensity at left and right endpoints in patch coords
            # right along channel within patch if in-bounds
            inten = [float(patch[lz, ly, lx])]
            rz, ry, rx = lz, ly, lx
            if c == 0 and lz + 1 < patch.shape[0]:
                rz = lz + 1
            elif c == 1 and ly + 1 < patch.shape[1]:
                ry = ly + 1
            elif c == 2 and lx + 1 < patch.shape[2]:
                rx = lx + 1
            inten.append(float(patch[rz, ry, rx]))
            vec = np.concatenate([f, onehot, np.asarray(inten, dtype=np.float32)])
            xs.append(vec)
            ys.append(int(e["target"]))
            ids.append(e["opaque_decision_id"])
    return np.stack(xs), np.asarray(ys, dtype=np.int64), ids


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray, seed: int = 20260918) -> tuple[np.ndarray, float]:
    """L2 logistic via IRLS / newton; returns (weights including bias), for binary y in {0,1}."""
    rng = np.random.default_rng(seed)
    n, d = x_train.shape
    X = np.concatenate([x_train, np.ones((n, 1), dtype=np.float64)], axis=1)
    w = np.zeros(d + 1, dtype=np.float64)
    y = y_train.astype(np.float64)
    lam = 1e-2
    for _ in range(100):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = p * (1 - p) + 1e-6
        grad = X.T @ (p - y) + lam * w
        # Hessian diagonal approx + X'WX
        H = (X.T * W) @ X + lam * np.eye(d + 1)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w = w - step
        if np.linalg.norm(step) < 1e-8:
            break
    return w, lam


def predict_proba(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    X = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    z = X @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def probe_metrics(proba: np.ndarray, y: np.ndarray) -> dict:
    same = proba[y == 1]
    diff = proba[y == 0]
    same_acc = float(np.mean(same >= 0.5)) if len(same) else float("nan")
    diff_acc = float(np.mean(diff < 0.5)) if len(diff) else float("nan")
    balanced = float(np.nanmean([same_acc, diff_acc]))
    margin = float(np.mean(same) - np.mean(diff)) if len(same) and len(diff) else float("nan")
    const_same = float(np.mean(y == 1))
    return {
        "SAME_acc_at_0p5": same_acc,
        "DIFF_acc_at_0p5": diff_acc,
        "balanced_acc_at_0p5": balanced,
        "margin_meanSAME_minus_meanDIFF": margin,
        "constant_same_accuracy": const_same,
        "n": int(len(y)),
        "n_same": int((y == 1).sum()),
        "n_diff": int((y == 0).sum()),
    }


def load_feature_model(ckpt_path: Path) -> FeatureMNet:
    model = FeatureMNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().eval()
    state = torch.load(win_to_wsl(ckpt_path), map_location="cuda", weights_only=False)
    model.load_state_dict(state["model_weights"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def arm_b(train_edges: list[dict], eval_edges: list[dict]) -> dict:
    results = {}
    for name, ckpt in (("init_step2250", INIT_CKPT), ("failed_step10", STEP10)):
        model = load_feature_model(ckpt)
        x_tr, y_tr, _ = extract_voxel_features(model, train_edges)
        x_ev, y_ev, _ = extract_voxel_features(model, eval_edges)
        w, lam = fit_logistic(x_tr, y_tr)
        proba = predict_proba(w, x_ev)
        metrics = probe_metrics(proba, y_ev)
        results[name] = {
            "checkpoint": str(ckpt).replace("\\", "/"),
            "checkpoint_sha256": sha256(win_to_wsl(ckpt)),
            "feature": "MNet up14 sum (kn0=32) at supervised voxel + channel one-hot + endpoint intensities",
            "l2_lambda": lam,
            "eval_metrics": metrics,
        }

    init_m = results["init_step2250"]["eval_metrics"]
    step_m = results["failed_step10"]["eval_metrics"]
    init_sep = init_m["margin_meanSAME_minus_meanDIFF"] > 0 and init_m["balanced_acc_at_0p5"] > init_m["constant_same_accuracy"]
    step_sep = step_m["margin_meanSAME_minus_meanDIFF"] > 0 and step_m["balanced_acc_at_0p5"] > step_m["constant_same_accuracy"]
    if init_sep and not step_sep:
        outcome = "PROBE_SEPARATES_AT_INIT_BUT_NOT_AT_STEP10"
    elif init_sep or step_sep:
        outcome = "PROBE_MARGIN_POSITIVE_AND_BALANCED_GT_CONSTANT"
    else:
        outcome = "PROBE_NO_SEPARATION"
    return {"arm_id": "B_LINEAR_PROBE_ON_FROZEN_FEATURES", "outcome": outcome, "per_checkpoint": results}


def make_local_patch(raw: np.ndarray, channel: int, left: list[int], size: int) -> tuple[np.ndarray, tuple]:
    """Cubic/crop patch of given ZYX size centered on left (clamped)."""
    z, y, x = left
    shape = raw.shape
    # keep Z at least size but survey Z=64
    ez = min(size, shape[0])
    ey = min(size, shape[1])
    ex = min(size, shape[2])
    origin = [
        max(0, min(z - ez // 2, shape[0] - ez)),
        max(0, min(y - ey // 2, shape[1] - ey)),
        max(0, min(x - ex // 2, shape[2] - ex)),
    ]
    slices = tuple(slice(o, o + e) for o, e in zip(origin, (ez, ey, ex)))
    patch = np.asarray(raw[slices], dtype=np.float32) / 255.0
    local = (channel, z - origin[0], y - origin[1], x - origin[2])
    return patch, local


def extract_raw_descriptor(edges: list[dict], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Scale ablation without MNet: local raw neighborhood descriptor (flatten + mean/std)."""
    xs, ys = [], []
    for e in edges:
        raw = load_raw(e["raw_path"])
        patch, local = make_local_patch(raw, int(e["channel_zyx"]), list(e["pair_left_zyx"]), size)
        c, lz, ly, lx = local
        # fixed-size descriptor: stats + small window around voxel (pad if needed)
        win = 5
        z0, z1 = max(0, lz - win), min(patch.shape[0], lz + win + 1)
        y0, y1 = max(0, ly - win), min(patch.shape[1], ly + win + 1)
        x0, x1 = max(0, lx - win), min(patch.shape[2], lx + win + 1)
        nb = patch[z0:z1, y0:y1, x0:x1].astype(np.float64)
        onehot = np.zeros(3, dtype=np.float64)
        onehot[c] = 1.0
        vec = np.concatenate(
            [
                onehot,
                [
                    float(patch[lz, ly, lx]),
                    float(nb.mean()),
                    float(nb.std()),
                    float(nb.min()),
                    float(nb.max()),
                    float(nb.shape[0] * nb.shape[1] * nb.shape[2]),
                ],
                # intensity difference along channel if possible
                [
                    float(
                        abs(
                            patch[lz, ly, lx]
                            - patch[
                                lz + (1 if c == 0 and lz + 1 < patch.shape[0] else 0),
                                ly + (1 if c == 1 and ly + 1 < patch.shape[1] else 0),
                                lx + (1 if c == 2 and lx + 1 < patch.shape[2] else 0),
                            ]
                        )
                    )
                ],
            ]
        )
        xs.append(vec)
        ys.append(int(e["target"]))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def arm_c(train_edges: list[dict], eval_edges: list[dict]) -> dict:
    # Read-only scale ablation on raw descriptors (one factor: neighborhood size)
    scales = [16, 32, 64, 128]
    per = {}
    best = None
    for s in scales:
        x_tr, y_tr = extract_raw_descriptor(train_edges, s)
        x_ev, y_ev = extract_raw_descriptor(eval_edges, s)
        w, _ = fit_logistic(x_tr, y_tr)
        metrics = probe_metrics(predict_proba(w, x_ev), y_ev)
        per[str(s)] = metrics
        score = (metrics["margin_meanSAME_minus_meanDIFF"], metrics["balanced_acc_at_0p5"])
        if best is None or score > best[0]:
            best = (score, s, metrics)

    base = per["128"]  # comparable to full YX crop extent
    small = per["16"]
    mid = per["32"]
    # Classify relative to base margin
    def helps(m):
        return m["margin_meanSAME_minus_meanDIFF"] > base["margin_meanSAME_minus_meanDIFF"] + 1e-6

    if helps(small) and small["margin_meanSAME_minus_meanDIFF"] > 0:
        outcome = "SMALLER_CONTEXT_HELPS"
    elif helps(mid) and mid["margin_meanSAME_minus_meanDIFF"] > max(base["margin_meanSAME_minus_meanDIFF"], 0):
        outcome = "SMALLER_CONTEXT_HELPS"
    elif best[1] == 128 and base["margin_meanSAME_minus_meanDIFF"] > 0:
        outcome = "LARGER_CONTEXT_HELPS"  # full context best among tested with positive margin
    elif best[1] >= 64 and helps(per[str(best[1])]) and best[1] > 32:
        outcome = "LARGER_CONTEXT_HELPS"
    else:
        # no scale yields clear positive margin improvement pattern
        any_pos = any(per[k]["margin_meanSAME_minus_meanDIFF"] > 0 for k in per)
        outcome = "LARGER_CONTEXT_HELPS" if (any_pos and best[1] >= 64) else "NO_SCALE_HELPS"
        if any_pos and best[1] <= 32:
            outcome = "SMALLER_CONTEXT_HELPS"
        elif not any_pos:
            outcome = "NO_SCALE_HELPS"

    return {
        "arm_id": "C_SCALE_ABLATION_READ_ONLY_THEN_ONE_FACTOR",
        "outcome": outcome,
        "descriptor": "raw local neighborhood stats + channel one-hot + face step (no MNet)",
        "scales_zyx_cap": scales,
        "per_scale_eval": per,
        "best_scale": best[1],
    }


def arm_d(train_edges: list[dict], eval_edges: list[dict]) -> dict:
    # Nested train subsets on arm-B init features (construction OK assumed)
    model = load_feature_model(INIT_CKPT)
    # deterministic order by opaque id
    train_sorted = sorted(train_edges, key=lambda e: hashlib.sha256(e["opaque_decision_id"].encode()).hexdigest())
    x_all, y_all, _ = extract_voxel_features(model, train_sorted)
    x_ev, y_ev, _ = extract_voxel_features(model, eval_edges)
    sizes = [6, 12, 18]
    curve = []
    for n in sizes:
        w, _ = fit_logistic(x_all[:n], y_all[:n])
        metrics = probe_metrics(predict_proba(w, x_ev), y_ev)
        curve.append({"n_train_edges": n, "eval_metrics": metrics})
    margins = [c["eval_metrics"]["margin_meanSAME_minus_meanDIFF"] for c in curve]
    # increasing if last > first by margin delta
    if margins[-1] > margins[0] + 0.02 and margins[-1] > 0:
        outcome = "MARGIN_INCREASES_WITH_N"
    elif margins[-1] > margins[0] + 0.02:
        outcome = "MARGIN_INCREASES_WITH_N"
    else:
        outcome = "MARGIN_FLAT"
    return {
        "arm_id": "D_SUPERVISION_SUFFICIENCY_BOUND",
        "outcome": outcome,
        "feature_source": "init_step2250 frozen MNet voxel features",
        "curve": curve,
    }


def synthesize(arm_results: dict) -> dict:
    a = arm_results["A"]["outcome"]
    b = arm_results["B"]["outcome"]
    c = arm_results["C"]["outcome"]
    d = arm_results["D"]["outcome"]
    supported = []
    if a == "CONSTRUCTION_DEFECT":
        supported.append("F3_INPUT_OR_TARGET_CONSTRUCTION")
    if b == "PROBE_NO_SEPARATION":
        supported.extend(["F1_INSUFFICIENT_PRODUCTION_SUPERVISION", "F4_SPATIAL_SCALE_SIGNAL", "F5_ARCHITECTURE_FEATURE_EXTRACTION"])
    if b == "PROBE_MARGIN_POSITIVE_AND_BALANCED_GT_CONSTANT":
        supported.append("F2_OPTIMIZATION_OR_CONFIG")
    if b == "PROBE_SEPARATES_AT_INIT_BUT_NOT_AT_STEP10":
        supported.append("F2_OPTIMIZATION_OR_CONFIG")
    if c in ("SMALLER_CONTEXT_HELPS", "LARGER_CONTEXT_HELPS"):
        supported.append("F4_SPATIAL_SCALE_SIGNAL")
    if c == "NO_SCALE_HELPS" and "F4_SPATIAL_SCALE_SIGNAL" in supported:
        supported = [s for s in supported if s != "F4_SPATIAL_SCALE_SIGNAL"]
    if d == "MARGIN_INCREASES_WITH_N":
        supported.append("F1_INSUFFICIENT_PRODUCTION_SUPERVISION")
    if d == "MARGIN_FLAT" and "F1_INSUFFICIENT_PRODUCTION_SUPERVISION" in supported and b == "PROBE_NO_SEPARATION":
        # demote F1 as sole cause
        supported = [s for s in supported if s != "F1_INSUFFICIENT_PRODUCTION_SUPERVISION"] + [
            "F1_NOT_SOLE_CAUSE_PER_ARM_D"
        ]

    # unique preserve order
    seen = set()
    ordered = []
    for s in supported:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    next_action = "NEW_EXPERIMENT_CONTRACT_CITING_IDENTIFIED_FAILURE_CLASS"
    if a == "CONSTRUCTION_DEFECT":
        next_action = "FIX_CONSTRUCTION_UNDER_NEW_FROZEN_CONTRACT"
    elif "F2_OPTIMIZATION_OR_CONFIG" in ordered and b != "PROBE_NO_SEPARATION":
        next_action = "NEW_TRAIN_CONTRACT_ONE_FACTOR_OPTIM_CHANGE"
    elif "F1_INSUFFICIENT_PRODUCTION_SUPERVISION" in ordered:
        next_action = "ACQUIRE_MORE_INDEPENDENT_PRODUCTION_GT"
    elif "F4_SPATIAL_SCALE_SIGNAL" in ordered:
        next_action = "NEW_TRAIN_CONTRACT_ONE_FACTOR_SCALE_CHANGE"
    elif "F5_ARCHITECTURE_FEATURE_EXTRACTION" in ordered or b == "PROBE_NO_SEPARATION":
        next_action = "NEW_TRAIN_CONTRACT_OR_FEATURE_PATH_AFTER_F5"

    return {
        "supported_failure_classes": ordered,
        "recommended_next_action": next_action,
        "still_forbidden": [
            "Weaken GATE_D",
            "Promote checkpoint-step10",
            "Dense segmentation",
            "Identical longer retrain as next move",
        ],
    }


def main() -> int:
    if RESULTS.exists() or OUT_ROOT.exists():
        raise FileExistsError("Refusing overwrite of failure-analysis outputs")
    if not FAIL_PERM.exists():
        raise SystemExit("permanent fail artifact missing")
    if not torch.cuda.is_available():
        raise SystemExit("ROCm/CUDA required for arms B–D feature extraction")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sup = json.loads(SUPERVISION.read_text(encoding="utf-8"))
    train_edges = list(sup["train_edges"])
    eval_edges = list(sup["eval_edges"])
    all_edges = train_edges + eval_edges

    OUT_ROOT.mkdir(parents=True)

    print("Arm A...", flush=True)
    a = arm_a(all_edges)
    (OUT_ROOT / "arm_A_construction_audit.json").write_text(json.dumps(a, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if a["outcome"] == "CONSTRUCTION_DEFECT":
        summary = {
            "id": "AFFINITY_TRAIN_PGT002_FAILURE_ANALYSIS_001_RESULTS",
            "created_at": _now(),
            "status": "STOPPED_AFTER_ARM_A_CONSTRUCTION_DEFECT",
            "arms": {"A": a},
            "synthesis": synthesize({"A": a, "B": {"outcome": "SKIPPED"}, "C": {"outcome": "SKIPPED"}, "D": {"outcome": "SKIPPED"}}),
        }
        RESULTS.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary["synthesis"], indent=2))
        return 2

    print("Arm B...", flush=True)
    b = arm_b(train_edges, eval_edges)
    (OUT_ROOT / "arm_B_linear_probe.json").write_text(json.dumps(b, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Arm C...", flush=True)
    c = arm_c(train_edges, eval_edges)
    (OUT_ROOT / "arm_C_scale_ablation.json").write_text(json.dumps(c, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Arm D...", flush=True)
    d = arm_d(train_edges, eval_edges)
    (OUT_ROOT / "arm_D_supervision_sufficiency.json").write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    arms = {"A": a, "B": b, "C": c, "D": d}
    synthesis = synthesize(arms)
    summary = {
        "id": "AFFINITY_TRAIN_PGT002_FAILURE_ANALYSIS_001_RESULTS",
        "created_at": _now(),
        "status": "COMPLETE",
        "protocol_id": protocol["id"],
        "parent_failed_run": "AFFINITY_TRAIN_PGT002_001_FAIL_PERMANENT",
        "arms": arms,
        "synthesis": synthesis,
        "gate_d_unchanged": True,
        "dense_segmentation": "STILL_CLOSED",
    }
    RESULTS.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol_path = PROTOCOL
    proto = json.loads(protocol_path.read_text(encoding="utf-8"))
    proto["status"] = "EXECUTED_RESULTS_RECORDED"
    proto["results_path"] = str(RESULTS.relative_to(REPO)).replace("\\", "/")
    protocol_path.write_text(json.dumps(proto, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "synthesis": synthesis, "A": a["outcome"], "B": b["outcome"], "C": c["outcome"], "D": d["outcome"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
