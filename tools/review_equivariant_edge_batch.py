"""Batch equivariant reviewer: axis-blind smoke (12) or SPEC002 validation (36)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from mvconnectome.equivariant_edge_presentation import render_edge_presentation
from mvconnectome.external_boundary_review import append_external_boundary_event

ELIGIBLE_STATUS = {
    "EXPERT_EQUIVARIANT_SMOKE_REVIEW_REQUIRED": {
        "n": 12,
        "title": "Presentation-repair smoke — axis-blind (12 decisions)",
        "done": "All 12 smoke decisions are recorded. Close the window, then run finalize_g3_presentation_repair_smoke.py",
        "dock": "Equivariant smoke decision",
        "batch_key": "smoke_batch_id",
    },
    "EXPERT_EQUIVARIANT_VALIDATION_REVIEW_REQUIRED": {
        "n": 36,
        "title": "SPEC002 validation — axis-blind (36 decisions)",
        "done": "All 36 validation decisions are recorded. Close the window, then run the matching finalize_affinity_spec002_validation*.py for this batch.",
        "dock": "Equivariant validation decision",
        "batch_key": "validation_batch_id",
    },
    "EXPERT_EQUIVARIANT_DISCOVERY_REVIEW_REQUIRED": {
        "n": 12,
        "title": "DIFFERENT-class discovery — axis-blind (abs≥12 faces)",
        "done": "All discovery decisions are recorded. Close the window, then run finalize_different_class_discovery.py",
        "dock": "Equivariant discovery decision",
        "batch_key": "discovery_batch_id",
    },
    "EXPERT_EQUIVARIANT_PRODUCTION_REVIEW_REQUIRED": {
        "n": 24,
        "title": "Production GT candidate — axis-blind (24 decisions)",
        "done": "All 24 production decisions recorded. Close window, then run the matching finalize_production_gt_batch_*.py",
        "dock": "Production GT decision",
        "batch_key": "production_batch_id",
    },
}


def channel_from_pair(left: list[int], right: list[int]) -> int:
    diffs = [int(r) - int(l) for l, r in zip(left, right)]
    if diffs.count(1) != 1 or sum(1 for d in diffs if d == 0) != 2:
        raise ValueError(f"pair is not a canonical +1 edge: {left} -> {right}")
    return diffs.index(1)


def _show(label: QLabel, image: np.ndarray) -> None:
    image = np.ascontiguousarray(image)
    qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(qimage.copy()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("master_queue", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--radius", type=int, default=12)
    args = parser.parse_args()

    queue = json.loads(args.master_queue.read_text(encoding="utf-8"))
    meta = ELIGIBLE_STATUS.get(queue.get("status"))
    if meta is None or not queue.get("axis_blind"):
        raise SystemExit("Master queue not eligible")
    questions = sorted(queue["questions"], key=lambda q: int(q["presentation_order_index"]))
    expected_n = int(queue.get("expected_n_decisions", meta["n"]))
    if len(questions) != expected_n:
        raise SystemExit(f"expected {expected_n} questions, got {len(questions)}")

    # Cache workspaces/raw by crop
    workspaces: dict[str, dict] = {}
    raws: dict[str, np.ndarray] = {}
    answered: set[str] = set()
    for q in questions:
        wp = Path(q["workspace_path"])
        ws = json.loads(wp.read_text(encoding="utf-8"))
        if ws.get("presentation_mode") != "EQUIVARIANT_EDGE_FACE_V1":
            raise SystemExit(f"{wp} missing equivariant presentation_mode")
        workspaces[q["crop_id"]] = {"path": wp, "ws": ws}
        raws[q["crop_id"]] = np.load(ws["raw"]["path"], mmap_mode="r", allow_pickle=False)
        log = Path(ws["event_log"]["path"])
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    answered.add(json.loads(line).get("question_reference"))

    import napari

    viewer = napari.Viewer(title=meta["title"])
    image_layer = viewer.add_image(
        raws[questions[0]["crop_id"]], name="raw_EM", colormap="gray",
        contrast_limits=(float(raws[questions[0]["crop_id"]].min()), float(raws[questions[0]["crop_id"]].max())),
    )

    panel = QWidget()
    layout = QVBoxLayout(panel)
    title = QLabel("Do the two marked locations belong to the same biological process/object across their shared face?")
    title.setWordWrap(True)
    help_text = QLabel(
        "SIDE A / SIDE B = transverse through each endpoint. "
        "LONGITUDINAL 1/2 = planes containing the edge. Red=A, Green=B. "
        "No axis labels. Order is blinded. UNCERTAIN / BAD QUESTION allowed."
    )
    help_text.setWordWrap(True)
    layout.addWidget(title)
    layout.addWidget(help_text)
    status = QLabel("")
    status.setWordWrap(True)
    layout.addWidget(status)

    labels = {name: QLabel() for name in ("SIDE_A", "SIDE_B", "LONGITUDINAL_1", "LONGITUDINAL_2")}
    grid = QGridLayout()
    for name, row, col in (("SIDE_A", 0, 0), ("SIDE_B", 0, 1), ("LONGITUDINAL_1", 1, 0), ("LONGITUDINAL_2", 1, 1)):
        cell = QVBoxLayout()
        cell.addWidget(QLabel(name.replace("_", " ")))
        labels[name].setMinimumSize(240, 240)
        labels[name].setScaledContents(True)
        cell.addWidget(labels[name])
        grid.addLayout(cell, row, col)
    layout.addLayout(grid)

    rationale = QLineEdit()
    rationale.setPlaceholderText("Optional rationale; do not name axes")
    layout.addWidget(rationale)

    nav = QHBoxLayout()
    prev_b, next_b = QPushButton("Previous"), QPushButton("Next")
    nav_label = QLabel("")
    nav.addWidget(prev_b)
    nav.addWidget(nav_label)
    nav.addWidget(next_b)
    layout.addLayout(nav)

    buttons = QHBoxLayout()
    same, different, uncertain, bad = QPushButton("SAME"), QPushButton("DIFFERENT"), QPushButton("UNCERTAIN"), QPushButton("BAD QUESTION")
    for b in (same, different, uncertain, bad):
        buttons.addWidget(b)
    layout.addLayout(buttons)

    index = [0]
    done_msg = meta["done"]
    batch_id = queue.get(meta["batch_key"])

    def pending() -> list[int]:
        return [i for i, q in enumerate(questions) if q["opaque_decision_id"] not in answered]

    def refresh() -> None:
        pend = pending()
        if not pend:
            status.setText(done_msg)
            return
        if index[0] not in pend:
            index[0] = pend[0]
        q = questions[index[0]]
        crop = q["crop_id"]
        raw = raws[crop]
        image_layer.data = raw
        image_layer.contrast_limits = (float(raw.min()), float(raw.max()))
        left = list(q["pair_left_zyx"])
        right = list(q["pair_right_zyx"])
        channel = channel_from_pair(left, right)
        pres = render_edge_presentation(raw, tuple(left), channel, radius=args.radius)
        for name in pres.panel_order:
            _show(labels[name], pres.panels[name])
        viewer.dims.set_point((0, 1, 2), tuple(float(v) for v in left))
        pos = pend.index(index[0]) + 1
        nav_label.setText(f"{pos} / {len(pend)} remaining  •  {q['opaque_decision_id']}")
        status.setText(f"Opaque id {q['opaque_decision_id']}. FOV {pres.fov_nm:.0f} nm.")
        prev_b.setEnabled(pend.index(index[0]) > 0)
        next_b.setEnabled(pend.index(index[0]) < len(pend) - 1)

    def move(delta: int) -> None:
        pend = pending()
        if not pend:
            return
        pos = pend.index(index[0]) if index[0] in pend else 0
        index[0] = pend[max(0, min(len(pend) - 1, pos + delta))]
        refresh()

    def decide(value: str) -> None:
        pend = pending()
        if not pend or index[0] not in pend:
            return
        q = questions[index[0]]
        ws_info = workspaces[q["crop_id"]]
        left = list(q["pair_left_zyx"])
        channel = channel_from_pair(left, list(q["pair_right_zyx"]))
        event = append_external_boundary_event(
            ws_info["path"],
            reviewer=args.reviewer,
            decision=value,
            pair_left_zyx=left,
            channel_zyx=channel,
            rationale=rationale.text(),
            orthogonal_views_inspected=True,
            question_reference=q["opaque_decision_id"],
            interface_id=batch_id,
        )
        answered.add(q["opaque_decision_id"])
        rationale.setText("")
        status.setText(f"Recorded {value} for {q['opaque_decision_id']} ({event['id']}).")
        pend = pending()
        if pend:
            index[0] = pend[0]
            refresh()
        else:
            status.setText(done_msg)

    prev_b.clicked.connect(lambda: move(-1))
    next_b.clicked.connect(lambda: move(1))
    same.clicked.connect(lambda: decide("SAME_PROCESS"))
    different.clicked.connect(lambda: decide("DIFFERENT_PROCESS"))
    uncertain.clicked.connect(lambda: decide("UNCERTAIN"))
    bad.clicked.connect(lambda: decide("BAD_QUESTION"))

    if pending():
        index[0] = pending()[0]
        refresh()
    viewer.window.add_dock_widget(panel, area="right", name=meta["dock"])
    napari.run()


if __name__ == "__main__":
    main()
