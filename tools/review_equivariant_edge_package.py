"""Equivariant edge-face reviewer (presentation-repair smoke).

Shows SIDE_A / SIDE_B / LONGITUDINAL_1 / LONGITUDINAL_2 with no Z/Y/X text.
Axis mapping is sealed; only opaque decision IDs are shown.
Does not modify frozen -008 artifacts.
"""
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


def _show(label: QLabel, image: np.ndarray) -> None:
    image = np.ascontiguousarray(image)
    qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(qimage.copy()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Equivariant presentation-repair edge review")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--smoke-queue", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=12)
    args = parser.parse_args()

    workspace = json.loads(args.workspace.read_text(encoding="utf-8"))
    if workspace.get("status") != "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED":
        raise SystemExit("Workspace is not an external raw-EM review workspace")
    if workspace.get("presentation_mode") != "EQUIVARIANT_EDGE_FACE_V1":
        raise SystemExit("Workspace is not marked for equivariant edge-face presentation")

    queue = json.loads(args.smoke_queue.read_text(encoding="utf-8"))
    if queue.get("workspace_id") != workspace["id"]:
        raise SystemExit("Smoke queue workspace_id mismatch")
    if queue.get("status") != "EXPERT_EQUIVARIANT_SMOKE_REVIEW_REQUIRED":
        raise SystemExit("Queue not eligible for equivariant smoke review")
    if queue.get("axis_blind") is not True:
        raise SystemExit("Queue must be axis-blind")

    questions = list(queue["questions"])
    if not questions:
        raise SystemExit("Empty smoke queue")

    raw = np.load(workspace["raw"]["path"], mmap_mode="r", allow_pickle=False)

    import napari

    viewer = napari.Viewer(title=f"{workspace['crop_id']} — equivariant edge-face review (axis-blind)")
    # Neighborhood cube shown without world-axis labels in the dock; volume is optional context.
    viewer.add_image(raw, name="raw_EM", colormap="gray", contrast_limits=(float(raw.min()), float(raw.max())))

    panel = QWidget()
    layout = QVBoxLayout(panel)
    title = QLabel("Do the two marked locations belong to the same biological process/object across their shared face?")
    title.setWordWrap(True)
    help_text = QLabel(
        "Panels: SIDE A, SIDE B (transverse through each endpoint), LONGITUDINAL 1/2 "
        "(planes containing the edge). Red=A, Green=B. Axis identity is hidden. "
        "UNCERTAIN and BAD QUESTION are valid."
    )
    help_text.setWordWrap(True)
    layout.addWidget(title)
    layout.addWidget(help_text)

    status = QLabel("")
    status.setWordWrap(True)
    layout.addWidget(status)

    grid = QGridLayout()
    labels = {name: QLabel() for name in ("SIDE_A", "SIDE_B", "LONGITUDINAL_1", "LONGITUDINAL_2")}
    captions = {
        "SIDE_A": "SIDE A",
        "SIDE_B": "SIDE B",
        "LONGITUDINAL_1": "LONGITUDINAL 1",
        "LONGITUDINAL_2": "LONGITUDINAL 2",
    }
    positions = [("SIDE_A", 0, 0), ("SIDE_B", 0, 1), ("LONGITUDINAL_1", 1, 0), ("LONGITUDINAL_2", 1, 1)]
    for name, row, col in positions:
        cell = QVBoxLayout()
        cap = QLabel(captions[name])
        labels[name].setMinimumSize(240, 240)
        labels[name].setScaledContents(True)
        cell.addWidget(cap)
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
    same, different, uncertain, bad = (
        QPushButton("SAME"),
        QPushButton("DIFFERENT"),
        QPushButton("UNCERTAIN"),
        QPushButton("BAD QUESTION"),
    )
    for b in (same, different, uncertain, bad):
        buttons.addWidget(b)
    layout.addLayout(buttons)

    index = [0]
    # Skip already-answered opaque ids
    log_path = Path(workspace["event_log"]["path"])
    answered = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                answered.add(json.loads(line).get("opaque_decision_id"))

    def pending_indices() -> list[int]:
        return [i for i, q in enumerate(questions) if q["opaque_decision_id"] not in answered]

    def refresh() -> None:
        pend = pending_indices()
        if not pend:
            status.setText("All decisions for this workspace are recorded.")
            return
        if index[0] not in pend:
            index[0] = pend[0]
        q = questions[index[0]]
        left = tuple(q["pair_left_zyx"])
        right = tuple(q.get("pair_right_zyx") or [left[0], left[1], left[2]])
        if "pair_right_zyx" not in q:
            # derive later via channel — should always be present in smoke queues
            raise SystemExit("pair_right_zyx required")
        diffs = [int(r) - int(l) for l, r in zip(left, right)]
        channel = diffs.index(1)
        if diffs.count(1) != 1:
            raise SystemExit("invalid pair")
        pres = render_edge_presentation(raw, left, channel, radius=args.radius)
        for name in pres.panel_order:
            _show(labels[name], pres.panels[name])
        viewer.dims.set_point((0, 1, 2), tuple(float(v) for v in left))
        pos = pend.index(index[0]) + 1
        nav_label.setText(f"Decision {pos} / {len(pend)}  •  id {q['opaque_decision_id']}")
        status.setText(
            f"Opaque id {q['opaque_decision_id']}. FOV {pres.fov_nm:.0f} nm. "
            "Answer SAME or DIFFERENT process across the shared face."
        )
        prev_b.setEnabled(pend.index(index[0]) > 0)
        next_b.setEnabled(pend.index(index[0]) < len(pend) - 1)

    def move(delta: int) -> None:
        pend = pending_indices()
        if not pend:
            return
        pos = pend.index(index[0]) if index[0] in pend else 0
        pos = max(0, min(len(pend) - 1, pos + delta))
        index[0] = pend[pos]
        refresh()

    def decide(value: str) -> None:
        pend = pending_indices()
        if not pend or index[0] not in pend:
            status.setText("Nothing pending.")
            return
        q = questions[index[0]]
        left = list(q["pair_left_zyx"])
        right = list(q["pair_right_zyx"])
        diffs = [int(r) - int(l) for l, r in zip(left, right)]
        channel = diffs.index(1)
        event = append_external_boundary_event(
            args.workspace,
            reviewer=args.reviewer,
            decision=value,
            pair_left_zyx=left,
            channel_zyx=channel,
            rationale=rationale.text(),
            orthogonal_views_inspected=True,
            question_reference=q["opaque_decision_id"],
            interface_id=queue.get("smoke_batch_id"),
        )
        # Append opaque id into the event log line by rewriting last event is unsafe;
        # instead write a sidecar companion record.
        companion = {
            "opaque_decision_id": q["opaque_decision_id"],
            "event_id": event["id"],
            "decision": value,
            "workspace_id": workspace["id"],
            "crop_id": workspace["crop_id"],
        }
        companion_path = log_path.with_suffix(".opaque.jsonl")
        with companion_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(companion, sort_keys=True) + "\n")
        answered.add(q["opaque_decision_id"])
        rationale.setText("")
        status.setText(f"Recorded {value} for {q['opaque_decision_id']} ({event['id']}).")
        pend = pending_indices()
        if pend:
            index[0] = pend[0]
            refresh()
        else:
            status.setText("All decisions for this workspace are recorded. You may close the window.")

    prev_b.clicked.connect(lambda: move(-1))
    next_b.clicked.connect(lambda: move(1))
    same.clicked.connect(lambda: decide("SAME_PROCESS"))
    different.clicked.connect(lambda: decide("DIFFERENT_PROCESS"))
    uncertain.clicked.connect(lambda: decide("UNCERTAIN"))
    bad.clicked.connect(lambda: decide("BAD_QUESTION"))

    if pending_indices():
        index[0] = pending_indices()[0]
        refresh()
    else:
        status.setText("All decisions already recorded.")

    viewer.window.add_dock_widget(panel, area="right", name="Equivariant edge decision")
    napari.run()


if __name__ == "__main__":
    main()
