"""Local raw-EM interface for an independent expert boundary review.

This tool intentionally shows no model segmentation.  The expert selects a
single adjacent Z/Y/X voxel pair after inspecting raw EM in XY, XZ, and YZ.
Decisions append immutable local training evidence; they do not create a
neuron, segment, synapse, or connection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _plane(raw: np.ndarray, a: tuple[int, int, int], b: tuple[int, int, int], name: str, fixed: int) -> np.ndarray:
    low, high = float(raw.min()), float(raw.max())
    rgb = np.zeros(raw.shape + (3,), dtype=np.uint8) if high <= low else np.clip((raw.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)[..., None].repeat(3, axis=-1)
    for point, color in ((a, (255, 55, 55)), (b, (60, 230, 110))):
        z, y, x = point
        if (name == "XY" and z != fixed) or (name == "XZ" and y != fixed) or (name == "YZ" and x != fixed):
            continue
        row, column = (y, x) if name == "XY" else (z, x) if name == "XZ" else (z, y)
        rgb[max(0, row - 2):row + 3, max(0, column - 2):column + 3] = color
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Review independent raw-EM DVID boundary pairs")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--scan-queue", type=Path, help="Frozen raw-EM navigation-only queue; never a membrane or model-candidate queue")
    parser.add_argument("--question-queue", type=Path, help="Frozen G1/raw diagnostic questions for this workspace; model output remains only a navigation aid")
    parser.add_argument("--interface-queue", type=Path, help="Frozen raw-EM-only grouped interface queue for G3; every member is reviewed independently")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mvconnectome.external_boundary_review import append_external_boundary_event
    try:
        import napari
        from qtpy.QtGui import QImage, QPixmap
        from qtpy.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget
    except ImportError as error:
        raise SystemExit("Napari and its Qt backend must be installed in the reviewer environment.") from error

    workspace = json.loads(args.workspace.read_text(encoding="utf-8"))
    if workspace.get("status") != "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED":
        raise SystemExit("Workspace is not an external raw-EM review workspace")
    raw = np.load(workspace["raw"]["path"], mmap_mode="r", allow_pickle=False)
    if sum(bool(value) for value in (args.scan_queue, args.question_queue, args.interface_queue)) > 1:
        raise SystemExit("Choose one frozen queue type")
    scan_locations: list[list[int]] = []
    diagnostic_questions: list[dict] = []
    if args.scan_queue:
        scan = json.loads(args.scan_queue.read_text(encoding="utf-8"))
        if scan.get("workspace_id") != workspace["id"] or scan.get("raw_sha256") != workspace["raw"]["sha256"]:
            raise SystemExit("Frozen scan queue does not match this immutable external-review workspace")
        if scan.get("status") != "RAW_EM_NAVIGATION_ONLY_REVIEW_REQUIRED":
            raise SystemExit("Scan queue is not a raw-EM navigation-only queue")
        scan_locations = scan.get("locations_zyx", [])
    if args.question_queue:
        question_queue = json.loads(args.question_queue.read_text(encoding="utf-8"))
        if question_queue.get("region", {}).get("id") != workspace["crop_id"] or question_queue.get("region", {}).get("raw_sha256") != workspace["raw"]["sha256"]:
            raise SystemExit("Frozen diagnostic question queue does not match this immutable external-review workspace")
        if question_queue.get("status") != "EXPERT_REVIEW_REQUIRED":
            raise SystemExit("Question queue is not eligible for expert review")
        diagnostic_questions = question_queue.get("questions", [])
        if not diagnostic_questions:
            raise SystemExit("Question queue is empty; do not substitute arbitrary pairs")
    if args.interface_queue:
        interface_queue = json.loads(args.interface_queue.read_text(encoding="utf-8"))
        if interface_queue.get("workspace_id") != workspace["id"] or interface_queue.get("raw_sha256") != workspace["raw"]["sha256"]:
            raise SystemExit("Frozen interface queue does not match this immutable external-review workspace")
        if interface_queue.get("status") != "EXPERT_INTERFACE_REVIEW_REQUIRED":
            raise SystemExit("Interface queue is not eligible for expert review")
        diagnostic_questions = interface_queue.get("questions", [])
        if not diagnostic_questions:
            raise SystemExit("Interface queue is empty")
    viewer = napari.Viewer(title=f"{workspace['crop_id']} — independent raw-EM boundary review")
    viewer.add_image(raw, name="raw_EM_authoritative", colormap="gray", contrast_limits=(float(raw.min()), float(raw.max())))
    a_layer = viewer.add_points(np.empty((0, 3)), name="A (red)", size=8, face_color="red")
    b_layer = viewer.add_points(np.empty((0, 3)), name="B (green)", size=8, face_color="lime")

    panel = QWidget(); layout = QVBoxLayout(panel)
    title = QLabel(f"{workspace['split']} — {workspace['crop_id']}")
    help_text = QLabel("Inspect raw EM in XY/XZ/YZ and neighbouring slices, then decide whether A/B are SAME or DIFFERENT neuronal process. UNCERTAIN and BAD QUESTION are valid. G1 information prioritizes inspection only.")
    title.setWordWrap(True); help_text.setWordWrap(True); layout.addWidget(title); layout.addWidget(help_text)
    controls = QGridLayout(); spinners = []
    for axis, (label, limit) in enumerate(zip(("Z", "Y", "X"), raw.shape, strict=True)):
        controls.addWidget(QLabel(f"A {label}"), 0, axis)
        spin = QSpinBox(); spin.setRange(0, limit - 1); spin.setValue(limit // 2); controls.addWidget(spin, 1, axis); spinners.append(spin)
    channel = QComboBox(); channel.addItems(["Z (0)", "Y (1)", "X (2)"])
    controls.addWidget(QLabel("Affinity direction A → B"), 2, 0, 1, 2); controls.addWidget(channel, 3, 0, 1, 2)
    cursor_button = QPushButton("Set A from current cursor"); controls.addWidget(cursor_button, 3, 2)
    layout.addLayout(controls)
    views = QGridLayout(); xy, xz, yz = QLabel(), QLabel(), QLabel()
    for widget, label, row, column in ((xy, "XY", 0, 0), (xz, "XZ", 0, 1), (yz, "YZ", 1, 0)):
        widget.setMinimumSize(220, 150); widget.setScaledContents(True); widget.setToolTip(f"{label}: raw EM; A red; B green")
        views.addWidget(widget, row, column)
    rationale = QLineEdit(); rationale.setPlaceholderText("Optional rationale; avoid biological identity claims")
    views.addWidget(rationale, 1, 1); layout.addLayout(views)
    status = QLabel("No decision recorded."); status.setWordWrap(True); layout.addWidget(status)
    scan_controls = QHBoxLayout(); scan_label = QLabel("Manual navigation")
    scan_previous, scan_next = QPushButton("Previous location"), QPushButton("Next location")
    if scan_locations or diagnostic_questions:
        scan_controls.addWidget(scan_previous); scan_controls.addWidget(scan_label); scan_controls.addWidget(scan_next); layout.addLayout(scan_controls)
    buttons = QHBoxLayout(); same, different, uncertain, bad = (QPushButton("SAME"), QPushButton("DIFFERENT"), QPushButton("UNCERTAIN"), QPushButton("BAD QUESTION"))
    for button in (same, different, uncertain, bad): buttons.addWidget(button)
    layout.addLayout(buttons)

    question_index = [0]

    def pair() -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        if diagnostic_questions:
            item = diagnostic_questions[question_index[0]]
            return tuple(item["pair_left_zyx"]), tuple(item["pair_right_zyx"]), int(item["channel_zyx"])
        a = tuple(spin.value() for spin in spinners); axis = channel.currentIndex(); b = list(a); b[axis] += 1
        if b[axis] >= raw.shape[axis]:
            raise ValueError("A is on the positive crop border for the selected affinity direction")
        return a, tuple(b), axis

    def show(image_label: QLabel, image: np.ndarray) -> None:
        image = np.ascontiguousarray(image)
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
        image_label.setPixmap(QPixmap.fromImage(qimage.copy()))

    scan_index = [0]

    def refresh() -> None:
        try:
            a, b, axis = pair()
        except ValueError as error:
            status.setText(str(error)); return
        a_layer.data = np.asarray([a], dtype=float); b_layer.data = np.asarray([b], dtype=float)
        viewer.dims.set_point((0, 1, 2), a)
        show(xy, _plane(np.asarray(raw[a[0]]), a, b, "XY", a[0]))
        show(xz, _plane(np.asarray(raw[:, a[1], :]), a, b, "XZ", a[1]))
        show(yz, _plane(np.asarray(raw[:, :, a[2]]), a, b, "YZ", a[2]))
        if diagnostic_questions:
            item = diagnostic_questions[question_index[0]]
            if item.get("model_navigation", True):
                navigation = f" Question {question_index[0] + 1}/{len(diagnostic_questions)}: {item['kind']}; G1 affinity={item['g1_affinity']:.3f}; raw-gradient score={item['raw_gradient_score']:.3f}."
            else:
                navigation = f" Interface {item['interface_id']}, member {item['interface_member']}/{item['interface_members']} — raw-EM-only candidate; gradient score={item['raw_gradient_score']:.3f}."
        else:
            navigation = f" Scan location {scan_index[0] + 1}/{len(scan_locations)} (navigation only)." if scan_locations else ""
        status.setText(f"A Z/Y/X {list(a)} → B Z/Y/X {list(b)}; channel {_channel_name(axis)}. Confirmed orthogonal review is required when saving.{navigation}")

    def decide(value: str) -> None:
        try:
            a, _, axis = pair()
            reference = diagnostic_questions[question_index[0]].get("id") or (diagnostic_questions[question_index[0]].get("kind") + f":{question_index[0] + 1}") if diagnostic_questions else None
            event = append_external_boundary_event(args.workspace, reviewer=args.reviewer, decision=value, pair_left_zyx=list(a), channel_zyx=axis, rationale=rationale.text(), orthogonal_views_inspected=True, question_reference=reference, interface_id=diagnostic_questions[question_index[0]].get("interface_id") if diagnostic_questions else None)
            status.setText(f"Recorded {value}: {event['id']} (append-only local affinity evidence).")
            if diagnostic_questions and question_index[0] < len(diagnostic_questions) - 1:
                question_index[0] += 1; set_diagnostic_question()
            elif scan_locations and scan_index[0] < len(scan_locations) - 1:
                scan_index[0] += 1; set_scan_location()
        except ValueError as error:
            status.setText(f"Not recorded: {error}")

    def set_cursor() -> None:
        position = viewer.cursor.position
        for spin, value in zip(spinners, position, strict=True): spin.setValue(max(spin.minimum(), min(spin.maximum(), int(round(value)))))
        refresh()

    def set_scan_location() -> None:
        point = scan_locations[scan_index[0]]
        for spin, value in zip(spinners, point, strict=True): spin.setValue(value)
        scan_label.setText(f"Location {scan_index[0] + 1} / {len(scan_locations)} — navigation only")
        scan_previous.setEnabled(scan_index[0] > 0); scan_next.setEnabled(scan_index[0] < len(scan_locations) - 1)
        refresh()

    def set_diagnostic_question() -> None:
        item = diagnostic_questions[question_index[0]]
        for spin, value in zip(spinners, item["pair_left_zyx"], strict=True): spin.setValue(value)
        channel.setCurrentIndex(int(item["channel_zyx"]))
        scan_label.setText(f"Question {question_index[0] + 1} / {len(diagnostic_questions)}")
        scan_previous.setEnabled(question_index[0] > 0); scan_next.setEnabled(question_index[0] < len(diagnostic_questions) - 1)
        refresh()

    def move_scan(delta: int) -> None:
        if diagnostic_questions:
            question_index[0] = max(0, min(len(diagnostic_questions) - 1, question_index[0] + delta)); set_diagnostic_question()
        else:
            scan_index[0] = max(0, min(len(scan_locations) - 1, scan_index[0] + delta)); set_scan_location()

    def _channel_name(axis: int) -> str:
        return ("Z", "Y", "X")[axis]

    for spin in spinners: spin.valueChanged.connect(refresh)
    channel.currentIndexChanged.connect(refresh); cursor_button.clicked.connect(set_cursor)
    if diagnostic_questions:
        for control in (*spinners, channel, cursor_button): control.setEnabled(False)
        set_diagnostic_question()
    elif scan_locations:
        scan_previous.clicked.connect(lambda: move_scan(-1)); scan_next.clicked.connect(lambda: move_scan(1))
    same.clicked.connect(lambda: decide("SAME_PROCESS")); different.clicked.connect(lambda: decide("DIFFERENT_PROCESS")); uncertain.clicked.connect(lambda: decide("UNCERTAIN")); bad.clicked.connect(lambda: decide("BAD_QUESTION"))
    viewer.window.add_dock_widget(panel, area="right", name="External boundary decision")
    if scan_locations: set_scan_location()
    else: refresh()
    napari.run()


if __name__ == "__main__":
    main()
