"""Open one immutable DVID crop in Napari for local instance-label review.

This is deliberately a reviewer tool, not an automatic annotation generator.
Raw EM and optional machine aids are read-only.  The editable labels layer
starts empty; reviewers must save it explicitly as labels_zyx.npy and register
it with `annotation-crop-register` after inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _render_plane(raw: np.ndarray, labels: np.ndarray, first: int, second: int) -> np.ndarray:
    """Return a compact raw-EM image with segment A/B emphasized."""
    low, high = float(raw.min()), float(raw.max())
    base = np.zeros(raw.shape + (3,), dtype=np.uint8) if high <= low else np.clip((raw.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)[..., None].repeat(3, axis=-1)
    a = labels == first; b = labels == second
    base[a] = (255, 70, 70); base[b] = (60, 225, 120)
    # A/B adjacency is emphasized without claiming that a visible contact is a
    # biological continuity or separation decision.
    contact = np.zeros(labels.shape, dtype=bool)
    for axis in range(2):
        left = [slice(None)] * 2; right = [slice(None)] * 2; left[axis] = slice(0, -1); right[axis] = slice(1, None)
        contact[tuple(left)] |= (a[tuple(left)] & b[tuple(right)]) | (b[tuple(left)] & a[tuple(right)])
        contact[tuple(right)] |= (a[tuple(left)] & b[tuple(right)]) | (b[tuple(left)] & a[tuple(right)])
    base[contact] = (255, 235, 40)
    return base


def _nearest_contact(labels: np.ndarray, first: int, second: int, preferred: list[int]) -> list[int]:
    """Find an A/B face contact nearest the ranked local candidate point."""
    candidates: list[np.ndarray] = []
    for axis in range(3):
        left = [slice(None)] * 3; right = [slice(None)] * 3; left[axis] = slice(0, -1); right[axis] = slice(1, None)
        a, b = labels[tuple(left)], labels[tuple(right)]
        mask = (a == first) & (b == second) | (a == second) & (b == first)
        points = np.argwhere(mask)
        if len(points): candidates.append(points)
    if not candidates: return preferred
    joined = np.concatenate(candidates)
    return [int(value) for value in joined[np.argmin(((joined - np.asarray(preferred)) ** 2).sum(axis=1))]]


def _raw_pair_plane(raw: np.ndarray, coordinates: list[tuple[int, int]], plane: str, fixed: int) -> np.ndarray:
    """Render a raw orthogonal plane with A/B/center marks, no segmentation."""
    low, high = float(raw.min()), float(raw.max())
    image = np.zeros(raw.shape + (3,), dtype=np.uint8) if high <= low else np.clip((raw.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)[..., None].repeat(3, axis=-1)
    colors = ((255, 55, 55), (55, 230, 110), (255, 225, 35))
    for (z, y, x), color in zip(coordinates, colors, strict=True):
        if (plane == "XY" and z != fixed) or (plane == "XZ" and y != fixed) or (plane == "YZ" and x != fixed):
            continue
        row, column = (y, x) if plane == "XY" else (z, x) if plane == "XZ" else (z, y)
        row0, row1 = max(0, row - 2), min(image.shape[0], row + 3)
        col0, col1 = max(0, column - 2), min(image.shape[1], column + 3)
        image[row0:row1, col0:col1] = color
    return image


def _run_membrane_reviewer(*, napari, questions_path: Path, workspace_path: Path, reviewer: str) -> None:
    """Review frozen raw-EM A/B crossings without a machine label layer."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mvconnectome.membrane_pilot import append_membrane_review_event
    from qtpy.QtGui import QImage, QPixmap
    from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

    workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
    queue = json.loads(questions_path.read_text(encoding="utf-8"))
    if workspace["questions"]["sha256"] != __import__("hashlib").sha256(questions_path.read_bytes()).hexdigest():
        raise SystemExit("Frozen membrane question hash does not match workspace")
    raw = np.load(workspace["raw"]["path"], mmap_mode="r", allow_pickle=False)
    questions = queue["questions"]
    viewer = napari.Viewer(title=f"{workspace['crop_id']} — raw-EM membrane crossing review")
    viewer.add_image(raw, name="raw_EM_authoritative", colormap="gray", contrast_limits=(float(raw.min()), float(raw.max())))
    a_layer = viewer.add_points(np.empty((0, 3)), name="A — red", size=8, face_color="red")
    b_layer = viewer.add_points(np.empty((0, 3)), name="B — green", size=8, face_color="lime")
    center_layer = viewer.add_points(np.empty((0, 3)), name="candidate membrane center — yellow", size=6, face_color="yellow")

    panel = QWidget(); layout = QVBoxLayout(panel)
    title = QLabel(); geometry = QLabel(); evidence = QLabel(); status = QLabel("No decision recorded yet.")
    for widget in (title, geometry, evidence, status): widget.setWordWrap(True); layout.addWidget(widget)
    views = QGridLayout(); xy = QLabel("XY"); xz = QLabel("XZ"); yz = QLabel("YZ")
    for label, name, row, column in ((xy, "XY", 0, 0), (xz, "XZ", 0, 1), (yz, "YZ", 1, 0)):
        label.setMinimumSize(200, 140); label.setScaledContents(True); label.setToolTip(f"{name}: raw EM; A red, B green, membrane center yellow")
        views.addWidget(label, row, column)
    views.addWidget(QLabel("Raw EM is authoritative.\nA = red\nB = green\nCandidate center = yellow\nNo machine segmentation is loaded."), 1, 1)
    layout.addLayout(views)
    controls = QHBoxLayout(); previous = QPushButton("Previous"); focus = QPushButton("Focus boundary"); next_button = QPushButton("Next")
    for button in (previous, focus, next_button): controls.addWidget(button)
    layout.addLayout(controls)
    choices = QHBoxLayout(); same = QPushButton("SAME"); different = QPushButton("DIFFERENT"); uncertain = QPushButton("UNCERTAIN"); bad = QPushButton("BAD QUESTION")
    for button in (same, different, uncertain, bad): choices.addWidget(button)
    layout.addLayout(choices)
    index = [0]

    def show_plane(label: QLabel, image: np.ndarray) -> None:
        image = np.ascontiguousarray(image)
        rendered = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(rendered.copy()))

    def show_question() -> None:
        item = questions[index[0]]; a, b, center = item["a_zyx"], item["b_zyx"], item["center_zyx"]
        displayed_center = [float(a[i] + b[i]) / 2.0 for i in range(3)]
        a_layer.data = np.asarray([a], dtype=float); b_layer.data = np.asarray([b], dtype=float); center_layer.data = np.asarray([displayed_center], dtype=float)
        viewer.dims.set_point((0, 1, 2), center)
        title.setText(f"Question {index[0] + 1} / {len(questions)} — {item['id']}")
        geometry.setText(f"A Z/Y/X: {a}  |  B Z/Y/X: {b}\nCenter: {center}; normal Z/Y/X: {item['normal_zyx']}; affinity channel Z/Y/X: {item['affinity_channel_zyx']}")
        orientation = item.get("orientation_confidence")
        orientation_text = "not applicable (continuity-control proposal)" if orientation is None else f"{float(orientation):.3f}"
        evidence.setText(
            f"Raw 3-D candidate score: {item['raw_feature_score']:.3f}; orientation confidence: {orientation_text}\n"
            "Question: Does raw EM support SAME continuity or a DIFFERENT process boundary?"
        )
        coords = [tuple(a), tuple(b), tuple(center)]
        show_plane(xy, _raw_pair_plane(np.asarray(raw[center[0]]), coords, "XY", center[0]))
        show_plane(xz, _raw_pair_plane(np.asarray(raw[:, center[1], :]), coords, "XZ", center[1]))
        show_plane(yz, _raw_pair_plane(np.asarray(raw[:, :, center[2]]), coords, "YZ", center[2]))
        previous.setEnabled(index[0] > 0); next_button.setEnabled(index[0] < len(questions) - 1)

    def move(delta: int) -> None:
        index[0] = max(0, min(len(questions) - 1, index[0] + delta)); show_question()

    def decide(decision: str) -> None:
        event = append_membrane_review_event(workspace_path, reviewer=reviewer, question_id=questions[index[0]]["id"], decision=decision)
        status.setText(f"Recorded {decision} as {event['id']} (append-only; no biological promotion).")
        if index[0] < len(questions) - 1: move(1)

    previous.clicked.connect(lambda: move(-1)); next_button.clicked.connect(lambda: move(1))
    focus.clicked.connect(lambda: (show_question(), setattr(viewer.camera, "zoom", max(float(viewer.camera.zoom), 9.0))))
    same.clicked.connect(lambda: decide("SAME_PROCESS")); different.clicked.connect(lambda: decide("DIFFERENT_PROCESS"))
    uncertain.clicked.connect(lambda: decide("UNCERTAIN")); bad.clicked.connect(lambda: decide("BAD_QUESTION"))
    viewer.window.add_dock_widget(panel, area="right", name="Raw-EM membrane decision")
    show_question()
    print("Raw-EM membrane pilot: decisions are append-only local affinity evidence; they do not create biological records.")
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("crop_id")
    parser.add_argument("--manifest", type=Path, default=Path("ground_truth/annotation_crops_manifest.json"))
    parser.add_argument("--machine-affinity", type=Path, help="Optional aligned ZYX affinity/boundary image; never treated as a label")
    parser.add_argument("--machine-supervoxels", type=Path, help="Optional aligned machine-only supervoxels, shown read-only")
    parser.add_argument("--queue", type=Path, help="Optional active-learning queue JSON for boundary questions")
    parser.add_argument("--workspace", type=Path, help="Machine-only workspace required to persist SAME/DIFFERENT/UNCERTAIN decisions")
    parser.add_argument("--membrane-questions", type=Path, help="Frozen raw-EM membrane-crossing pilot questions")
    parser.add_argument("--membrane-workspace", type=Path, help="Append-only raw-EM membrane-pilot workspace")
    parser.add_argument("--reviewer", help="Recorded reviewer identity required with --workspace")
    args = parser.parse_args()
    membrane_mode = bool(args.membrane_questions or args.membrane_workspace)
    if membrane_mode:
        if not (args.membrane_questions and args.membrane_workspace and args.reviewer):
            parser.error("raw-EM membrane review requires --membrane-questions, --membrane-workspace, and --reviewer")
        if any((args.machine_affinity, args.machine_supervoxels, args.queue, args.workspace)):
            parser.error("raw-EM membrane review is raw-first and cannot mix the supervoxel review arguments")
        try:
            import napari
        except ImportError as error:
            raise SystemExit("Napari is required for interactive review; install it in a separate reviewer environment.") from error
        _run_membrane_reviewer(napari=napari, questions_path=args.membrane_questions, workspace_path=args.membrane_workspace, reviewer=args.reviewer)
        return
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    crop = next((item for item in manifest["crops"] if item["id"] == args.crop_id), None)
    if crop is None:
        parser.error(f"Unknown crop {args.crop_id}")
    if crop["annotation_status"] != "UNANNOTATED":
        parser.error(f"{args.crop_id} is not an unannotated review workspace")
    if bool(args.workspace) != bool(args.reviewer):
        parser.error("--workspace and --reviewer must be supplied together")
    if args.workspace and not args.queue:
        parser.error("--workspace requires --queue so every decision retains its question context")
    try:
        import napari
    except ImportError as error:
        raise SystemExit("Napari is required for interactive review; install it in a separate reviewer environment.") from error
    raw = np.load(crop["raw_crop_path"], mmap_mode="r", allow_pickle=False)
    viewer = napari.Viewer(title=f"{args.crop_id} — DVID review; raw/source coordinates are immutable")
    viewer.add_image(raw, name="raw_EM_read_only", colormap="gray", contrast_limits=(float(raw.min()), float(raw.max())))
    if args.machine_affinity:
        aid = np.load(args.machine_affinity, mmap_mode="r", allow_pickle=False)
        if aid.shape == raw.shape:
            viewer.add_image(aid, name="machine_aid_not_ground_truth", colormap="magenta", blending="additive", opacity=0.45)
        elif aid.ndim == 4 and aid.shape[0] == 3 and tuple(aid.shape[1:]) == tuple(raw.shape):
            # SegNeuron's verified contract is CZYX with channels Z/Y/X. Do
            # not silently collapse it: expose every directional aid plus an
            # explicitly derived mean for convenient review.
            for name, channel in zip(("Z", "Y", "X"), aid, strict=True):
                viewer.add_image(channel, name=f"machine_affinity_{name}_not_ground_truth", colormap="magenta", blending="additive", opacity=0.30, visible=False)
            viewer.add_image(aid.mean(axis=0), name="machine_affinity_mean_not_ground_truth", colormap="magenta", blending="additive", opacity=0.45)
        else:
            parser.error("Machine aid must be aligned ZYX, or a three-channel CZYX affinity tensor")
    if args.machine_supervoxels:
        segments = np.load(args.machine_supervoxels, mmap_mode="r", allow_pickle=False)
        if segments.shape != raw.shape or segments.ndim != 3:
            parser.error("Machine supervoxels must be an aligned ZYX label volume")
        machine_layer = viewer.add_labels(segments, name="machine_supervoxels_read_only", opacity=0.45)
        machine_layer.editable = False
    if args.queue:
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
        points = np.asarray([item["local_point_zyx"] for item in queue.get("candidates", [])], dtype=float)
        if len(points):
            viewer.add_points(points, name="ranked_boundary_questions_not_decisions", size=4, face_color="yellow")
        if args.workspace:
            # Keep the scientific core importable from a source checkout while
            # Napari itself remains isolated in .venv-reviewer.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from mvconnectome.proofreading import append_review_event
            from qtpy.QtGui import QImage, QPixmap
            from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

            candidates = queue.get("candidates", [])
            panel = QWidget(); layout = QVBoxLayout(panel)
            title = QLabel(); segments_text = QLabel(); location = QLabel(); detail = QLabel(); status = QLabel("No decision recorded yet.")
            for widget in (title, segments_text, location, detail, status):
                widget.setWordWrap(True); layout.addWidget(widget)
            views = QGridLayout(); xy = QLabel("XY"); xz = QLabel("XZ"); yz = QLabel("YZ")
            for item, name, row, column in ((xy, "XY", 0, 0), (xz, "XZ", 0, 1), (yz, "YZ", 1, 0)):
                item.setMinimumSize(180, 130); item.setScaledContents(True); item.setToolTip(f"{name}: raw EM with A red, B green, shared face yellow")
                views.addWidget(item, row, column)
            legend = QLabel("A = red\nB = green\nA/B face = yellow\nRaw EM = grayscale")
            views.addWidget(legend, 1, 1); layout.addLayout(views)
            z_controls = QHBoxLayout(); z_label = QLabel("Z neighborhood")
            z_controls.addWidget(z_label); z_buttons: list[QPushButton] = []
            for delta in range(-5, 6):
                button = QPushButton("[0]" if delta == 0 else f"{delta:+d}"); button.setMaximumWidth(34)
                z_controls.addWidget(button); z_buttons.append(button)
            layout.addLayout(z_controls)
            controls = QHBoxLayout(); previous = QPushButton("Previous"); next_button = QPushButton("Next")
            focus = QPushButton("Focus boundary"); controls.addWidget(previous); controls.addWidget(focus); controls.addWidget(next_button); layout.addLayout(controls)
            decisions = QHBoxLayout(); same = QPushButton("SAME"); different = QPushButton("DIFFERENT"); uncertain = QPushButton("UNCERTAIN"); bad = QPushButton("BAD QUESTION")
            for button in (same, different, uncertain, bad): decisions.addWidget(button)
            layout.addLayout(decisions)
            index = [0]

            z_offset = [0]

            def show_plane(label: QLabel, image: np.ndarray) -> None:
                image = np.ascontiguousarray(image)
                qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
                label.setPixmap(QPixmap.fromImage(qimage.copy()))

            def show_question() -> None:
                if not candidates:
                    title.setText("No ranked boundary questions are available."); return
                item = candidates[index[0]]; point = _nearest_contact(segments, item["supervoxel_a"], item["supervoxel_b"], item["local_point_zyx"])
                point[0] = max(0, min(raw.shape[0] - 1, point[0] + z_offset[0]))
                title.setText(f"Question {index[0] + 1} / {len(candidates)}")
                segments_text.setText(f"Segments: {item['supervoxel_a']} ↔ {item['supervoxel_b']}")
                location.setText(f"Local Z/Y/X: {point[0]}, {point[1]}, {point[2]}")
                affinity = "not used for selection" if item.get("affinity_mean") is None else f"{item['affinity_mean']:.3f}"
                detail.setText(f"Interface voxels: {item['interface_voxel_count']} | model affinity: {affinity} | raw boundary: {item.get('raw_boundary_mean', 0.0):.3f}\n{item['question']}")
                viewer.dims.set_point((0, 1, 2), point)
                show_plane(xy, _render_plane(np.asarray(raw[point[0]]), np.asarray(segments[point[0]]), item["supervoxel_a"], item["supervoxel_b"]))
                show_plane(xz, _render_plane(np.asarray(raw[:, point[1], :]), np.asarray(segments[:, point[1], :]), item["supervoxel_a"], item["supervoxel_b"]))
                show_plane(yz, _render_plane(np.asarray(raw[:, :, point[2]]), np.asarray(segments[:, :, point[2]]), item["supervoxel_a"], item["supervoxel_b"]))
                z_label.setText(f"Z neighborhood: center {point[0]} ({z_offset[0]:+d})")
                for delta, button in zip(range(-5, 6), z_buttons, strict=True): button.setEnabled(0 <= point[0] + delta - z_offset[0] < raw.shape[0])
                previous.setEnabled(index[0] > 0); next_button.setEnabled(index[0] < len(candidates) - 1)

            def move(delta: int) -> None:
                index[0] = max(0, min(len(candidates) - 1, index[0] + delta)); z_offset[0] = 0; show_question()

            def focus_boundary() -> None:
                z_offset[0] = 0; show_question()
                viewer.camera.zoom = max(float(viewer.camera.zoom), 8.0)

            def decide(kind: str) -> None:
                item = candidates[index[0]]
                append_review_event(args.workspace, event={"kind": kind, "reviewer": args.reviewer, "candidate": item})
                status.setText(f"Recorded {kind} for {item['id']} (append-only; no biological promotion).")
                if index[0] < len(candidates) - 1:
                    move(1)

            previous.clicked.connect(lambda: move(-1)); next_button.clicked.connect(lambda: move(1))
            same.clicked.connect(lambda: decide("SAME_PROCESS")); different.clicked.connect(lambda: decide("DIFFERENT_PROCESS")); uncertain.clicked.connect(lambda: decide("UNCERTAIN")); bad.clicked.connect(lambda: decide("BAD_QUESTION")); focus.clicked.connect(focus_boundary)
            for delta, button in zip(range(-5, 6), z_buttons, strict=True): button.clicked.connect(lambda checked=False, value=delta: (z_offset.__setitem__(0, value), show_question()))
            viewer.window.add_dock_widget(panel, area="right", name="Proofreading decision")
            show_question()
    labels = viewer.add_labels(np.zeros(raw.shape, dtype=np.uint32), name="reviewed_instances_editable")
    labels.metadata.update({"required_export": "labels_zyx.npy", "ignore_label": 4294967295,
                            "source_crop_id": crop["id"], "source_raw_sha256": crop["raw_crop_sha256"],
                            "coordinate_origin_xyz": crop["source_origin_xyz"], "review_state": "DRAFT"})
    print("Review raw EM in XY/XZ/YZ. Machine supervoxels and ranked points are aids, not labels. The decision panel appends reviewer decisions without biological promotion.")
    napari.run()


if __name__ == "__main__":
    main()
