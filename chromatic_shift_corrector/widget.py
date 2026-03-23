"""Main docked widget for the Image Processing Pipeline viewer."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QThread, Signal, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QDoubleSpinBox,
    QHeaderView,
    QAbstractItemView,
)

from chromatic_shift_corrector.data_loader import (
    H5Loader,
    validate_channels,
    z_dimensions_summary,
)
from chromatic_shift_corrector.h5_utils import H5FileManager, scan_h5_files
from chromatic_shift_corrector.preview_engine import extract_subvolume
from chromatic_shift_corrector.gpu_utils import gpu_available, gpu_fail_reason, gpu_name
from chromatic_shift_corrector.processing_worker import (
    ProcessingWorker,
    ExportProcessedWorker,
    FullStackProcessingWorker,
)
from chromatic_shift_corrector.utils import DEFAULT_COLORMAPS, MAX_CHANNELS

COLORMAP_OPTIONS = [
    "green", "magenta", "cyan", "yellow", "red", "blue",
    "gray", "viridis", "inferno", "turbo",
]

FILTER_TYPES = ["None", "Rolling Ball", "Gaussian Blur", "Unsharp Mask"]


class ImageProcessingWidget(QWidget):
    """Docked widget providing an image processing pipeline workflow."""

    def __init__(self, napari_viewer: Any) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.loaders: list[Any] = []
        self._preview_layer_names: list[str] = []
        self._processed_layer_names: list[str] = []
        self._shapes_layer = None
        self._h5_file_manager: H5FileManager | None = None
        self._processing_worker: ProcessingWorker | None = None
        self._export_worker: ExportProcessedWorker | None = None
        self._fullstack_worker: FullStackProcessingWorker | None = None
        self._filenames: list[str] = []
        self._colormaps: list[str] = []
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        outer = QVBoxLayout()
        outer.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._build_data_section())
        layout.addWidget(self._build_pipeline_section())
        layout.addWidget(self._build_preview_section())
        layout.addWidget(self._build_fullstack_section())

        layout.addStretch()
        container.setLayout(layout)
        scroll.setWidget(container)
        outer.addWidget(scroll)
        self.setLayout(outer)

    def _build_data_section(self) -> QGroupBox:
        grp = QGroupBox("Data Loading")
        lay = QVBoxLayout()

        row_dir = QHBoxLayout()
        self.btn_select_dir = QPushButton("Select Directory")
        self.btn_select_dir.clicked.connect(self._on_select_directory)
        self.lbl_dir = QLineEdit()
        self.lbl_dir.setReadOnly(True)
        self.lbl_dir.setPlaceholderText("No directory selected")
        row_dir.addWidget(self.btn_select_dir)
        row_dir.addWidget(self.lbl_dir)
        lay.addLayout(row_dir)

        self.file_table = QTableWidget(0, 3)
        self.file_table.setHorizontalHeaderLabels(["Include", "Filename", "Colormap"])
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.file_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.file_table.verticalHeader().setVisible(False)
        lay.addWidget(self.file_table)

        row_voxel = QHBoxLayout()
        row_voxel.addWidget(QLabel("XY pixel (\u00b5m):"))
        self.spin_voxel_xy = QDoubleSpinBox()
        self.spin_voxel_xy.setDecimals(4)
        self.spin_voxel_xy.setRange(0.0001, 100.0)
        self.spin_voxel_xy.setValue(0.3)
        row_voxel.addWidget(self.spin_voxel_xy)
        row_voxel.addWidget(QLabel("Z step (\u00b5m):"))
        self.spin_voxel_z = QDoubleSpinBox()
        self.spin_voxel_z.setDecimals(4)
        self.spin_voxel_z.setRange(0.0001, 100.0)
        self.spin_voxel_z.setValue(1.0)
        row_voxel.addWidget(self.spin_voxel_z)
        lay.addLayout(row_voxel)

        self.lbl_voxel_source = QLabel("")
        self.lbl_voxel_source.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_voxel_source.setVisible(False)
        lay.addWidget(self.lbl_voxel_source)

        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self._on_load_data)
        lay.addWidget(self.btn_load)

        grp.setLayout(lay)
        return grp

    def _build_pipeline_section(self) -> QGroupBox:
        grp = QGroupBox("Processing Pipeline")
        lay = QVBoxLayout()

        # GPU status
        self.lbl_gpu_status = QLabel()
        self._update_gpu_indicator()
        lay.addWidget(self.lbl_gpu_status)

        self.chk_use_gpu = QCheckBox("Use GPU")
        self.chk_use_gpu.setEnabled(gpu_available())
        self.chk_use_gpu.setChecked(gpu_available())
        lay.addWidget(self.chk_use_gpu)

        # Channel selector
        row_ch = QHBoxLayout()
        row_ch.addWidget(QLabel("Channel:"))
        self.combo_channel = QComboBox()
        self.combo_channel.addItem("All channels")
        row_ch.addWidget(self.combo_channel)
        lay.addLayout(row_ch)

        # Pipeline slots
        self._pipeline_slots: list[dict] = []
        for slot_idx in range(3):
            slot = self._build_pipeline_slot(slot_idx)
            self._pipeline_slots.append(slot)
            lay.addWidget(QLabel(f"Step {slot_idx + 1}:"))
            lay.addWidget(slot["combo"])
            lay.addWidget(slot["params_widget"])

        grp.setLayout(lay)
        return grp

    def _build_pipeline_slot(self, idx: int) -> dict:
        combo = QComboBox()
        combo.addItems(FILTER_TYPES)

        params_widget = QWidget()
        params_lay = QVBoxLayout()
        params_lay.setContentsMargins(20, 0, 0, 0)

        # Rolling ball params
        rb_widget = QWidget()
        rb_lay = QHBoxLayout()
        rb_lay.setContentsMargins(0, 0, 0, 0)
        rb_lay.addWidget(QLabel("Radius:"))
        spin_radius = QSpinBox()
        spin_radius.setRange(1, 500)
        spin_radius.setValue(50)
        rb_lay.addWidget(spin_radius)
        rb_widget.setLayout(rb_lay)
        rb_widget.setVisible(False)
        params_lay.addWidget(rb_widget)

        # Gaussian params
        gauss_widget = QWidget()
        gauss_lay = QHBoxLayout()
        gauss_lay.setContentsMargins(0, 0, 0, 0)
        gauss_lay.addWidget(QLabel("Sigma:"))
        spin_sigma_g = QDoubleSpinBox()
        spin_sigma_g.setRange(0.1, 50.0)
        spin_sigma_g.setValue(1.0)
        spin_sigma_g.setSingleStep(0.1)
        gauss_lay.addWidget(spin_sigma_g)
        gauss_widget.setLayout(gauss_lay)
        gauss_widget.setVisible(False)
        params_lay.addWidget(gauss_widget)

        # Unsharp mask params
        um_widget = QWidget()
        um_lay = QHBoxLayout()
        um_lay.setContentsMargins(0, 0, 0, 0)
        um_lay.addWidget(QLabel("Sigma:"))
        spin_sigma_u = QDoubleSpinBox()
        spin_sigma_u.setRange(0.1, 50.0)
        spin_sigma_u.setValue(1.0)
        spin_sigma_u.setSingleStep(0.1)
        um_lay.addWidget(spin_sigma_u)
        um_lay.addWidget(QLabel("Weight:"))
        spin_weight = QDoubleSpinBox()
        spin_weight.setRange(0.0, 1.0)
        spin_weight.setValue(0.6)
        spin_weight.setSingleStep(0.05)
        um_lay.addWidget(spin_weight)
        um_widget.setLayout(um_lay)
        um_widget.setVisible(False)
        params_lay.addWidget(um_widget)

        params_widget.setLayout(params_lay)

        slot = {
            "combo": combo,
            "params_widget": params_widget,
            "rb_widget": rb_widget,
            "gauss_widget": gauss_widget,
            "um_widget": um_widget,
            "spin_radius": spin_radius,
            "spin_sigma_g": spin_sigma_g,
            "spin_sigma_u": spin_sigma_u,
            "spin_weight": spin_weight,
        }

        combo.currentIndexChanged.connect(lambda _idx, s=slot: self._on_filter_changed(s))
        return slot

    def _on_filter_changed(self, slot: dict) -> None:
        text = slot["combo"].currentText()
        slot["rb_widget"].setVisible(text == "Rolling Ball")
        slot["gauss_widget"].setVisible(text == "Gaussian Blur")
        slot["um_widget"].setVisible(text == "Unsharp Mask")

    def _build_preview_section(self) -> QGroupBox:
        grp = QGroupBox("ROI Preview & Processing")
        lay = QVBoxLayout()

        self.btn_draw_roi = QPushButton("Draw ROI Rectangle")
        self.btn_draw_roi.clicked.connect(self._on_draw_roi)
        lay.addWidget(self.btn_draw_roi)

        row_z = QHBoxLayout()
        row_z.addWidget(QLabel("Z start:"))
        self.spin_z_start = QSpinBox()
        self.spin_z_start.setRange(0, 0)
        row_z.addWidget(self.spin_z_start)
        row_z.addWidget(QLabel("Z end:"))
        self.spin_z_end = QSpinBox()
        self.spin_z_end.setRange(0, 0)
        row_z.addWidget(self.spin_z_end)
        lay.addLayout(row_z)

        row_preview = QHBoxLayout()
        self.btn_load_preview = QPushButton("Load Preview")
        self.btn_load_preview.clicked.connect(self._on_load_preview)
        self.btn_clear_preview = QPushButton("Clear Preview")
        self.btn_clear_preview.clicked.connect(self._on_clear_preview)
        row_preview.addWidget(self.btn_load_preview)
        row_preview.addWidget(self.btn_clear_preview)
        lay.addLayout(row_preview)

        row_process = QHBoxLayout()
        self.btn_process_roi = QPushButton("Process ROI")
        self.btn_process_roi.clicked.connect(self._on_process_roi)
        self.btn_export_roi = QPushButton("Export Processed ROI")
        self.btn_export_roi.clicked.connect(self._on_export_processed_roi)
        row_process.addWidget(self.btn_process_roi)
        row_process.addWidget(self.btn_export_roi)
        lay.addLayout(row_process)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("")
        self.lbl_progress.setVisible(False)
        lay.addWidget(self.lbl_progress)

        grp.setLayout(lay)
        return grp

    def _build_fullstack_section(self) -> QGroupBox:
        grp = QGroupBox("Full Stack Processing")
        lay = QVBoxLayout()

        self.btn_process_full = QPushButton("Process Full Stack")
        self.btn_process_full.clicked.connect(self._on_process_full_stack)
        lay.addWidget(self.btn_process_full)

        self.fullstack_progress = QProgressBar()
        self.fullstack_progress.setVisible(False)
        lay.addWidget(self.fullstack_progress)

        self.lbl_fullstack_progress = QLabel("")
        self.lbl_fullstack_progress.setVisible(False)
        lay.addWidget(self.lbl_fullstack_progress)

        grp.setLayout(lay)
        return grp

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _update_gpu_indicator(self) -> None:
        if gpu_available():
            name = gpu_name()
            self.lbl_gpu_status.setText(f"\u25cf GPU: {name}")
            self.lbl_gpu_status.setStyleSheet("color: #4a90d9; font-weight: bold;")
        else:
            reason = gpu_fail_reason()
            self.lbl_gpu_status.setText("\u25cf CPU mode")
            self.lbl_gpu_status.setStyleSheet("color: #999; font-weight: bold;")
            self.lbl_gpu_status.setToolTip(reason or "No compatible GPU detected.")

    def _collect_pipeline_steps(self) -> list[dict]:
        steps = []
        for slot in self._pipeline_slots:
            text = slot["combo"].currentText()
            if text == "None":
                continue
            if text == "Rolling Ball":
                steps.append({
                    "type": "rolling_ball",
                    "enabled": True,
                    "params": {"radius": slot["spin_radius"].value()},
                })
            elif text == "Gaussian Blur":
                steps.append({
                    "type": "gaussian",
                    "enabled": True,
                    "params": {"sigma": slot["spin_sigma_g"].value()},
                })
            elif text == "Unsharp Mask":
                steps.append({
                    "type": "unsharp_mask",
                    "enabled": True,
                    "params": {
                        "sigma": slot["spin_sigma_u"].value(),
                        "mask_weight": slot["spin_weight"].value(),
                    },
                })
        return steps

    def _has_roi(self) -> bool:
        return (
            self._shapes_layer is not None
            and self._shapes_layer in self.viewer.layers
            and len(self._shapes_layer.data) > 0
        )

    def _get_roi_bounds(self) -> tuple[int, int, int, int] | None:
        if self._shapes_layer is None or len(self._shapes_layer.data) == 0:
            return None
        rect = self._shapes_layer.data[-1]
        ys = rect[:, 0]
        xs = rect[:, 1]
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    def _get_selected_channel_indices(self) -> list[int]:
        idx = self.combo_channel.currentIndex()
        if idx == 0:  # "All channels"
            return list(range(len(self.loaders)))
        return [idx - 1]  # offset by 1 for "All channels" entry

    # ------------------------------------------------------------------ #
    # Data Loading
    # ------------------------------------------------------------------ #

    def _on_select_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if not path:
            return
        self.lbl_dir.setText(path)
        dir_path = Path(path)
        files = scan_h5_files(dir_path)
        self._populate_file_table(files)
        self.lbl_voxel_source.setVisible(False)
        if files:
            self._try_autofill_h5_voxel(files[0])

    def _try_autofill_h5_voxel(self, h5_path: Path) -> None:
        try:
            from chromatic_shift_corrector.h5_utils import parse_h5_metadata
            import h5py
            with h5py.File(str(h5_path), "r") as f:
                meta = parse_h5_metadata(f)
            xy = meta.get("voxel_size_xy_um")
            z = meta.get("voxel_size_z_um")
            if xy and xy > 0:
                self.spin_voxel_xy.setValue(xy)
            if z and z > 0:
                self.spin_voxel_z.setValue(z)
            if xy or z:
                self.lbl_voxel_source.setText("from H5 metadata")
                self.lbl_voxel_source.setVisible(True)
        except Exception:
            pass

    def _populate_file_table(self, files: list[Path]) -> None:
        n = min(len(files), MAX_CHANNELS)
        self.file_table.setRowCount(n)
        for i in range(n):
            chk = QCheckBox()
            chk.setChecked(True)
            self.file_table.setCellWidget(i, 0, chk)

            item = QTableWidgetItem(files[i].name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, str(files[i]))
            self.file_table.setItem(i, 1, item)

            combo = QComboBox()
            combo.addItems(COLORMAP_OPTIONS)
            default_idx = i if i < len(DEFAULT_COLORMAPS) else 0
            try:
                combo.setCurrentIndex(COLORMAP_OPTIONS.index(DEFAULT_COLORMAPS[default_idx]))
            except (ValueError, IndexError):
                combo.setCurrentIndex(0)
            self.file_table.setCellWidget(i, 2, combo)

    def _on_load_data(self) -> None:
        self._close_loaders()
        self._on_clear_preview()
        self._clear_processed_layers()

        layers_to_remove = [l for l in self.viewer.layers if l.name.startswith("ch")]
        for l in layers_to_remove:
            self.viewer.layers.remove(l)

        rows = self.file_table.rowCount()
        selected: list[tuple[int, Path, str]] = []
        for i in range(rows):
            chk: QCheckBox = self.file_table.cellWidget(i, 0)
            if not chk.isChecked():
                continue
            path = Path(self.file_table.item(i, 1).data(Qt.UserRole))
            cmap_combo: QComboBox = self.file_table.cellWidget(i, 2)
            selected.append((i, path, cmap_combo.currentText()))

        if not selected:
            QMessageBox.warning(self, "No files", "No files selected.")
            return

        loaders: list[Any] = []
        filenames: list[str] = []
        colormaps: list[str] = []
        try:
            self._h5_file_manager = H5FileManager()
            for _, path, cmap in selected:
                loaders.append(H5Loader(path, self._h5_file_manager))
                filenames.append(path.name)
                colormaps.append(cmap)

            ref_meta = loaders[0].h5_metadata
            xy = ref_meta.get("voxel_size_xy_um")
            z = ref_meta.get("voxel_size_z_um")
            if xy and xy > 0:
                self.spin_voxel_xy.setValue(xy)
            if z and z > 0:
                self.spin_voxel_z.setValue(z)
            if xy or z:
                self.lbl_voxel_source.setText("from H5 metadata")
                self.lbl_voxel_source.setVisible(True)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        ok, msg = validate_channels(loaders)
        if not ok:
            QMessageBox.critical(self, "Validation Error", msg)
            for ld in loaders:
                ld.close()
            if self._h5_file_manager:
                self._h5_file_manager.close_all()
            return

        z_summary = z_dimensions_summary(loaders)
        z_vals = list(z_summary.values())
        if len(set(z_vals)) > 1:
            detail = "\n".join(f"  {k}: {v} planes" for k, v in z_summary.items())
            QMessageBox.warning(
                self, "Z Dimension Mismatch",
                f"Channels have different Z depths:\n{detail}\n\n"
                "The minimum Z will be used for shared operations.",
            )

        self.loaders = loaders
        self._filenames = filenames
        self._colormaps = colormaps

        # Add layers to napari
        for i, loader in enumerate(self.loaders):
            name = f"ch{i}_{filenames[i]}"
            cmap = colormaps[i]
            if len(loader.multiscale) > 1:
                self.viewer.add_image(
                    loader.multiscale, name=name, colormap=cmap,
                    blending="additive", visible=True, multiscale=True,
                )
            else:
                self.viewer.add_image(
                    loader.dask_array, name=name, colormap=cmap,
                    blending="additive", visible=True,
                )

        # Update channel selector
        self.combo_channel.clear()
        self.combo_channel.addItem("All channels")
        for i, fn in enumerate(filenames):
            self.combo_channel.addItem(f"ch{i}: {fn}")

        # Update Z spinboxes
        min_z = min(ld.shape[0] for ld in self.loaders)
        self.spin_z_start.setRange(0, max(min_z - 1, 0))
        self.spin_z_end.setRange(0, max(min_z - 1, 0))
        self.spin_z_start.setValue(0)
        self.spin_z_end.setValue(min(min_z - 1, 99))

    def _close_loaders(self) -> None:
        for ld in self.loaders:
            try:
                ld.close()
            except Exception:
                pass
        self.loaders.clear()
        self._filenames.clear()
        self._colormaps.clear()
        if self._h5_file_manager is not None:
            self._h5_file_manager.close_all()
            self._h5_file_manager = None

    # ------------------------------------------------------------------ #
    # ROI Preview
    # ------------------------------------------------------------------ #

    def _on_draw_roi(self) -> None:
        if self._shapes_layer is None or self._shapes_layer not in self.viewer.layers:
            self._shapes_layer = self.viewer.add_shapes(
                name="ROI", shape_type="rectangle",
                edge_color="white", face_color="transparent", edge_width=2,
            )
        self.viewer.layers.selection.active = self._shapes_layer
        self._shapes_layer.mode = "add_rectangle"

    def _on_load_preview(self) -> None:
        if not self.loaders:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        bounds = self._get_roi_bounds()
        if bounds is None:
            QMessageBox.warning(self, "No ROI", "Draw a rectangle ROI first.")
            return

        y_start, y_end, x_start, x_end = bounds
        z_start = self.spin_z_start.value()
        z_end = self.spin_z_end.value() + 1

        if z_end <= z_start:
            QMessageBox.warning(self, "Invalid Z", "Z end must be > Z start.")
            return

        self._on_clear_preview()

        for i, loader in enumerate(self.loaders):
            sub = extract_subvolume(
                loader.dask_array, z_start, z_end,
                y_start, y_end, x_start, x_end,
            )
            layer_name = f"{self._filenames[i]}_preview"
            self._preview_layer_names.append(layer_name)
            self.viewer.add_image(
                sub, name=layer_name, colormap=self._colormaps[i],
                blending="additive",
                translate=(z_start, y_start, x_start),
                visible=True,
            )

    def _on_clear_preview(self) -> None:
        for name in self._preview_layer_names:
            try:
                self.viewer.layers.remove(self.viewer.layers[name])
            except (KeyError, ValueError):
                pass
        self._preview_layer_names.clear()

    def _clear_processed_layers(self) -> None:
        for name in self._processed_layer_names:
            try:
                self.viewer.layers.remove(self.viewer.layers[name])
            except (KeyError, ValueError):
                pass
        self._processed_layer_names.clear()

    # ------------------------------------------------------------------ #
    # ROI Processing
    # ------------------------------------------------------------------ #

    def _on_process_roi(self) -> None:
        if not self.loaders:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        bounds = self._get_roi_bounds()
        if bounds is None:
            QMessageBox.warning(self, "No ROI", "Draw a rectangle ROI first.")
            return

        steps = self._collect_pipeline_steps()
        if not steps:
            QMessageBox.warning(self, "No Pipeline", "Configure at least one processing step.")
            return

        y_start, y_end, x_start, x_end = bounds
        z_start = self.spin_z_start.value()
        z_end = self.spin_z_end.value() + 1
        if z_end <= z_start:
            QMessageBox.warning(self, "Invalid Z", "Z end must be > Z start.")
            return

        channel_indices = self._get_selected_channel_indices()
        use_gpu = self.chk_use_gpu.isChecked()

        # Remove existing processed layers for these channels
        for ch_i in channel_indices:
            layer_name = f"{self._filenames[ch_i]}_processed"
            if layer_name in self._processed_layer_names:
                try:
                    self.viewer.layers.remove(self.viewer.layers[layer_name])
                except (KeyError, ValueError):
                    pass
                self._processed_layer_names.remove(layer_name)

        # Process channels sequentially via chained workers
        self._roi_process_queue = list(channel_indices)
        self._roi_process_bounds = (z_start, z_end, y_start, y_end, x_start, x_end)
        self._roi_process_steps = steps
        self._roi_process_gpu = use_gpu
        self._set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress.setVisible(True)
        self._process_next_roi_channel()

    def _process_next_roi_channel(self) -> None:
        if not self._roi_process_queue:
            self._set_ui_enabled(True)
            self.progress_bar.setVisible(False)
            self.lbl_progress.setVisible(False)
            return

        ch_i = self._roi_process_queue.pop(0)
        z_start, z_end, y_start, y_end, x_start, x_end = self._roi_process_bounds
        self.lbl_progress.setText(f"Processing ch{ch_i}: {self._filenames[ch_i]}...")

        sub = extract_subvolume(
            self.loaders[ch_i].dask_array, z_start, z_end,
            y_start, y_end, x_start, x_end,
        )

        self._current_roi_ch = ch_i
        self._processing_worker = ProcessingWorker(
            sub, self._roi_process_steps, self._roi_process_gpu,
        )
        self._processing_worker.progress.connect(self._on_roi_progress)
        self._processing_worker.finished.connect(self._on_roi_finished)
        self._processing_worker.error.connect(self._on_roi_error)
        self._processing_worker.start()

    def _on_roi_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setValue(int(100 * done / total))

    def _on_roi_finished(self, result: np.ndarray) -> None:
        ch_i = self._current_roi_ch
        z_start = self._roi_process_bounds[0]
        y_start = self._roi_process_bounds[2]
        x_start = self._roi_process_bounds[4]

        layer_name = f"{self._filenames[ch_i]}_processed"
        self._processed_layer_names.append(layer_name)
        self.viewer.add_image(
            result, name=layer_name, colormap=self._colormaps[ch_i],
            blending="additive",
            translate=(z_start, y_start, x_start),
            visible=True,
        )
        self._process_next_roi_channel()

    def _on_roi_error(self, tb: str) -> None:
        self._set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        QMessageBox.critical(self, "Processing Error", f"Processing failed:\n{tb}")

    # ------------------------------------------------------------------ #
    # Export Processed ROI
    # ------------------------------------------------------------------ #

    def _on_export_processed_roi(self) -> None:
        if not self.loaders:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        bounds = self._get_roi_bounds()
        if bounds is None:
            QMessageBox.warning(self, "No ROI", "Draw a rectangle ROI first.")
            return
        steps = self._collect_pipeline_steps()
        if not steps:
            QMessageBox.warning(self, "No Pipeline", "Configure at least one processing step.")
            return

        y_start, y_end, x_start, x_end = bounds
        z_start = self.spin_z_start.value()
        z_end = self.spin_z_end.value() + 1
        if z_end <= z_start:
            QMessageBox.warning(self, "Invalid Z", "Z end must be > Z start.")
            return

        channel_indices = self._get_selected_channel_indices()
        if len(channel_indices) == 1:
            default_name = f"{self._filenames[channel_indices[0]]}_processed.tif"
        else:
            default_name = "processed_roi.tif"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Processed ROI", default_name, "BigTIFF (*.tif)",
        )
        if not path:
            return

        ch_i = channel_indices[0]  # Export first selected channel
        sub = extract_subvolume(
            self.loaders[ch_i].dask_array, z_start, z_end,
            y_start, y_end, x_start, x_end,
        )

        self._set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress.setVisible(True)
        self.lbl_progress.setText("Exporting processed ROI...")

        self._export_worker = ExportProcessedWorker(
            sub, steps, self.chk_use_gpu.isChecked(), Path(path),
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_progress(self, done: int, total: int) -> None:
        if total > 0:
            pct = int(100 * done / total)
            self.progress_bar.setValue(pct)
            self.lbl_progress.setText(f"Exporting: {done}/{total} planes ({pct}%)")

    def _on_export_finished(self, path: str) -> None:
        self._set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        if path:
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")
        else:
            QMessageBox.information(self, "Cancelled", "Export was cancelled.")

    def _on_export_error(self, tb: str) -> None:
        self._set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        QMessageBox.critical(self, "Export Error", f"Export failed:\n{tb}")

    # ------------------------------------------------------------------ #
    # Full Stack Processing
    # ------------------------------------------------------------------ #

    def _on_process_full_stack(self) -> None:
        if not self.loaders:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        steps = self._collect_pipeline_steps()
        if not steps:
            QMessageBox.warning(self, "No Pipeline", "Configure at least one processing step.")
            return

        channel_indices = self._get_selected_channel_indices()

        # Estimate memory
        total_bytes = sum(
            self.loaders[i].shape[0] * self.loaders[i].shape[1] * self.loaders[i].shape[2] * 2
            for i in channel_indices
        )
        gb = total_bytes / (1024 ** 3)
        ans = QMessageBox.question(
            self, "Confirm Full Stack",
            f"This will process {len(channel_indices)} channel(s).\n"
            f"Estimated output size: {gb:.1f} GB\n\nContinue?",
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if ans != QMessageBox.Ok:
            return

        self._fullstack_queue = list(channel_indices)
        self._fullstack_steps = steps
        self._fullstack_gpu = self.chk_use_gpu.isChecked()
        self._set_ui_enabled(False)
        self.fullstack_progress.setVisible(True)
        self.fullstack_progress.setValue(0)
        self.lbl_fullstack_progress.setVisible(True)
        self._process_next_fullstack_channel()

    def _process_next_fullstack_channel(self) -> None:
        if not self._fullstack_queue:
            self._set_ui_enabled(True)
            self.fullstack_progress.setVisible(False)
            self.lbl_fullstack_progress.setVisible(False)
            return

        ch_i = self._fullstack_queue.pop(0)
        self._current_fullstack_ch = ch_i
        self.lbl_fullstack_progress.setText(
            f"Processing ch{ch_i}: {self._filenames[ch_i]}..."
        )

        # Remove existing processed layer for this channel
        layer_name = f"{self._filenames[ch_i]}_processed"
        if layer_name in self._processed_layer_names:
            try:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
            except (KeyError, ValueError):
                pass
            self._processed_layer_names.remove(layer_name)

        self._fullstack_worker = FullStackProcessingWorker(
            self.loaders[ch_i].dask_array,
            self._fullstack_steps,
            self._fullstack_gpu,
        )
        self._fullstack_worker.progress.connect(self._on_fullstack_progress)
        self._fullstack_worker.finished.connect(self._on_fullstack_finished)
        self._fullstack_worker.error.connect(self._on_fullstack_error)
        self._fullstack_worker.start()

    def _on_fullstack_progress(self, done: int, total: int) -> None:
        if total > 0:
            pct = int(100 * done / total)
            self.fullstack_progress.setValue(pct)
            self.lbl_fullstack_progress.setText(
                f"Processing slice {done}/{total} ({pct}%)"
            )

    def _on_fullstack_finished(self, result: np.ndarray) -> None:
        ch_i = self._current_fullstack_ch
        layer_name = f"{self._filenames[ch_i]}_processed"
        self._processed_layer_names.append(layer_name)
        self.viewer.add_image(
            result, name=layer_name, colormap=self._colormaps[ch_i],
            blending="additive", visible=True,
        )
        self._process_next_fullstack_channel()

    def _on_fullstack_error(self, tb: str) -> None:
        self._set_ui_enabled(True)
        self.fullstack_progress.setVisible(False)
        self.lbl_fullstack_progress.setVisible(False)
        QMessageBox.critical(self, "Processing Error", f"Processing failed:\n{tb}")

    # ------------------------------------------------------------------ #
    # UI state
    # ------------------------------------------------------------------ #

    def _set_ui_enabled(self, enabled: bool) -> None:
        for w in [
            self.btn_select_dir,
            self.btn_load,
            self.btn_draw_roi,
            self.btn_load_preview,
            self.btn_clear_preview,
            self.btn_process_roi,
            self.btn_export_roi,
            self.btn_process_full,
            self.file_table,
            self.combo_channel,
            self.chk_use_gpu,
        ]:
            w.setEnabled(enabled)
