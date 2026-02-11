"""Main docked widget (Qt-based panel) for the Chromatic Shift Corrector."""

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
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QDoubleSpinBox,
    QHeaderView,
    QAbstractItemView,
    QRadioButton,
    QButtonGroup,
)

from chromatic_shift_corrector.data_loader import (
    BigTIFFLoader,
    scan_bigtiff_files,
    validate_channels,
    z_dimensions_summary,
)
from chromatic_shift_corrector.export_engine import (
    compute_chunk_size,
    estimate_output_sizes,
    run_export,
)
from chromatic_shift_corrector.preview_engine import generate_preview
from chromatic_shift_corrector.shift_manager import ShiftManager
from chromatic_shift_corrector.utils import DEFAULT_COLORMAPS, MAX_CHANNELS, parse_voxel_size_from_xml

# Available napari colormaps for the dropdown.
COLORMAP_OPTIONS = [
    "green",
    "magenta",
    "cyan",
    "yellow",
    "red",
    "blue",
    "gray",
    "viridis",
    "inferno",
    "turbo",
]


class ExportWorker(QThread):
    """Background thread for full-volume export."""

    progress = Signal(int, int)  # (planes_done, total_planes)
    finished = Signal(str)  # metadata path or empty on cancel
    error = Signal(str)

    def __init__(
        self,
        loaders: list[BigTIFFLoader],
        shift_manager: ShiftManager,
        output_dir: Path,
        ram_percent: int,
        voxel_xy: float,
        voxel_z: float,
    ) -> None:
        super().__init__()
        self.loaders = loaders
        self.shift_manager = shift_manager
        self.output_dir = output_dir
        self.ram_percent = ram_percent
        self.voxel_xy = voxel_xy
        self.voxel_z = voxel_z
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            meta_path = run_export(
                self.loaders,
                self.shift_manager,
                self.output_dir,
                self.ram_percent,
                progress_callback=lambda done, total: self.progress.emit(done, total),
                cancel_check=lambda: self._cancelled,
                voxel_xy=self.voxel_xy,
                voxel_z=self.voxel_z,
            )
            self.finished.emit(str(meta_path) if not self._cancelled else "")
        except Exception:
            self.error.emit(traceback.format_exc())


class ChromaticShiftWidget(QWidget):
    """Docked widget providing the full chromatic-shift-correction workflow."""

    def __init__(self, napari_viewer: Any) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.shift_manager = ShiftManager()
        self.loaders: list[BigTIFFLoader] = []
        self._preview_layer_names: list[str] = []
        self._shapes_layer = None
        self._export_worker: ExportWorker | None = None

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
        layout.addWidget(self._build_shift_section())
        layout.addWidget(self._build_preview_section())
        layout.addWidget(self._build_export_section())

        layout.addStretch()
        container.setLayout(layout)
        scroll.setWidget(container)
        outer.addWidget(scroll)
        self.setLayout(outer)

    # ---- Data Loading Section ---------------------------------------- #

    def _build_data_section(self) -> QGroupBox:
        grp = QGroupBox("Data Loading")
        lay = QVBoxLayout()

        # Directory selector
        row_dir = QHBoxLayout()
        self.btn_select_dir = QPushButton("Select Directory")
        self.btn_select_dir.clicked.connect(self._on_select_directory)
        self.lbl_dir = QLineEdit()
        self.lbl_dir.setReadOnly(True)
        self.lbl_dir.setPlaceholderText("No directory selected")
        row_dir.addWidget(self.btn_select_dir)
        row_dir.addWidget(self.lbl_dir)
        lay.addLayout(row_dir)

        # File table: checkbox | filename | channel order | reference radio | colormap
        self.file_table = QTableWidget(0, 5)
        self.file_table.setHorizontalHeaderLabels(
            ["Include", "Filename", "Channel #", "Reference", "Colormap"]
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.file_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.file_table.verticalHeader().setVisible(False)
        self.ref_button_group = QButtonGroup(self)
        lay.addWidget(self.file_table)

        # Voxel metadata
        row_voxel = QHBoxLayout()
        row_voxel.addWidget(QLabel("XY pixel (µm):"))
        self.spin_voxel_xy = QDoubleSpinBox()
        self.spin_voxel_xy.setDecimals(4)
        self.spin_voxel_xy.setRange(0.0001, 100.0)
        self.spin_voxel_xy.setValue(0.3)
        row_voxel.addWidget(self.spin_voxel_xy)
        row_voxel.addWidget(QLabel("Z step (µm):"))
        self.spin_voxel_z = QDoubleSpinBox()
        self.spin_voxel_z.setDecimals(4)
        self.spin_voxel_z.setRange(0.0001, 100.0)
        self.spin_voxel_z.setValue(1.0)
        row_voxel.addWidget(self.spin_voxel_z)
        lay.addLayout(row_voxel)

        # Load button
        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self._on_load_data)
        lay.addWidget(self.btn_load)

        grp.setLayout(lay)
        return grp

    # ---- Shift Specification Section --------------------------------- #

    def _build_shift_section(self) -> QGroupBox:
        grp = QGroupBox("Shift Specification")
        lay = QVBoxLayout()

        self.shift_table = QTableWidget(0, 5)
        self.shift_table.setHorizontalHeaderLabels(
            ["Channel", "Colormap", "X shift", "Y shift", "Z shift"]
        )
        self.shift_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.shift_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.shift_table.verticalHeader().setVisible(False)
        lay.addWidget(self.shift_table)

        # Axis labels
        lay.addWidget(
            QLabel("Positive X = right, Positive Y = down, Positive Z = higher index")
        )

        self.btn_reset_shifts = QPushButton("Reset All Shifts")
        self.btn_reset_shifts.clicked.connect(self._on_reset_shifts)
        lay.addWidget(self.btn_reset_shifts)

        grp.setLayout(lay)
        return grp

    # ---- ROI Preview Section ----------------------------------------- #

    def _build_preview_section(self) -> QGroupBox:
        grp = QGroupBox("ROI Preview")
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

        row_btns = QHBoxLayout()
        self.btn_load_preview = QPushButton("Load Preview")
        self.btn_load_preview.clicked.connect(self._on_load_preview)
        self.btn_clear_preview = QPushButton("Clear Preview")
        self.btn_clear_preview.clicked.connect(self._on_clear_preview)
        row_btns.addWidget(self.btn_load_preview)
        row_btns.addWidget(self.btn_clear_preview)
        lay.addLayout(row_btns)

        grp.setLayout(lay)
        return grp

    # ---- Export Section ----------------------------------------------- #

    def _build_export_section(self) -> QGroupBox:
        grp = QGroupBox("Export")
        lay = QVBoxLayout()

        row_outdir = QHBoxLayout()
        self.btn_select_outdir = QPushButton("Select Output Directory")
        self.btn_select_outdir.clicked.connect(self._on_select_output_dir)
        self.lbl_outdir = QLineEdit()
        self.lbl_outdir.setReadOnly(True)
        self.lbl_outdir.setPlaceholderText("No output directory selected")
        row_outdir.addWidget(self.btn_select_outdir)
        row_outdir.addWidget(self.lbl_outdir)
        lay.addLayout(row_outdir)

        row_ram = QHBoxLayout()
        row_ram.addWidget(QLabel("RAM allocation:"))
        self.slider_ram = QSlider(Qt.Horizontal)
        self.slider_ram.setRange(50, 95)
        self.slider_ram.setValue(90)
        self.slider_ram.setTickPosition(QSlider.TicksBelow)
        self.slider_ram.setTickInterval(5)
        self.lbl_ram = QLabel("90%")
        self.slider_ram.valueChanged.connect(
            lambda v: self.lbl_ram.setText(f"{v}%")
        )
        row_ram.addWidget(self.slider_ram)
        row_ram.addWidget(self.lbl_ram)
        lay.addLayout(row_ram)

        self.btn_export = QPushButton("Apply && Export")
        self.btn_export.clicked.connect(self._on_export)
        lay.addWidget(self.btn_export)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("")
        self.lbl_progress.setVisible(False)
        lay.addWidget(self.lbl_progress)

        grp.setLayout(lay)
        return grp

    # ------------------------------------------------------------------ #
    # Callbacks — Data Loading
    # ------------------------------------------------------------------ #

    def _on_select_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if not path:
            return
        self.lbl_dir.setText(path)
        dir_path = Path(path)
        files = scan_bigtiff_files(dir_path)
        self._populate_file_table(files)

        # Auto-fill voxel sizes from XML metadata if available.
        voxel = parse_voxel_size_from_xml(dir_path)
        if voxel is not None:
            self.spin_voxel_xy.setValue(voxel[0])
            self.spin_voxel_z.setValue(voxel[1])

    def _populate_file_table(self, files: list[Path]) -> None:
        # Clear existing radio buttons from group
        for btn in self.ref_button_group.buttons():
            self.ref_button_group.removeButton(btn)

        n = min(len(files), MAX_CHANNELS)
        self.file_table.setRowCount(n)
        for i in range(n):
            # Checkbox
            chk = QCheckBox()
            chk.setChecked(True)
            self.file_table.setCellWidget(i, 0, chk)

            # Filename
            item = QTableWidgetItem(files[i].name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, str(files[i]))  # store full path
            self.file_table.setItem(i, 1, item)

            # Channel order
            spin = QSpinBox()
            spin.setRange(0, MAX_CHANNELS - 1)
            spin.setValue(i)
            self.file_table.setCellWidget(i, 2, spin)

            # Reference radio
            radio = QRadioButton()
            if i == 0:
                radio.setChecked(True)
            self.ref_button_group.addButton(radio, i)
            self.file_table.setCellWidget(i, 3, radio)

            # Colormap
            combo = QComboBox()
            combo.addItems(COLORMAP_OPTIONS)
            default_idx = i if i < len(DEFAULT_COLORMAPS) else 0
            try:
                combo.setCurrentIndex(COLORMAP_OPTIONS.index(DEFAULT_COLORMAPS[default_idx]))
            except (ValueError, IndexError):
                combo.setCurrentIndex(0)
            self.file_table.setCellWidget(i, 4, combo)

    def _on_load_data(self) -> None:
        # Close any previously loaded data.
        self._close_loaders()
        self._on_clear_preview()

        # Remove existing image layers.
        layers_to_remove = [l for l in self.viewer.layers if l.name.startswith("ch")]
        for l in layers_to_remove:
            self.viewer.layers.remove(l)

        # Gather selected files in channel order.
        rows = self.file_table.rowCount()
        selected: list[tuple[int, Path, str]] = []  # (order, path, colormap)
        ref_row = self.ref_button_group.checkedId()

        for i in range(rows):
            chk: QCheckBox = self.file_table.cellWidget(i, 0)
            if not chk.isChecked():
                continue
            order_spin: QSpinBox = self.file_table.cellWidget(i, 2)
            path = Path(self.file_table.item(i, 1).data(Qt.UserRole))
            cmap_combo: QComboBox = self.file_table.cellWidget(i, 4)
            selected.append((order_spin.value(), path, cmap_combo.currentText()))

        if not selected:
            QMessageBox.warning(self, "No files", "No files selected.")
            return

        selected.sort(key=lambda t: t[0])

        # Load files.
        loaders: list[BigTIFFLoader] = []
        filenames: list[str] = []
        colormaps: list[str] = []
        try:
            for _, path, cmap in selected:
                loaders.append(BigTIFFLoader(path))
                filenames.append(path.name)
                colormaps.append(cmap)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        # Validate.
        ok, msg = validate_channels(loaders)
        if not ok:
            QMessageBox.critical(self, "Validation Error", msg)
            for ld in loaders:
                ld.close()
            return

        # Warn about mismatched Z.
        z_summary = z_dimensions_summary(loaders)
        z_vals = list(z_summary.values())
        if len(set(z_vals)) > 1:
            detail = "\n".join(f"  {k}: {v} planes" for k, v in z_summary.items())
            QMessageBox.warning(
                self,
                "Z Dimension Mismatch",
                f"Channels have different Z depths:\n{detail}\n\n"
                "The minimum Z will be used for shared operations.",
            )

        self.loaders = loaders

        # Determine reference index within the selected set.
        # ref_row is the row index in the file table; map to channel index.
        ref_channel = 0
        for idx, (order_val, path, _) in enumerate(selected):
            chk_row = None
            for r in range(self.file_table.rowCount()):
                item = self.file_table.item(r, 1)
                if item and Path(item.data(Qt.UserRole)) == path:
                    chk_row = r
                    break
            if chk_row == ref_row:
                ref_channel = idx
                break

        self.shift_manager.init_channels(filenames, ref_channel, colormaps)

        # Add layers to napari.
        for i, loader in enumerate(self.loaders):
            t = self.shift_manager[i]
            self.viewer.add_image(
                loader.dask_array,
                name=f"ch{i}_{t.filename}",
                colormap=t.colormap,
                blending="additive",
                visible=True,
            )

        # Update shift table.
        self._rebuild_shift_table()

        # Update Z spinboxes.
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

    # ------------------------------------------------------------------ #
    # Callbacks — Shift Table
    # ------------------------------------------------------------------ #

    def _rebuild_shift_table(self) -> None:
        n = len(self.shift_manager)
        self.shift_table.setRowCount(n)
        for i in range(n):
            t = self.shift_manager[i]

            # Channel name
            name_item = QTableWidgetItem(t.filename)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.shift_table.setItem(i, 0, name_item)

            # Colormap label
            cmap_item = QTableWidgetItem(t.colormap)
            cmap_item.setFlags(cmap_item.flags() & ~Qt.ItemIsEditable)
            self.shift_table.setItem(i, 1, cmap_item)

            # X, Y, Z spinboxes
            for col, axis in [(2, "x"), (3, "y"), (4, "z")]:
                spin = QSpinBox()
                spin.setRange(-9999, 9999)
                spin.setValue(getattr(t, f"shift_{axis}"))
                spin.setEnabled(not t.is_reference)
                spin.valueChanged.connect(
                    self._make_shift_callback(i, axis)
                )
                self.shift_table.setCellWidget(i, col, spin)

    def _make_shift_callback(self, channel_idx: int, axis: str):
        def _cb(value: int) -> None:
            self.shift_manager.set_shift(channel_idx, axis, value)
        return _cb

    def _on_reset_shifts(self) -> None:
        self.shift_manager.reset_all()
        self._rebuild_shift_table()

    # ------------------------------------------------------------------ #
    # Callbacks — ROI Preview
    # ------------------------------------------------------------------ #

    def _on_draw_roi(self) -> None:
        """Activate the rectangle drawing tool in napari."""
        if self._shapes_layer is None or self._shapes_layer not in self.viewer.layers:
            self._shapes_layer = self.viewer.add_shapes(
                name="ROI",
                shape_type="rectangle",
                edge_color="white",
                face_color="transparent",
                edge_width=2,
            )
        self.viewer.layers.selection.active = self._shapes_layer
        self._shapes_layer.mode = "add_rectangle"

    def _get_roi_bounds(self) -> tuple[int, int, int, int] | None:
        """Extract the bounding box of the last drawn rectangle.

        Returns (y_start, y_end, x_start, x_end) or None.
        """
        if self._shapes_layer is None or len(self._shapes_layer.data) == 0:
            return None
        rect = self._shapes_layer.data[-1]  # Last drawn shape
        # rect is Nx2 array of (row, col) vertices
        ys = rect[:, 0]
        xs = rect[:, 1]
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    def _on_load_preview(self) -> None:
        if not self.loaders:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        bounds = self._get_roi_bounds()
        if bounds is None:
            QMessageBox.warning(
                self, "No ROI", "Draw a rectangle ROI on the viewer first."
            )
            return

        y_start, y_end, x_start, x_end = bounds
        z_start = self.spin_z_start.value()
        z_end = self.spin_z_end.value() + 1  # make inclusive→exclusive

        if z_end <= z_start:
            QMessageBox.warning(self, "Invalid Z", "Z end must be > Z start.")
            return

        if not self.shift_manager.has_any_shift():
            QMessageBox.information(
                self,
                "No Shifts",
                "No shifts are specified — preview will be identical to the original.",
            )

        # Clear previous previews.
        self._on_clear_preview()

        for i, loader in enumerate(self.loaders):
            t = self.shift_manager[i]
            preview = generate_preview(
                loader.dask_array,
                t,
                z_start,
                z_end,
                y_start,
                y_end,
                x_start,
                x_end,
            )
            layer_name = f"{t.filename}_preview_corrected"
            self._preview_layer_names.append(layer_name)
            self.viewer.add_image(
                preview,
                name=layer_name,
                colormap=t.colormap,
                blending="additive",
                translate=(z_start, y_start, x_start),
                visible=True,
            )

    def _on_clear_preview(self) -> None:
        for name in self._preview_layer_names:
            try:
                layer = self.viewer.layers[name]
                self.viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass
        self._preview_layer_names.clear()

    # ------------------------------------------------------------------ #
    # Callbacks — Export
    # ------------------------------------------------------------------ #

    def _on_select_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.lbl_outdir.setText(path)

    def _on_export(self) -> None:
        if not self.loaders:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        outdir = self.lbl_outdir.text().strip()
        if not outdir:
            QMessageBox.warning(self, "No Output Dir", "Select an output directory.")
            return

        outdir_path = Path(outdir)

        # Check for existing corrected files.
        channel_dicts = self.shift_manager.to_channel_dicts()
        existing = [
            d["filename_corrected"]
            for d in channel_dicts
            if (outdir_path / d["filename_corrected"]).exists()
        ]
        if existing:
            ans = QMessageBox.question(
                self,
                "Overwrite?",
                f"The following files already exist and will be overwritten:\n"
                + "\n".join(f"  {f}" for f in existing)
                + "\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        # Build confirmation dialog.
        ram_pct = self.slider_ram.value()
        sizes = estimate_output_sizes(self.loaders)
        ref_shape = self.loaders[0].shape

        lines = ["Shifts to apply:\n"]
        for i, t in enumerate(self.shift_manager.transforms):
            tag = " (reference)" if t.is_reference else ""
            lines.append(
                f"  ch{i} {t.filename}{tag}: "
                f"X={t.shift_x:+d}, Y={t.shift_y:+d}, Z={t.shift_z:+d}"
            )
            # Edges that become zeros.
            for axis, val, dim_name in [
                (t.shift_x, ref_shape[2], "X"),
                (t.shift_y, ref_shape[1], "Y"),
                (t.shift_z, ref_shape[0], "Z"),
            ]:
                if axis != 0:
                    side = "end" if axis > 0 else "start"
                    lines.append(
                        f"      {dim_name}: {abs(axis)} zero-padded voxels at {side}"
                    )

        lines.append(f"\nOutput dimensions: {ref_shape[2]}x{ref_shape[1]}x{ref_shape[0]} (XYZ)")
        total_bytes = sum(sizes)
        lines.append(f"Estimated total output size: {total_bytes / (1024**3):.2f} GB")
        lines.append(f"RAM allocation: {ram_pct}%")

        ans = QMessageBox.question(
            self,
            "Confirm Export",
            "\n".join(lines),
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if ans != QMessageBox.Ok:
            return

        # Disable UI and start export.
        self._set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress.setVisible(True)
        self.lbl_progress.setText("Starting export...")

        self._export_worker = ExportWorker(
            self.loaders,
            self.shift_manager,
            outdir_path,
            ram_pct,
            self.spin_voxel_xy.value(),
            self.spin_voxel_z.value(),
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_progress(self, done: int, total: int) -> None:
        if total > 0:
            pct = int(100 * done / total)
            self.progress_bar.setValue(pct)
            self.lbl_progress.setText(f"Processing: {done}/{total} planes ({pct}%)")

    def _on_export_finished(self, meta_path: str) -> None:
        self._set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        if meta_path:
            QMessageBox.information(
                self,
                "Export Complete",
                f"All channels exported successfully.\nMetadata: {meta_path}",
            )
        else:
            QMessageBox.information(self, "Cancelled", "Export was cancelled.")

    def _on_export_error(self, tb: str) -> None:
        self._set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        QMessageBox.critical(self, "Export Error", f"Export failed:\n{tb}")

    def _set_ui_enabled(self, enabled: bool) -> None:
        """Enable or disable all interactive widgets."""
        for w in [
            self.btn_select_dir,
            self.btn_load,
            self.btn_reset_shifts,
            self.btn_draw_roi,
            self.btn_load_preview,
            self.btn_clear_preview,
            self.btn_select_outdir,
            self.btn_export,
            self.slider_ram,
            self.file_table,
            self.shift_table,
        ]:
            w.setEnabled(enabled)
