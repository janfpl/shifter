"""Main docked widget (Qt-based panel) for the Chromatic Shift Corrector."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QThread, Signal, Qt
from qtpy.QtGui import QColor
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
    H5Loader,
    scan_bigtiff_files,
    validate_channels,
    z_dimensions_summary,
)
from chromatic_shift_corrector.export_engine import (
    compute_chunk_size,
    estimate_output_sizes,
    run_export,
    run_export_h5,
)
from chromatic_shift_corrector.h5_utils import H5FileManager, scan_h5_files
from chromatic_shift_corrector.preview_engine import extract_subvolume, generate_preview
from chromatic_shift_corrector.shift_manager import ShiftManager
from chromatic_shift_corrector.utils import DEFAULT_COLORMAPS, MAX_CHANNELS, parse_voxel_size_from_xml
from chromatic_shift_corrector.registration import (
    ALGORITHM_REGISTRY,
    MAX_SEARCH_RANGE,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    GUIDANCE_TEXT,
    RegistrationResult,
    confidence_color_rgb,
    gpu_available,
    gpu_name,
    preprocess,
)

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
    """Background thread for full-volume export (BigTIFF or H5)."""

    progress = Signal(int, int)  # (planes_done, total_planes)
    finished = Signal(str)  # metadata path or empty on cancel
    error = Signal(str)

    def __init__(
        self,
        loaders: list[Any],
        shift_manager: ShiftManager,
        output_dir: Path,
        ram_percent: int,
        voxel_xy: float,
        voxel_z: float,
        input_format: str = "bigtiff",
    ) -> None:
        super().__init__()
        self.loaders = loaders
        self.shift_manager = shift_manager
        self.output_dir = output_dir
        self.ram_percent = ram_percent
        self.voxel_xy = voxel_xy
        self.voxel_z = voxel_z
        self.input_format = input_format
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            export_fn = run_export_h5 if self.input_format == "h5" else run_export
            meta_path = export_fn(
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


class RegistrationWorker(QThread):
    """Background thread for auto-registration."""

    progress = Signal(int, int, str)  # (done, total, description)
    finished = Signal(list)  # list of (channel_index, RegistrationResult)
    error = Signal(str)

    def __init__(
        self,
        loaders: list[BigTIFFLoader],
        reference_index: int,
        channels_to_register: list[int],
        algorithm_name: str,
        algorithm_kwargs: dict,
        search_range_xy: int,
        search_range_z: int,
        roi_bounds: tuple[int, int, int, int],
        z_start: int,
        z_end: int,
        background_subtraction: bool,
        gaussian_smoothing: bool,
        use_gpu: bool,
    ) -> None:
        super().__init__()
        self.loaders = loaders
        self.reference_index = reference_index
        self.channels_to_register = channels_to_register
        self.algorithm_name = algorithm_name
        self.algorithm_kwargs = algorithm_kwargs
        self.search_range_xy = search_range_xy
        self.search_range_z = search_range_z
        self.roi_bounds = roi_bounds
        self.z_start = z_start
        self.z_end = z_end
        self.background_subtraction = background_subtraction
        self.gaussian_smoothing = gaussian_smoothing
        self.use_gpu = use_gpu

    def run(self) -> None:
        try:
            results = self._run_registration()
            self.finished.emit(results)
        except Exception:
            self.error.emit(traceback.format_exc())

    def _run_registration(self) -> list:
        y_start, y_end, x_start, x_end = self.roi_bounds

        # Extract reference sub-volume.
        ref_loader = self.loaders[self.reference_index]
        ref_vol = extract_subvolume(
            ref_loader.dask_array,
            self.z_start, self.z_end,
            y_start, y_end, x_start, x_end,
        )

        # Preprocess reference.
        ref_vol = preprocess(
            ref_vol,
            background_subtraction=self.background_subtraction,
            gaussian_smoothing=self.gaussian_smoothing,
            use_gpu=self.use_gpu,
        )

        # Instantiate algorithm.
        algo_cls = ALGORITHM_REGISTRY[self.algorithm_name]
        algo = algo_cls(**self.algorithm_kwargs)

        total = len(self.channels_to_register)
        results = []

        for idx, ch_i in enumerate(self.channels_to_register):
            loader = self.loaders[ch_i]
            ch_name = Path(loader.dask_array.name).name if hasattr(loader.dask_array, 'name') else f"channel {ch_i}"
            self.progress.emit(idx, total, f"Registering channel {idx + 1}/{total}...")

            # Extract moving sub-volume.
            mov_vol = extract_subvolume(
                loader.dask_array,
                self.z_start, self.z_end,
                y_start, y_end, x_start, x_end,
            )

            # Preprocess moving.
            mov_vol = preprocess(
                mov_vol,
                background_subtraction=self.background_subtraction,
                gaussian_smoothing=self.gaussian_smoothing,
                use_gpu=self.use_gpu,
            )

            # Run registration with GPU OOM fallback.
            try:
                result = algo.register(
                    ref_vol, mov_vol,
                    self.search_range_xy, self.search_range_z,
                    use_gpu=self.use_gpu,
                )
            except Exception:
                # If GPU fails (e.g. OOM), retry on CPU.
                if self.use_gpu:
                    result = algo.register(
                        ref_vol, mov_vol,
                        self.search_range_xy, self.search_range_z,
                        use_gpu=False,
                    )
                else:
                    raise

            results.append((ch_i, result))

        self.progress.emit(total, total, "Registration complete.")
        return results


class ChromaticShiftWidget(QWidget):
    """Docked widget providing the full chromatic-shift-correction workflow."""

    def __init__(self, napari_viewer: Any) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.shift_manager = ShiftManager()
        self.loaders: list[Any] = []  # BigTIFFLoader or H5Loader
        self._preview_layer_names: list[str] = []
        self._shapes_layer = None
        self._export_worker: ExportWorker | None = None
        self._registration_worker: RegistrationWorker | None = None

        # Per-channel confidence scores (channel_index -> confidence float).
        self._confidence_scores: dict[int, float] = {}

        # Format state: "bigtiff" or "h5".
        self._input_format: str = "bigtiff"
        self._h5_file_manager: H5FileManager | None = None

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
        layout.addWidget(self._build_registration_section())
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

        # Format selector
        row_format = QHBoxLayout()
        row_format.addWidget(QLabel("Format:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["BigTIFF (.tif)", "Luxendo H5 (.lux.h5)"])
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)
        row_format.addWidget(self.combo_format)
        lay.addLayout(row_format)

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

        # Voxel source label (shown for H5 when auto-filled from metadata).
        self.lbl_voxel_source = QLabel("")
        self.lbl_voxel_source.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_voxel_source.setVisible(False)
        lay.addWidget(self.lbl_voxel_source)

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

        self.shift_table = QTableWidget(0, 6)
        self.shift_table.setHorizontalHeaderLabels(
            ["Channel", "Colormap", "X shift", "Y shift", "Z shift", "Confidence"]
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

        # Confidence guidance text
        self.lbl_confidence_guide = QLabel(GUIDANCE_TEXT)
        self.lbl_confidence_guide.setWordWrap(True)
        self.lbl_confidence_guide.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_confidence_guide.setVisible(False)
        lay.addWidget(self.lbl_confidence_guide)

        self.btn_reset_shifts = QPushButton("Reset All Shifts")
        self.btn_reset_shifts.clicked.connect(self._on_reset_shifts)
        lay.addWidget(self.btn_reset_shifts)

        grp.setLayout(lay)
        return grp

    # ---- Auto-Registration Section ----------------------------------- #

    def _build_registration_section(self) -> QGroupBox:
        grp = QGroupBox("Auto-Registration")
        lay = QVBoxLayout()

        # Algorithm selector.
        row_algo = QHBoxLayout()
        row_algo.addWidget(QLabel("Algorithm:"))
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems(list(ALGORITHM_REGISTRY.keys()))
        self.combo_algorithm.currentTextChanged.connect(self._on_algorithm_changed)
        row_algo.addWidget(self.combo_algorithm)
        lay.addLayout(row_algo)

        # Algorithm-specific options container.
        self._algo_options_container = QWidget()
        self._algo_options_layout = QVBoxLayout()
        self._algo_options_layout.setContentsMargins(0, 0, 0, 0)

        # Phase correlation normalization option.
        row_norm = QHBoxLayout()
        row_norm.addWidget(QLabel("Normalization:"))
        self.combo_normalization = QComboBox()
        self.combo_normalization.addItems(["Phase", "None"])
        row_norm.addWidget(self.combo_normalization)
        self._phase_norm_widget = QWidget()
        self._phase_norm_widget.setLayout(row_norm)
        self._algo_options_layout.addWidget(self._phase_norm_widget)

        self._algo_options_container.setLayout(self._algo_options_layout)
        lay.addWidget(self._algo_options_container)

        # Channel selection.
        lay.addWidget(QLabel("Channels to register:"))
        self._reg_channel_checkboxes: list[tuple[int, QCheckBox]] = []
        self._reg_channels_container = QWidget()
        self._reg_channels_layout = QVBoxLayout()
        self._reg_channels_layout.setContentsMargins(0, 0, 0, 0)
        self._reg_channels_container.setLayout(self._reg_channels_layout)
        lay.addWidget(self._reg_channels_container)

        # Search range.
        row_sr_xy = QHBoxLayout()
        row_sr_xy.addWidget(QLabel("XY search range (voxels):"))
        self.spin_sr_xy = QSpinBox()
        self.spin_sr_xy.setRange(1, MAX_SEARCH_RANGE)
        self.spin_sr_xy.setValue(20)
        row_sr_xy.addWidget(self.spin_sr_xy)
        lay.addLayout(row_sr_xy)

        row_sr_z = QHBoxLayout()
        row_sr_z.addWidget(QLabel("Z search range (voxels):"))
        self.spin_sr_z = QSpinBox()
        self.spin_sr_z.setRange(1, MAX_SEARCH_RANGE)
        self.spin_sr_z.setValue(50)
        row_sr_z.addWidget(self.spin_sr_z)
        lay.addLayout(row_sr_z)

        # Preprocessing toggles.
        self.chk_bg_sub = QCheckBox("Background subtraction (percentile-based)")
        self.chk_bg_sub.setChecked(False)
        lay.addWidget(self.chk_bg_sub)

        self.chk_gaussian = QCheckBox("Gaussian smoothing (\u03c3 = 2, 2, 1)")
        self.chk_gaussian.setChecked(False)
        lay.addWidget(self.chk_gaussian)

        # GPU indicator.
        self.lbl_gpu_status = QLabel()
        self._update_gpu_indicator()
        lay.addWidget(self.lbl_gpu_status)

        # Run button.
        self.btn_run_registration = QPushButton("Run Auto-Registration")
        self.btn_run_registration.clicked.connect(self._on_run_registration)
        self.btn_run_registration.setEnabled(False)
        self.btn_run_registration.setToolTip("Draw an ROI and specify Z range first.")
        lay.addWidget(self.btn_run_registration)

        # Progress.
        self.reg_progress_bar = QProgressBar()
        self.reg_progress_bar.setVisible(False)
        lay.addWidget(self.reg_progress_bar)

        self.lbl_reg_progress = QLabel("")
        self.lbl_reg_progress.setVisible(False)
        lay.addWidget(self.lbl_reg_progress)

        # Results summary.
        self.lbl_reg_result = QLabel("")
        self.lbl_reg_result.setWordWrap(True)
        self.lbl_reg_result.setVisible(False)
        lay.addWidget(self.lbl_reg_result)

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

        # Output format indicator.
        self.lbl_output_format = QLabel("Output format: BigTIFF")
        self.lbl_output_format.setStyleSheet("font-weight: bold;")
        lay.addWidget(self.lbl_output_format)

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
    # Helpers
    # ------------------------------------------------------------------ #

    def _update_gpu_indicator(self) -> None:
        """Update the GPU status label."""
        if gpu_available():
            name = gpu_name()
            self.lbl_gpu_status.setText(f"\u25cf GPU: {name}")
            self.lbl_gpu_status.setStyleSheet("color: #4a90d9; font-weight: bold;")
        else:
            self.lbl_gpu_status.setText("\u25cf CPU mode")
            self.lbl_gpu_status.setStyleSheet("color: #999; font-weight: bold;")

    def _on_algorithm_changed(self, text: str) -> None:
        """Show/hide algorithm-specific options."""
        self._phase_norm_widget.setVisible(text == "Phase Cross-Correlation")

    def _rebuild_registration_channels(self) -> None:
        """Rebuild the channel checkboxes in the auto-registration section."""
        # Clear existing.
        for _, chk in self._reg_channel_checkboxes:
            self._reg_channels_layout.removeWidget(chk)
            chk.deleteLater()
        self._reg_channel_checkboxes.clear()

        ref_idx = self.shift_manager.reference_index
        for i in range(len(self.shift_manager)):
            t = self.shift_manager[i]
            if t.is_reference:
                continue
            chk = QCheckBox(f"ch{i}: {t.filename}")
            chk.setChecked(True)
            self._reg_channels_layout.addWidget(chk)
            self._reg_channel_checkboxes.append((i, chk))

    def _has_roi(self) -> bool:
        """Check if an ROI rectangle has been drawn."""
        return (
            self._shapes_layer is not None
            and self._shapes_layer in self.viewer.layers
            and len(self._shapes_layer.data) > 0
        )

    def _update_run_button_state(self) -> None:
        """Enable the Run button only when ROI is defined and data loaded."""
        enabled = bool(self.loaders) and self._has_roi()
        self.btn_run_registration.setEnabled(enabled)
        if not enabled:
            self.btn_run_registration.setToolTip(
                "Draw an ROI and specify Z range first."
            )
        else:
            self.btn_run_registration.setToolTip("")

    def _on_format_changed(self, index: int) -> None:
        """Handle format dropdown change."""
        new_format = "h5" if index == 1 else "bigtiff"

        # If data is loaded and format is changing, confirm clear.
        if self.loaders and new_format != self._input_format:
            ans = QMessageBox.question(
                self,
                "Change Format",
                "Changing the format will unload all data. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                # Revert combo without triggering signal.
                self.combo_format.blockSignals(True)
                self.combo_format.setCurrentIndex(0 if self._input_format == "bigtiff" else 1)
                self.combo_format.blockSignals(False)
                return
            self._close_loaders()
            self._on_clear_preview()
            layers_to_remove = [l for l in self.viewer.layers if l.name.startswith("ch")]
            for l in layers_to_remove:
                self.viewer.layers.remove(l)
            self.file_table.setRowCount(0)
            self.shift_manager = ShiftManager()
            self._rebuild_shift_table()
            self.lbl_dir.clear()

        self._input_format = new_format
        # Update format-dependent labels.
        if new_format == "h5":
            self.lbl_output_format.setText("Output format: Luxendo H5 (.lux.h5)")
        else:
            self.lbl_output_format.setText("Output format: BigTIFF")

    # ------------------------------------------------------------------ #
    # Callbacks — Data Loading
    # ------------------------------------------------------------------ #

    def _on_select_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if not path:
            return
        self.lbl_dir.setText(path)
        dir_path = Path(path)

        if self._input_format == "h5":
            files = scan_h5_files(dir_path)
            self._populate_file_table(files)
            self.lbl_voxel_source.setVisible(False)
            # Try auto-filling voxel sizes from the first H5 file's metadata.
            if files:
                self._try_autofill_h5_voxel(files[0])
        else:
            files = scan_bigtiff_files(dir_path)
            self._populate_file_table(files)
            self.lbl_voxel_source.setVisible(False)
            # Auto-fill voxel sizes from XML metadata if available.
            voxel = parse_voxel_size_from_xml(dir_path)
            if voxel is not None:
                self.spin_voxel_xy.setValue(voxel[0])
                self.spin_voxel_z.setValue(voxel[1])

    def _try_autofill_h5_voxel(self, h5_path: Path) -> None:
        """Try to auto-fill voxel sizes from H5 metadata."""
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

        # Load files — format-specific.
        loaders: list[Any] = []
        filenames: list[str] = []
        colormaps: list[str] = []
        try:
            if self._input_format == "h5":
                self._h5_file_manager = H5FileManager()
                for _, path, cmap in selected:
                    loaders.append(H5Loader(path, self._h5_file_manager))
                    filenames.append(path.name)
                    colormaps.append(cmap)
                # Auto-fill voxel sizes from reference channel H5 metadata.
                ref_loader_idx = 0
                for idx, (_, path, _) in enumerate(selected):
                    for r in range(self.file_table.rowCount()):
                        item = self.file_table.item(r, 1)
                        if item and Path(item.data(Qt.UserRole)) == path:
                            if r == ref_row:
                                ref_loader_idx = idx
                            break
                ref_meta = loaders[ref_loader_idx].h5_metadata
                xy = ref_meta.get("voxel_size_xy_um")
                z = ref_meta.get("voxel_size_z_um")
                if xy and xy > 0:
                    self.spin_voxel_xy.setValue(xy)
                if z and z > 0:
                    self.spin_voxel_z.setValue(z)
                if xy or z:
                    self.lbl_voxel_source.setText("from H5 metadata")
                    self.lbl_voxel_source.setVisible(True)
            else:
                for _, path, cmap in selected:
                    loaders.append(BigTIFFLoader(path))
                    filenames.append(path.name)
                    colormaps.append(cmap)
                self.lbl_voxel_source.setVisible(False)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        # Validate.
        ok, msg = validate_channels(loaders)
        if not ok:
            QMessageBox.critical(self, "Validation Error", msg)
            for ld in loaders:
                ld.close()
            if self._h5_file_manager:
                self._h5_file_manager.close_all()
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
            # H5 loaders with pyramid levels use multiscale for napari.
            if self._input_format == "h5" and len(loader.multiscale) > 1:
                self.viewer.add_image(
                    loader.multiscale,
                    name=f"ch{i}_{t.filename}",
                    colormap=t.colormap,
                    blending="additive",
                    visible=True,
                    multiscale=True,
                )
            else:
                self.viewer.add_image(
                    loader.dask_array,
                    name=f"ch{i}_{t.filename}",
                    colormap=t.colormap,
                    blending="additive",
                    visible=True,
                )

        # Clear old confidence scores and update UI.
        self._confidence_scores.clear()
        self._rebuild_shift_table()
        self._rebuild_registration_channels()
        self._update_run_button_state()

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
        # Close H5 file handles if any.
        if self._h5_file_manager is not None:
            self._h5_file_manager.close_all()
            self._h5_file_manager = None

    # ------------------------------------------------------------------ #
    # Callbacks — Shift Table
    # ------------------------------------------------------------------ #

    def _rebuild_shift_table(self) -> None:
        n = len(self.shift_manager)
        self.shift_table.setRowCount(n)
        has_confidence = bool(self._confidence_scores)

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

            # Confidence column.
            if t.is_reference:
                conf_item = QTableWidgetItem("\u2014")
                conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)
                conf_item.setTextAlignment(Qt.AlignCenter)
                self.shift_table.setItem(i, 5, conf_item)
            elif i in self._confidence_scores:
                conf = self._confidence_scores[i]
                r, g, b = confidence_color_rgb(conf)
                conf_item = QTableWidgetItem(f"{conf:.2f}")
                conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)
                conf_item.setTextAlignment(Qt.AlignCenter)
                conf_item.setBackground(QColor(r, g, b, 80))
                conf_item.setForeground(QColor(r, g, b))
                self.shift_table.setItem(i, 5, conf_item)
            else:
                conf_item = QTableWidgetItem("\u2014")
                conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)
                conf_item.setTextAlignment(Qt.AlignCenter)
                self.shift_table.setItem(i, 5, conf_item)

        # Show confidence guidance if we have scores.
        self.lbl_confidence_guide.setVisible(has_confidence)

    def _make_shift_callback(self, channel_idx: int, axis: str):
        def _cb(value: int) -> None:
            self.shift_manager.set_shift(channel_idx, axis, value)
        return _cb

    def _on_reset_shifts(self) -> None:
        self.shift_manager.reset_all()
        self._confidence_scores.clear()
        self._rebuild_shift_table()

    # ------------------------------------------------------------------ #
    # Callbacks — Auto-Registration
    # ------------------------------------------------------------------ #

    def _on_run_registration(self) -> None:
        """Validate inputs and launch the registration worker thread."""
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
        z_end = self.spin_z_end.value() + 1  # inclusive → exclusive

        if z_end <= z_start:
            QMessageBox.warning(self, "Invalid Z", "Z end must be > Z start.")
            return

        # Determine which channels to register.
        channels_to_register = []
        for ch_i, chk in self._reg_channel_checkboxes:
            if chk.isChecked():
                channels_to_register.append(ch_i)

        if not channels_to_register:
            QMessageBox.warning(
                self, "No Channels", "Select at least one channel to register."
            )
            return

        # Validate sub-volume size vs search range.
        sr_xy = self.spin_sr_xy.value()
        sr_z = self.spin_sr_z.value()
        roi_nz = z_end - z_start
        roi_ny = y_end - y_start
        roi_nx = x_end - x_start

        if roi_nx < 2 * sr_xy or roi_ny < 2 * sr_xy or roi_nz < 2 * sr_z:
            QMessageBox.warning(
                self,
                "ROI Too Small",
                f"The ROI sub-volume ({roi_nx}\u00d7{roi_ny}\u00d7{roi_nz}) is smaller "
                f"than 2\u00d7 the search range ({2*sr_xy}\u00d7{2*sr_xy}\u00d7{2*sr_z}).\n\n"
                "Please increase the ROI size or decrease the search range.",
            )
            return

        # Build algorithm kwargs.
        algo_name = self.combo_algorithm.currentText()
        algo_kwargs: dict = {}
        if algo_name == "Phase Cross-Correlation":
            norm_text = self.combo_normalization.currentText()
            algo_kwargs["normalization"] = "phase" if norm_text == "Phase" else None

        ref_idx = self.shift_manager.reference_index
        if ref_idx is None:
            QMessageBox.warning(self, "No Reference", "No reference channel set.")
            return

        # Disable UI and start worker.
        self._set_ui_enabled(False)
        self.reg_progress_bar.setVisible(True)
        self.reg_progress_bar.setValue(0)
        self.lbl_reg_progress.setVisible(True)
        self.lbl_reg_progress.setText("Starting registration...")
        self.lbl_reg_result.setVisible(False)

        self._registration_worker = RegistrationWorker(
            loaders=self.loaders,
            reference_index=ref_idx,
            channels_to_register=channels_to_register,
            algorithm_name=algo_name,
            algorithm_kwargs=algo_kwargs,
            search_range_xy=sr_xy,
            search_range_z=sr_z,
            roi_bounds=bounds,
            z_start=z_start,
            z_end=z_end,
            background_subtraction=self.chk_bg_sub.isChecked(),
            gaussian_smoothing=self.chk_gaussian.isChecked(),
            use_gpu=gpu_available(),
        )
        self._registration_worker.progress.connect(self._on_reg_progress)
        self._registration_worker.finished.connect(self._on_reg_finished)
        self._registration_worker.error.connect(self._on_reg_error)
        self._registration_worker.start()

    def _on_reg_progress(self, done: int, total: int, desc: str) -> None:
        if total > 0:
            pct = int(100 * done / total)
            self.reg_progress_bar.setValue(pct)
        self.lbl_reg_progress.setText(desc)

    def _on_reg_finished(self, results: list) -> None:
        self._set_ui_enabled(True)
        self.reg_progress_bar.setVisible(False)
        self.lbl_reg_progress.setVisible(False)

        if not results:
            self.lbl_reg_result.setText("No registration results.")
            self.lbl_reg_result.setVisible(True)
            return

        # Apply results to the shift manager and store confidence.
        sr_xy = self.spin_sr_xy.value()
        sr_z = self.spin_sr_z.value()
        warnings: list[str] = []
        low_confidence_channels: list[str] = []
        confidences: list[float] = []

        for ch_i, result in results:
            self.shift_manager.set_shift(ch_i, "x", result.shift_x)
            self.shift_manager.set_shift(ch_i, "y", result.shift_y)
            self.shift_manager.set_shift(ch_i, "z", result.shift_z)
            self._confidence_scores[ch_i] = result.confidence
            confidences.append(result.confidence)

            # Check if shift hit search range limit.
            if abs(result.shift_x) >= sr_xy or abs(result.shift_y) >= sr_xy:
                warnings.append(
                    f"ch{ch_i}: XY shift hit search range limit. "
                    "Consider increasing the XY search range."
                )
            if abs(result.shift_z) >= sr_z:
                warnings.append(
                    f"ch{ch_i}: Z shift hit search range limit. "
                    "Consider increasing the Z search range."
                )

            if result.confidence < CONFIDENCE_LOW:
                t = self.shift_manager[ch_i]
                low_confidence_channels.append(f"ch{ch_i} ({t.filename})")

        # Rebuild shift table with new values and confidence.
        self._rebuild_shift_table()

        # Build result summary.
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        summary_lines = [f"Registration complete. Average confidence: {avg_conf:.2f}"]

        if low_confidence_channels:
            summary_lines.append(
                f"\u26a0 Low confidence for: {', '.join(low_confidence_channels)}. "
                "Consider manual adjustment."
            )

        for w in warnings:
            summary_lines.append(f"\u26a0 {w}")

        if all(c < CONFIDENCE_LOW for c in confidences):
            summary_lines.append(
                "\u26a0 All channels have low confidence. Try a different algorithm, "
                "enable preprocessing, or use an ROI with more structural content."
            )

        self.lbl_reg_result.setText("\n".join(summary_lines))
        self.lbl_reg_result.setVisible(True)

    def _on_reg_error(self, tb: str) -> None:
        self._set_ui_enabled(True)
        self.reg_progress_bar.setVisible(False)
        self.lbl_reg_progress.setVisible(False)
        QMessageBox.critical(self, "Registration Error", f"Registration failed:\n{tb}")

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
        # After drawing, update the run button state.
        self._shapes_layer.events.data.connect(
            lambda _: self._update_run_button_state()
        )

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
            input_format=self._input_format,
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
            self.combo_format,
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
            self.btn_run_registration,
            self.combo_algorithm,
            self.spin_sr_xy,
            self.spin_sr_z,
            self.chk_bg_sub,
            self.chk_gaussian,
        ]:
            w.setEnabled(enabled)
