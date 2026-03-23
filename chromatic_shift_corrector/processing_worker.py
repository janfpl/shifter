"""QThread workers for image processing and export."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import tifffile
from qtpy.QtCore import QThread, Signal

from chromatic_shift_corrector.processing import apply_pipeline

logger = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """Run the processing pipeline on a numpy subvolume (ROI or chunk)."""

    progress = Signal(int, int)  # (steps_done, total_steps)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(
        self,
        data: np.ndarray,
        steps: list[dict],
        use_gpu: bool = False,
    ) -> None:
        super().__init__()
        self.data = data
        self.steps = steps
        self.use_gpu = use_gpu
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            result = apply_pipeline(
                self.data,
                self.steps,
                use_gpu=self.use_gpu,
                progress_callback=lambda done, total: self.progress.emit(done, total),
                cancel_check=lambda: self._cancelled,
            )
            if not self._cancelled:
                self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class ExportProcessedWorker(QThread):
    """Process data and write result to BigTIFF."""

    progress = Signal(int, int)  # (planes_done, total_planes)
    finished = Signal(str)  # output path
    error = Signal(str)

    def __init__(
        self,
        data: np.ndarray,
        steps: list[dict],
        use_gpu: bool,
        output_path: Path,
    ) -> None:
        super().__init__()
        self.data = data
        self.steps = steps
        self.use_gpu = use_gpu
        self.output_path = Path(output_path)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            # Process
            processed = apply_pipeline(
                self.data,
                self.steps,
                use_gpu=self.use_gpu,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                self.finished.emit("")
                return

            # Write BigTIFF
            nz = processed.shape[0]
            with tifffile.TiffWriter(str(self.output_path), bigtiff=True) as tw:
                for z in range(nz):
                    if self._cancelled:
                        self.finished.emit("")
                        return
                    tw.write(processed[z], photometric="minisblack", contiguous=True)
                    self.progress.emit(z + 1, nz)

            self.finished.emit(str(self.output_path))
        except Exception:
            self.error.emit(traceback.format_exc())


def compute_chunk_size(
    xy_shape: tuple[int, int],
    ram_percent: int = 50,
    bytes_per_voxel: int = 2,
) -> int:
    """Determine how many Z-planes to process per chunk.

    We need to hold input + output slabs simultaneously.
    """
    total_ram = psutil.virtual_memory().total
    available = int(total_ram * ram_percent / 100)
    ny, nx = xy_shape
    plane_bytes = ny * nx * bytes_per_voxel
    # Input + output = 2 slabs
    bytes_per_z = 2 * plane_bytes
    return max(1, available // bytes_per_z)


class FullStackProcessingWorker(QThread):
    """Process a full dask volume in Z-chunks and accumulate the result."""

    progress = Signal(int, int)  # (planes_done, total_planes)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(
        self,
        dask_array: Any,  # dask.array.Array
        steps: list[dict],
        use_gpu: bool = False,
        ram_percent: int = 50,
    ) -> None:
        super().__init__()
        self.dask_array = dask_array
        self.steps = steps
        self.use_gpu = use_gpu
        self.ram_percent = ram_percent
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            nz, ny, nx = self.dask_array.shape
            chunk_z = compute_chunk_size(
                (ny, nx), self.ram_percent, self.dask_array.dtype.itemsize,
            )
            # Allocate output
            result = np.empty((nz, ny, nx), dtype=self.dask_array.dtype)
            planes_done = 0

            for z_start in range(0, nz, chunk_z):
                if self._cancelled:
                    return
                z_end = min(z_start + chunk_z, nz)
                chunk = np.asarray(self.dask_array[z_start:z_end])

                processed = apply_pipeline(
                    chunk,
                    self.steps,
                    use_gpu=self.use_gpu,
                    cancel_check=lambda: self._cancelled,
                )
                if self._cancelled:
                    return

                result[z_start:z_end] = processed
                planes_done += z_end - z_start
                self.progress.emit(planes_done, nz)

            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())
