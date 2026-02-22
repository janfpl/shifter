"""Chunked full-volume processing and BigTIFF / Luxendo H5 export.

XY-shift application within each slab is parallelised across CPU cores
using ``concurrent.futures.ThreadPoolExecutor``.  Performance timestamps
are emitted via :mod:`chromatic_shift_corrector.perf_logger`.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil
import tifffile

from chromatic_shift_corrector.perf_logger import log_event, timed_operation
from chromatic_shift_corrector.utils import (
    apply_integer_shift_2d,
    build_metadata,
    save_metadata,
)

if TYPE_CHECKING:
    import dask.array as da
    from chromatic_shift_corrector.shift_manager import ChannelTransform, ShiftManager

logger = logging.getLogger(__name__)

# Number of threads for per-plane XY shift (capped to avoid over-subscription).
_XY_WORKERS = min(os.cpu_count() or 1, 16)


def compute_chunk_size(
    xy_shape: tuple[int, int],
    n_channels: int,
    ram_percent: int = 90,
    bytes_per_voxel: int = 2,
) -> int:
    """Determine how many Z-planes to process per chunk.

    We need to hold *n_channels* input slabs plus *n_channels* output slabs
    simultaneously. Each slab has shape (chunk_z, Y, X) at *bytes_per_voxel*.

    Parameters
    ----------
    xy_shape : (int, int)
        (Y, X) dimensions.
    n_channels : int
        Number of channels.
    ram_percent : int
        Percentage of system RAM to use (50-95).
    bytes_per_voxel : int
        Bytes per voxel (2 for uint16).

    Returns
    -------
    int
        Number of Z-planes per processing chunk (>= 1).
    """
    total_ram = psutil.virtual_memory().total
    available = int(total_ram * ram_percent / 100)

    ny, nx = xy_shape
    plane_bytes = ny * nx * bytes_per_voxel

    # Each chunk we need: n_channels * chunk_z planes for reading
    # + n_channels * chunk_z planes for writing = 2 * n_channels * chunk_z
    bytes_per_z = 2 * n_channels * plane_bytes

    chunk_z = max(1, available // bytes_per_z)
    return chunk_z


def estimate_output_sizes(
    loaders: list[Any],
    bytes_per_voxel: int = 2,
    roi: tuple[int, int, int, int, int, int] | None = None,
) -> list[int]:
    """Estimate output file sizes in bytes.

    Parameters
    ----------
    roi : tuple, optional
        (z_start, z_end, y_start, y_end, x_start, x_end) crop region.
        If provided, sizes are based on the ROI dimensions.
    """
    sizes = []
    if roi is not None:
        rz_s, rz_e, ry_s, ry_e, rx_s, rx_e = roi
        roi_nz = rz_e - rz_s
        roi_ny = ry_e - ry_s
        roi_nx = rx_e - rx_s
        for _ in loaders:
            sizes.append(roi_nz * roi_ny * roi_nx * bytes_per_voxel)
    else:
        for loader in loaders:
            nz, ny, nx = loader.shape
            sizes.append(nz * ny * nx * bytes_per_voxel)
    return sizes


def _shift_slab_xy(slab: np.ndarray, sy: int, sx: int) -> np.ndarray:
    """Apply XY shift to every plane in *slab*, parallelised across cores."""
    if sy == 0 and sx == 0:
        return slab

    n_planes = slab.shape[0]

    def _shift_plane(i: int) -> None:
        slab[i] = apply_integer_shift_2d(slab[i], (sy, sx))

    if n_planes <= 2:
        for i in range(n_planes):
            _shift_plane(i)
    else:
        with ThreadPoolExecutor(max_workers=min(_XY_WORKERS, n_planes)) as pool:
            list(pool.map(_shift_plane, range(n_planes)))

    return slab


def _read_roi_slab(
    dask_arr: da.Array,
    roi: tuple[int, int, int, int, int, int],
    out_z_start: int,
    out_z_end: int,
    sz: int,
    sy: int,
    sx: int,
) -> np.ndarray:
    """Read a Z-slab from *dask_arr* for an ROI export with shifts applied.

    Instead of reading full XY planes and cropping, this reads only the input
    sub-region that maps to the ROI output after the shift is applied, and
    places it at the correct offset in a zero-initialized output slab.

    Parameters
    ----------
    dask_arr : dask.array.Array
        Source array (Z, Y, X).
    roi : tuple
        (z_start, z_end, y_start, y_end, x_start, x_end) in full-volume coords.
    out_z_start, out_z_end : int
        Output slab Z range (in ROI-local coords, i.e. 0-based).
    sz, sy, sx : int
        Shift in Z, Y, X voxels.

    Returns
    -------
    np.ndarray
        Slab of shape (n_planes, roi_ny, roi_nx), dtype uint16.
    """
    nz, ny, nx = dask_arr.shape
    rz_s, rz_e, ry_s, ry_e, rx_s, rx_e = roi
    roi_ny = ry_e - ry_s
    roi_nx = rx_e - rx_s
    n_planes = out_z_end - out_z_start

    # Map output Z (ROI-local) back to full-volume Z, then to input Z.
    full_z_start = rz_s + out_z_start
    full_z_end = rz_s + out_z_end
    in_z_start = full_z_start - sz
    in_z_end = full_z_end - sz

    # Input XY region needed (accounting for shift).
    in_y_start = ry_s - sy
    in_y_end = ry_e - sy
    in_x_start = rx_s - sx
    in_x_end = rx_e - sx

    # Clamp Z to valid input range.
    read_z_start = max(in_z_start, 0)
    read_z_end = min(in_z_end, nz)

    # Clamp XY to valid input range.
    read_y_start = max(in_y_start, 0)
    read_y_end = min(in_y_end, ny)
    read_x_start = max(in_x_start, 0)
    read_x_end = min(in_x_end, nx)

    slab = np.zeros((n_planes, roi_ny, roi_nx), dtype=np.uint16)

    if read_z_start >= read_z_end or read_y_start >= read_y_end or read_x_start >= read_x_end:
        return slab

    raw = np.asarray(
        dask_arr[read_z_start:read_z_end, read_y_start:read_y_end, read_x_start:read_x_end]
    )

    # Destination offsets in the output slab.
    dst_z = read_z_start - in_z_start
    dst_y = read_y_start - in_y_start
    dst_x = read_x_start - in_x_start

    slab[
        dst_z : dst_z + raw.shape[0],
        dst_y : dst_y + raw.shape[1],
        dst_x : dst_x + raw.shape[2],
    ] = raw

    return slab


def _export_channel_roi_tiff(
    dask_arr: da.Array,
    transform: ChannelTransform,
    output_path: Path,
    chunk_z: int,
    roi: tuple[int, int, int, int, int, int],
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
) -> None:
    """Export an ROI-cropped corrected channel to a BigTIFF file."""
    rz_s, rz_e, ry_s, ry_e, rx_s, rx_e = roi
    out_nz = rz_e - rz_s
    sz, sy, sx = transform.shift_z, transform.shift_y, transform.shift_x

    log_event(f"Export TIFF ROI channel: {output_path.name} | "
              f"roi_shape=({out_nz},{ry_e - ry_s},{rx_e - rx_s}) "
              f"shift=({sz},{sy},{sx}) chunk_z={chunk_z}")

    with timed_operation(f"Write TIFF ROI: {output_path.name}"):
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as tw:
            planes_done = 0
            for slab_start in range(0, out_nz, chunk_z):
                if cancel_check and cancel_check():
                    return

                slab_end = min(slab_start + chunk_z, out_nz)
                slab = _read_roi_slab(
                    dask_arr, roi, slab_start, slab_end, sz, sy, sx,
                )

                for i in range(slab.shape[0]):
                    tw.write(slab[i], photometric="minisblack", contiguous=True)

                planes_done += slab.shape[0]
                if progress_callback:
                    progress_callback(planes_done, out_nz)


def export_channel(
    dask_arr: da.Array,
    transform: ChannelTransform,
    output_path: Path,
    chunk_z: int,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
    roi: tuple[int, int, int, int, int, int] | None = None,
) -> None:
    """Export a single corrected channel to a BigTIFF file.

    Processing is done slab-by-slab along Z. For Z-shifts the read window is
    offset; for XY shifts each plane is shifted via slicing with zero-fill.

    Parameters
    ----------
    dask_arr : dask.array.Array
        Source lazy array of shape (Z, Y, X), dtype uint16.
    transform : ChannelTransform
        Shift parameters.
    output_path : Path
        Destination BigTIFF path.
    chunk_z : int
        Number of Z-planes per processing chunk.
    progress_callback : callable, optional
        Called with (planes_completed, total_planes) after each chunk.
    cancel_check : callable, optional
        Called before each chunk; if it returns True, export is aborted.
    roi : tuple, optional
        (z_start, z_end, y_start, y_end, x_start, x_end) in full-volume
        coordinates. When provided, only the ROI region is exported.
    """
    nz, ny, nx = dask_arr.shape
    sz, sy, sx = transform.shift_z, transform.shift_y, transform.shift_x

    if roi is not None:
        _export_channel_roi_tiff(
            dask_arr, transform, output_path, chunk_z,
            roi, progress_callback, cancel_check,
        )
        return

    log_event(f"Export TIFF channel: {output_path.name} | "
              f"shape=({nz},{ny},{nx}) shift=({sz},{sy},{sx}) chunk_z={chunk_z}")

    with timed_operation(f"Write TIFF: {output_path.name}"):
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as tw:
            planes_done = 0
            for out_z_start in range(0, nz, chunk_z):
                if cancel_check and cancel_check():
                    return

                out_z_end = min(out_z_start + chunk_z, nz)
                n_planes = out_z_end - out_z_start

                # Determine which input Z-planes we need.
                in_z_start = out_z_start - sz
                in_z_end = out_z_end - sz

                # Clamp to valid input range and figure out padding.
                read_start = max(in_z_start, 0)
                read_end = min(in_z_end, nz)

                if read_start >= read_end:
                    slab = np.zeros((n_planes, ny, nx), dtype=np.uint16)
                else:
                    raw = np.asarray(dask_arr[read_start:read_end])
                    slab = np.zeros((n_planes, ny, nx), dtype=np.uint16)
                    dst_start = read_start - in_z_start
                    dst_end = dst_start + (read_end - read_start)
                    slab[dst_start:dst_end] = raw

                # Apply XY shifts (parallel across planes).
                slab = _shift_slab_xy(slab, sy, sx)

                # Write planes.
                for i in range(n_planes):
                    tw.write(
                        slab[i],
                        photometric="minisblack",
                        contiguous=True,
                    )

                planes_done += n_planes
                if progress_callback:
                    progress_callback(planes_done, nz)


def run_export(
    loaders: list[Any],
    shift_manager: ShiftManager,
    output_dir: Path,
    ram_percent: int = 90,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
    voxel_xy: float = 1.0,
    voxel_z: float = 1.0,
    roi: tuple[int, int, int, int, int, int] | None = None,
) -> Path:
    """Export all channels with corrections applied.

    Parameters
    ----------
    loaders : list
        One loader per channel, in channel order.
    shift_manager : ShiftManager
        Contains all channel transforms.
    output_dir : Path
        Directory to write corrected files into.
    ram_percent : int
        Percentage of system RAM to allocate.
    progress_callback : callable, optional
        Called with (current_step, total_steps) where total_steps accounts for
        all planes across all channels.
    cancel_check : callable, optional
        Return True to abort.
    voxel_xy, voxel_z : float
        Voxel sizes for metadata.
    roi : tuple, optional
        (z_start, z_end, y_start, y_end, x_start, x_end) crop region.

    Returns
    -------
    Path
        Path to the written metadata JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = len(loaders)
    ref_shape = loaders[0].shape

    if roi is not None:
        rz_s, rz_e, ry_s, ry_e, rx_s, rx_e = roi
        roi_ny, roi_nx = ry_e - ry_s, rx_e - rx_s
        xy_shape = (roi_ny, roi_nx)
        planes_per_channel = rz_e - rz_s
        suffix = "_corrected_roi"
    else:
        xy_shape = (ref_shape[1], ref_shape[2])
        planes_per_channel = None  # varies per loader
        suffix = "_corrected"

    chunk_z = compute_chunk_size(xy_shape, n_channels, ram_percent)

    # Total planes across all channels for progress reporting.
    if roi is not None:
        total_planes = planes_per_channel * n_channels
    else:
        total_planes = sum(loader.shape[0] for loader in loaders)
    global_done = 0

    def _channel_progress(done: int, _total: int) -> None:
        nonlocal global_done
        if progress_callback:
            progress_callback(global_done + done, total_planes)

    for i, (loader, transform) in enumerate(
        zip(loaders, shift_manager.transforms)
    ):
        stem = transform.filename.rsplit(".", 1)[0]
        ext = transform.filename.rsplit(".", 1)[1] if "." in transform.filename else "tif"
        out_path = output_dir / f"{stem}{suffix}.{ext}"

        export_channel(
            loader.dask_array,
            transform,
            out_path,
            chunk_z,
            progress_callback=_channel_progress,
            cancel_check=cancel_check,
            roi=roi,
        )
        if roi is not None:
            global_done += planes_per_channel
        else:
            global_done += loader.shape[0]

    # Write metadata.
    channel_dicts = shift_manager.to_channel_dicts(output_suffix=suffix)
    ref_idx = shift_manager.reference_index or 0
    vol_shape = ref_shape
    if roi is not None:
        vol_shape = (rz_e - rz_s, ry_e - ry_s, rx_e - rx_s)
    metadata = build_metadata(
        channel_dicts, ref_idx, voxel_xy, voxel_z, vol_shape, ram_percent,
        roi_bounds=roi,
    )
    meta_path = save_metadata(metadata, output_dir)
    return meta_path


# ===================================================================== #
# Luxendo H5 export
# ===================================================================== #


def export_channel_h5(
    dask_arr: da.Array,
    transform: ChannelTransform,
    output_path: Path,
    original_h5: Any,
    chunk_z: int,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
    roi: tuple[int, int, int, int, int, int] | None = None,
) -> None:
    """Export a single corrected channel to a Luxendo .lux.h5 file.

    Writes the shift-corrected ``Data`` dataset, copies the original
    ``metadata`` dataset verbatim, then regenerates all pyramid levels
    that existed in the original file.

    Parameters
    ----------
    dask_arr : dask.array.Array
        Source lazy array (full-res) of shape (Z, Y, X), dtype uint16.
    transform : ChannelTransform
        Shift parameters.
    output_path : Path
        Destination .lux.h5 file path.
    original_h5 : h5py.File
        Open handle to the original H5 file (for metadata and pyramid info).
    chunk_z : int
        Number of Z-planes per processing chunk.
    progress_callback : callable, optional
        Called with (planes_completed, total_planes) after each chunk.
    cancel_check : callable, optional
        Called before each chunk; if True, abort.
    roi : tuple, optional
        (z_start, z_end, y_start, y_end, x_start, x_end) crop region.
    """
    import h5py

    from chromatic_shift_corrector.h5_utils import (
        detect_pyramid_levels,
        generate_pyramid_level,
    )

    nz, ny, nx = dask_arr.shape
    sz, sy, sx = transform.shift_z, transform.shift_y, transform.shift_x

    if roi is not None:
        rz_s, rz_e, ry_s, ry_e, rx_s, rx_e = roi
        out_nz = rz_e - rz_s
        out_ny = ry_e - ry_s
        out_nx = rx_e - rx_s
    else:
        out_nz, out_ny, out_nx = nz, ny, nx

    log_event(f"Export H5 channel: {output_path.name} | "
              f"shape=({out_nz},{out_ny},{out_nx}) shift=({sz},{sy},{sx}) "
              f"chunk_z={chunk_z} roi={roi is not None}")

    # Determine chunking from original Data dataset.
    orig_data = original_h5["Data"]
    orig_chunks = orig_data.chunks if orig_data.chunks else (64, 64, 64)
    # Clamp chunk sizes to output dimensions for ROI exports.
    h5_chunks = tuple(min(c, d) for c, d in zip(orig_chunks, (out_nz, out_ny, out_nx)))

    with timed_operation(f"Write H5: {output_path.name}"):
        with h5py.File(str(output_path), "w") as out_h5:
            # Create output Data dataset.
            ds = out_h5.create_dataset(
                "Data",
                shape=(out_nz, out_ny, out_nx),
                dtype=np.uint16,
                chunks=h5_chunks,
            )

            if roi is not None:
                # ROI export: read only the needed sub-region.
                planes_done = 0
                for slab_start in range(0, out_nz, chunk_z):
                    if cancel_check and cancel_check():
                        return

                    slab_end = min(slab_start + chunk_z, out_nz)
                    slab = _read_roi_slab(
                        dask_arr, roi, slab_start, slab_end, sz, sy, sx,
                    )
                    ds[slab_start:slab_end] = slab

                    planes_done += slab.shape[0]
                    if progress_callback:
                        progress_callback(planes_done, out_nz)
            else:
                # Full-volume export.
                planes_done = 0
                for out_z_start in range(0, nz, chunk_z):
                    if cancel_check and cancel_check():
                        return

                    out_z_end = min(out_z_start + chunk_z, nz)
                    n_planes = out_z_end - out_z_start

                    in_z_start = out_z_start - sz
                    in_z_end = out_z_end - sz

                    read_start = max(in_z_start, 0)
                    read_end = min(in_z_end, nz)

                    if read_start >= read_end:
                        slab = np.zeros((n_planes, ny, nx), dtype=np.uint16)
                    else:
                        raw = np.asarray(dask_arr[read_start:read_end])
                        slab = np.zeros((n_planes, ny, nx), dtype=np.uint16)
                        dst_start = read_start - in_z_start
                        dst_end = dst_start + (read_end - read_start)
                        slab[dst_start:dst_end] = raw

                    # Apply XY shifts (parallel across planes).
                    slab = _shift_slab_xy(slab, sy, sx)

                    ds[out_z_start:out_z_end] = slab

                    planes_done += n_planes
                    if progress_callback:
                        progress_callback(planes_done, nz)

            # Copy metadata verbatim.
            if "metadata" in original_h5:
                raw_meta = original_h5["metadata"][()]
                out_h5.create_dataset("metadata", data=raw_meta)

            # Regenerate pyramid levels.
            pyramid_levels = detect_pyramid_levels(original_h5)
            for level_name, fw, fh, fd in pyramid_levels:
                if cancel_check and cancel_check():
                    return

                # Match original chunking for this pyramid level.
                orig_pyr = original_h5[level_name]
                pyr_chunks = orig_pyr.chunks if orig_pyr.chunks else None

                logger.info("Regenerating pyramid level %s (factors %d\u00d7%d\u00d7%d)", level_name, fw, fh, fd)
                generate_pyramid_level(out_h5, level_name, fw, fh, fd, chunks=pyr_chunks)


def run_export_h5(
    loaders: list[Any],
    shift_manager: ShiftManager,
    output_dir: Path,
    ram_percent: int = 90,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
    voxel_xy: float = 1.0,
    voxel_z: float = 1.0,
    roi: tuple[int, int, int, int, int, int] | None = None,
) -> Path:
    """Export all H5 channels with corrections applied.

    Parameters
    ----------
    loaders : list[H5Loader]
        One loader per channel, in channel order.
    shift_manager : ShiftManager
        Contains all channel transforms.
    output_dir : Path
        Directory to write corrected files into.
    ram_percent : int
        Percentage of system RAM to allocate.
    progress_callback : callable, optional
        Called with (current_step, total_steps).
    cancel_check : callable, optional
        Return True to abort.
    voxel_xy, voxel_z : float
        Voxel sizes for metadata.
    roi : tuple, optional
        (z_start, z_end, y_start, y_end, x_start, x_end) crop region.

    Returns
    -------
    Path
        Path to the written metadata JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = len(loaders)
    ref_shape = loaders[0].shape

    if roi is not None:
        rz_s, rz_e, ry_s, ry_e, rx_s, rx_e = roi
        roi_ny, roi_nx = ry_e - ry_s, rx_e - rx_s
        xy_shape = (roi_ny, roi_nx)
        planes_per_channel = rz_e - rz_s
        suffix = "_corrected_roi"
    else:
        xy_shape = (ref_shape[1], ref_shape[2])
        planes_per_channel = None
        suffix = "_corrected"

    chunk_z = compute_chunk_size(xy_shape, n_channels, ram_percent)

    if roi is not None:
        total_planes = planes_per_channel * n_channels
    else:
        total_planes = sum(loader.shape[0] for loader in loaders)
    global_done = 0

    def _channel_progress(done: int, _total: int) -> None:
        nonlocal global_done
        if progress_callback:
            progress_callback(global_done + done, total_planes)

    for i, (loader, transform) in enumerate(
        zip(loaders, shift_manager.transforms)
    ):
        # Output filename: {stem}{suffix}.lux.h5 or {stem}{suffix}.h5.
        name = transform.filename
        if name.lower().endswith(".lux.h5"):
            stem = name[: -len(".lux.h5")]
            out_name = f"{stem}{suffix}.lux.h5"
        elif name.lower().endswith(".h5"):
            stem = name[: -len(".h5")]
            out_name = f"{stem}{suffix}.h5"
        else:
            out_name = f"{name}{suffix}.lux.h5"
        out_path = output_dir / out_name

        export_channel_h5(
            loader.dask_array,
            transform,
            out_path,
            loader._h5,
            chunk_z,
            progress_callback=_channel_progress,
            cancel_check=cancel_check,
            roi=roi,
        )
        if roi is not None:
            global_done += planes_per_channel
        else:
            global_done += loader.shape[0]

    # Build metadata with H5-specific fields.
    channel_dicts = shift_manager.to_channel_dicts(output_suffix=suffix)
    ref_idx = shift_manager.reference_index or 0

    # Augment channel dicts with H5-specific info.
    for i, cd in enumerate(channel_dicts):
        loader = loaders[i]
        # Fix corrected filename to match H5 naming.
        name = cd["filename_original"]
        if name.lower().endswith(".lux.h5"):
            stem = name[: -len(".lux.h5")]
            cd["filename_corrected"] = f"{stem}{suffix}.lux.h5"
        elif name.lower().endswith(".h5"):
            stem = name[: -len(".h5")]
            cd["filename_corrected"] = f"{stem}{suffix}.h5"

        # Add channel_description and pyramid info.
        cd["channel_description"] = loader.channel_description
        cd["pyramid_levels_regenerated"] = [
            lvl[0] for lvl in loader.pyramid_levels
        ]

    vol_shape = ref_shape
    if roi is not None:
        vol_shape = (rz_e - rz_s, ry_e - ry_s, rx_e - rx_s)
    metadata = build_metadata(
        channel_dicts, ref_idx, voxel_xy, voxel_z, vol_shape, ram_percent,
        roi_bounds=roi,
    )
    metadata["input_format"] = "luxendo_h5"
    metadata["voxel_size_source"] = "h5_metadata"

    meta_path = save_metadata(metadata, output_dir)
    return meta_path
