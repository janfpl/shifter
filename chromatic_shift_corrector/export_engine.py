"""Chunked full-volume processing and BigTIFF / Luxendo H5 export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil
import tifffile

from chromatic_shift_corrector.utils import (
    apply_integer_shift_2d,
    build_metadata,
    save_metadata,
)

if TYPE_CHECKING:
    import dask.array as da
    from chromatic_shift_corrector.shift_manager import ChannelTransform, ShiftManager

logger = logging.getLogger(__name__)


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
) -> list[int]:
    """Estimate output file sizes in bytes (same shape as input)."""
    sizes = []
    for loader in loaders:
        nz, ny, nx = loader.shape
        sizes.append(nz * ny * nx * bytes_per_voxel)
    return sizes


def export_channel(
    dask_arr: da.Array,
    transform: ChannelTransform,
    output_path: Path,
    chunk_z: int,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
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
    """
    nz, ny, nx = dask_arr.shape
    sz, sy, sx = transform.shift_z, transform.shift_y, transform.shift_x

    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tw:
        planes_done = 0
        for out_z_start in range(0, nz, chunk_z):
            if cancel_check and cancel_check():
                return

            out_z_end = min(out_z_start + chunk_z, nz)
            n_planes = out_z_end - out_z_start

            # Determine which input Z-planes we need.
            # Output plane out_z was originally at input plane out_z - sz.
            in_z_start = out_z_start - sz
            in_z_end = out_z_end - sz

            # Clamp to valid input range and figure out padding.
            read_start = max(in_z_start, 0)
            read_end = min(in_z_end, nz)

            if read_start >= read_end:
                # Entire chunk is out of bounds → write zeros.
                slab = np.zeros((n_planes, ny, nx), dtype=np.uint16)
            else:
                raw = np.asarray(dask_arr[read_start:read_end])

                # Build output slab with potential Z-padding.
                slab = np.zeros((n_planes, ny, nx), dtype=np.uint16)
                # Where in the output slab does the read data land?
                dst_start = read_start - in_z_start  # offset within slab
                dst_end = dst_start + (read_end - read_start)
                slab[dst_start:dst_end] = raw

            # Apply XY shifts plane-by-plane.
            if sy != 0 or sx != 0:
                for i in range(n_planes):
                    slab[i] = apply_integer_shift_2d(slab[i], (sy, sx))

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

    Returns
    -------
    Path
        Path to the written metadata JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = len(loaders)
    ref_shape = loaders[0].shape
    xy_shape = (ref_shape[1], ref_shape[2])

    chunk_z = compute_chunk_size(xy_shape, n_channels, ram_percent)

    # Total planes across all channels for progress reporting.
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
        out_path = output_dir / f"{stem}_corrected.{ext}"

        export_channel(
            loader.dask_array,
            transform,
            out_path,
            chunk_z,
            progress_callback=_channel_progress,
            cancel_check=cancel_check,
        )
        global_done += loader.shape[0]

    # Write metadata.
    channel_dicts = shift_manager.to_channel_dicts()
    ref_idx = shift_manager.reference_index or 0
    metadata = build_metadata(
        channel_dicts, ref_idx, voxel_xy, voxel_z, ref_shape, ram_percent
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
    """
    import h5py

    from chromatic_shift_corrector.h5_utils import (
        detect_pyramid_levels,
        generate_pyramid_level,
    )

    nz, ny, nx = dask_arr.shape
    sz, sy, sx = transform.shift_z, transform.shift_y, transform.shift_x

    # Determine chunking from original Data dataset.
    orig_data = original_h5["Data"]
    orig_chunks = orig_data.chunks if orig_data.chunks else (64, 64, 64)

    with h5py.File(str(output_path), "w") as out_h5:
        # Create output Data dataset.
        ds = out_h5.create_dataset(
            "Data",
            shape=(nz, ny, nx),
            dtype=np.uint16,
            chunks=orig_chunks,
        )

        # Write corrected full-resolution data slab-by-slab.
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

            if sy != 0 or sx != 0:
                for i in range(n_planes):
                    slab[i] = apply_integer_shift_2d(slab[i], (sy, sx))

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

            logger.info("Regenerating pyramid level %s (factors %d×%d×%d)", level_name, fw, fh, fd)
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

    Returns
    -------
    Path
        Path to the written metadata JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = len(loaders)
    ref_shape = loaders[0].shape
    xy_shape = (ref_shape[1], ref_shape[2])

    chunk_z = compute_chunk_size(xy_shape, n_channels, ram_percent)

    total_planes = sum(loader.shape[0] for loader in loaders)
    global_done = 0

    def _channel_progress(done: int, _total: int) -> None:
        nonlocal global_done
        if progress_callback:
            progress_callback(global_done + done, total_planes)

    for i, (loader, transform) in enumerate(
        zip(loaders, shift_manager.transforms)
    ):
        # Output filename: {stem}_corrected.lux.h5 or {stem}_corrected.h5.
        name = transform.filename
        if name.lower().endswith(".lux.h5"):
            stem = name[: -len(".lux.h5")]
            out_name = f"{stem}_corrected.lux.h5"
        elif name.lower().endswith(".h5"):
            stem = name[: -len(".h5")]
            out_name = f"{stem}_corrected.h5"
        else:
            out_name = f"{name}_corrected.lux.h5"
        out_path = output_dir / out_name

        export_channel_h5(
            loader.dask_array,
            transform,
            out_path,
            loader._h5,
            chunk_z,
            progress_callback=_channel_progress,
            cancel_check=cancel_check,
        )
        global_done += loader.shape[0]

    # Build metadata with H5-specific fields.
    channel_dicts = shift_manager.to_channel_dicts(output_suffix="_corrected")
    ref_idx = shift_manager.reference_index or 0

    # Augment channel dicts with H5-specific info.
    for i, cd in enumerate(channel_dicts):
        loader = loaders[i]
        # Fix corrected filename to match H5 naming.
        name = cd["filename_original"]
        if name.lower().endswith(".lux.h5"):
            stem = name[: -len(".lux.h5")]
            cd["filename_corrected"] = f"{stem}_corrected.lux.h5"
        elif name.lower().endswith(".h5"):
            stem = name[: -len(".h5")]
            cd["filename_corrected"] = f"{stem}_corrected.h5"

        # Add channel_description and pyramid info.
        cd["channel_description"] = loader.channel_description
        cd["pyramid_levels_regenerated"] = [
            lvl[0] for lvl in loader.pyramid_levels
        ]

    metadata = build_metadata(
        channel_dicts, ref_idx, voxel_xy, voxel_z, ref_shape, ram_percent
    )
    metadata["input_format"] = "luxendo_h5"
    metadata["voxel_size_source"] = "h5_metadata"

    meta_path = save_metadata(metadata, output_dir)
    return meta_path
