"""Chunked full-volume processing and BigTIFF / Luxendo H5 export.

XY-shift application within each slab is parallelised across CPU cores
using ``concurrent.futures.ThreadPoolExecutor``.  Performance timestamps
are emitted via :mod:`shifter.perf_logger`.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil
import tifffile

from shifter.perf_logger import (
    is_debug_enabled,
    log_debug,
    log_event,
    log_memory,
    memory_status,
    timed_operation,
)
from shifter.utils import (
    apply_integer_shift_2d,
    build_metadata,
    h5_output_filename,
    save_metadata,
    worker_count,
)

if TYPE_CHECKING:
    import dask.array as da
    from shifter.shift_manager import ChannelTransform, ShiftManager

logger = logging.getLogger(__name__)

# Threads for the per-plane XY shift. Uses the shared worker budget, which
# leaves a few cores free for the OS and the napari UI (see utils.worker_count).
_XY_WORKERS = worker_count()


# --- Chunk-size policy ----------------------------------------------------- #
# Materialising one export slab via ``np.asarray(dask_arr[read_start:read_end])``
# transiently holds several full-size copies of the slab in RAM at once: dask
# reads the source chunks, then ``concatenate3`` allocates a *fresh* output
# buffer while those inputs are still alive (~2x the slab), and the caller then
# copies that result into a zero-initialised output slab (another 1x). Budgeting
# for this many copies keeps the true peak within the RAM allowance rather than
# the ~2x the old model assumed.
_SLAB_PEAK_COPIES = 3

# Hard cap on the *output* bytes of a single slab, independent of installed RAM.
# Export throughput is bound by disk I/O and per-chunk decode, not by how many
# Z-planes are batched into one read, so there is no speed benefit to building
# enormous slabs -- only a large, fragile memory footprint. This cap is what
# prevents a high-RAM machine from sizing a slab at, e.g., 57 GiB and then
# running out of memory when the system is already loaded.
_MAX_SLAB_BYTES = 4 * 1024**3


def compute_chunk_size(
    xy_shape: tuple[int, int],
    n_channels: int = 1,
    ram_percent: int = 90,
    bytes_per_voxel: int = 2,
) -> int:
    """Determine how many Z-planes to process per slab.

    The returned ``chunk_z`` is bounded by two independent limits:

    1. **Available RAM.** At most *ram_percent* of the *currently available*
       system memory (``psutil.virtual_memory().available``, **not** ``.total``)
       may be consumed by the transient full-size copies made while a slab is
       read, concatenated by dask, and written. The previous implementation
       budgeted against total RAM, which is why an export could size a slab at
       ~57 GiB on a machine that was already >90% full and then crash with an
       ``ArrayMemoryError``.
    2. **An absolute slab-size cap** (:data:`_MAX_SLAB_BYTES`) so that even on a
       machine with hundreds of GiB free, a single slab stays modest. Larger
       slabs do not export any faster (the pipeline is disk-bound) but multiply
       peak RAM, GC pressure, and the blast radius of a bad estimate.

    Channels are exported strictly sequentially (see :func:`run_export` /
    :func:`run_export_h5`), so only one channel's slab is ever resident at a
    time; *n_channels* therefore no longer scales the budget and is retained
    only for backward compatibility with existing callers.

    Parameters
    ----------
    xy_shape : (int, int)
        (Y, X) dimensions of a single plane.
    n_channels : int
        Number of channels (unused; kept for backward compatibility).
    ram_percent : int
        Percentage of *available* system RAM to use (50-95).
    bytes_per_voxel : int
        Bytes per voxel (2 for uint16).

    Returns
    -------
    int
        Number of Z-planes per processing slab (>= 1).
    """
    available = psutil.virtual_memory().available
    budget = int(available * ram_percent / 100)

    ny, nx = xy_shape
    plane_bytes = max(1, ny * nx * bytes_per_voxel)

    # Peak RAM per Z-plane while a slab is read, concatenated, and copied.
    peak_bytes_per_z = plane_bytes * _SLAB_PEAK_COPIES
    chunk_z_by_ram = budget // peak_bytes_per_z
    chunk_z_by_cap = _MAX_SLAB_BYTES // plane_bytes

    chunk_z = max(1, int(min(chunk_z_by_ram, chunk_z_by_cap)))

    log_debug(
        "compute_chunk_size: chunk_z=%d (ram-bound=%d, cap-bound=%d) | "
        "plane=%.1fMiB slab_out=%.2fGiB est_peak=%.2fGiB | "
        "budget=%.2fGiB avail=%.2fGiB ram%%=%d n_channels=%d"
        % (
            chunk_z, chunk_z_by_ram, chunk_z_by_cap,
            plane_bytes / 1024**2,
            chunk_z * plane_bytes / 1024**3,
            chunk_z * peak_bytes_per_z / 1024**3,
            budget / 1024**3, available / 1024**3, ram_percent, n_channels,
        )
    )
    return chunk_z


def _channel_output_bytes(
    loader: Any,
    roi: tuple[int, int, int, int, int, int] | None = None,
    bytes_per_voxel: int = 2,
    include_pyramids: bool = True,
) -> int:
    """Total output bytes for one exported channel: full-res + regenerated pyramids.

    Pyramid levels are regenerated only for H5 exports and are exposed via
    ``loader.pyramid_levels``. BigTIFF loaders lack that attribute and so
    contribute full-resolution bytes only, which matches BigTIFF export (no
    pyramids are written). When *include_pyramids* is False the pyramid bytes are
    omitted (used when the pyramid-regeneration step is turned off). Sizes are
    raw/uncompressed (output datasets use no compression), so this closely tracks
    on-disk size aside from minor HDF5 chunk/metadata overhead.
    """
    from shifter.h5_utils import compute_pyramid_level_shape

    if roi is not None:
        rz_s, rz_e, ry_s, ry_e, rx_s, rx_e = roi
        out_shape: tuple[int, int, int] = (rz_e - rz_s, ry_e - ry_s, rx_e - rx_s)
    else:
        out_shape = tuple(loader.shape)

    total = out_shape[0] * out_shape[1] * out_shape[2] * bytes_per_voxel
    if include_pyramids:
        for _name, fw, fh, fd in getattr(loader, "pyramid_levels", []):
            pshape = compute_pyramid_level_shape(out_shape, fw, fh, fd)
            if all(d > 0 for d in pshape):
                total += pshape[0] * pshape[1] * pshape[2] * bytes_per_voxel
    return total


def estimate_output_sizes(
    loaders: list[Any],
    bytes_per_voxel: int = 2,
    roi: tuple[int, int, int, int, int, int] | None = None,
    include_pyramids: bool = True,
) -> list[int]:
    """Estimate per-channel output file sizes in bytes.

    Includes the regenerated resolution-pyramid levels for H5 exports (unless
    *include_pyramids* is False). The previous full-resolution-only estimate
    omitted them and under-reported the total by the pyramid sum (typically
    ~12-16% for 2x2x2 + 3x3x3 factors); this matches the post-export accounting
    behind ``bytes_written_gb``.

    Parameters
    ----------
    roi : tuple, optional
        (z_start, z_end, y_start, y_end, x_start, x_end) crop region.
        If provided, sizes are based on the ROI dimensions.
    include_pyramids : bool
        Count regenerated pyramid levels (True) or full-res only (False).
    """
    return [
        _channel_output_bytes(loader, roi, bytes_per_voxel, include_pyramids)
        for loader in loaders
    ]


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


def _read_full_slab(
    dask_arr: da.Array,
    out_z_start: int,
    out_z_end: int,
    sz: int,
) -> np.ndarray:
    """Read a full-XY Z-slab for a full-volume export, with the Z-shift applied.

    Returns a writable ``(n_planes, ny, nx)`` uint16 array whose planes are the
    source planes offset by *sz* along Z and zero-padded where the shifted read
    window falls outside the source volume.

    When the shifted window lies fully inside the volume -- the common interior
    case, i.e. every slab except the one or two at the shifted Z-boundary -- the
    materialised dask result is returned directly. This avoids allocating a
    second full-size slab and copying into it, which previously doubled the peak
    RAM of the read step. Only boundary slabs (or a slab entirely outside the
    source range) allocate a zero-filled buffer.
    """
    nz, ny, nx = dask_arr.shape
    n_planes = out_z_end - out_z_start

    in_z_start = out_z_start - sz
    in_z_end = out_z_end - sz
    read_start = max(in_z_start, 0)
    read_end = min(in_z_end, nz)

    if read_start >= read_end:
        # Entire slab maps outside the source volume -> all zeros.
        return np.zeros((n_planes, ny, nx), dtype=np.uint16)

    raw = np.asarray(dask_arr[read_start:read_end])

    if read_start == in_z_start and read_end == in_z_end:
        # No Z-padding needed: the read already covers every output plane, so
        # return the freshly materialised array directly instead of allocating
        # and copying into a second full-size slab. This is safe to mutate in
        # place (the subsequent XY shift does so): computing a dask slice yields
        # a freshly allocated array, and the production sources (h5py datasets,
        # zarr stores) always copy out of storage on read, so ``raw`` never
        # aliases any live buffer the caller still needs.
        return raw

    slab = np.zeros((n_planes, ny, nx), dtype=np.uint16)
    dst_start = read_start - in_z_start
    slab[dst_start : dst_start + (read_end - read_start)] = raw
    return slab


def _log_slab_perf(
    kind: str,
    name: str,
    z_start: int,
    z_end: int,
    nbytes: int,
    t0: float,
) -> None:
    """Emit a DEBUG per-slab timing/throughput line plus a memory snapshot."""
    if not is_debug_enabled():
        return
    dt = time.perf_counter() - t0
    mib = nbytes / 1024**2
    mibps = mib / dt if dt > 0 else 0.0
    # One self-contained line per slab: timing, throughput, and the memory
    # state at that moment. Folding the memory snapshot in (rather than a
    # separate MEM line) halves the log volume now that debug is on by default.
    log_debug(
        f"slab {kind} {name} z=[{z_start}:{z_end}] planes={z_end - z_start} "
        f"{mib:.1f}MiB in {dt * 1000:.0f}ms ({mibps:.0f} MiB/s) | {memory_status()}"
    )


def _log_pyramid_summary(name: str, summary: dict[str, Any]) -> None:
    """Log the fused pyramid totals for a channel (INFO).

    Pyramids are now built from the corrected slabs in memory, so there is no
    reread of the output: the interesting numbers are compute vs write time and
    the bytes produced.
    """
    from shifter.h5_utils import pyramid_backend

    compute_s = summary["compute_s"]
    write_s = summary["write_s"]
    mib = summary["bytes_written"] / 1024**2
    write_mibps = mib / write_s if write_s > 0 else 0.0
    log_event(
        f"pyramids {name} | levels={','.join(summary['levels']) or '-'} "
        f"backend={pyramid_backend()} "
        f"compute={compute_s:.1f}s write={write_s:.1f}s ({write_mibps:.0f} MiB/s) "
        f"wrote={mib:.0f}MiB reread=0B"
        + (f" | SKIPPED={','.join(summary['skipped'])}" if summary["skipped"] else "")
        + (f" | INCOMPLETE: {'; '.join(summary['incomplete'])}" if summary["incomplete"] else "")
    )


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
    out_ny, out_nx = ry_e - ry_s, rx_e - rx_s
    sz, sy, sx = transform.shift_z, transform.shift_y, transform.shift_x
    total_bytes = out_nz * out_ny * out_nx * 2
    bytes_done = 0

    log_event(f"Export TIFF ROI channel: {output_path.name} | "
              f"roi_shape=({out_nz},{out_ny},{out_nx}) "
              f"shift=({sz},{sy},{sx}) chunk_z={chunk_z}")

    with timed_operation(f"Write TIFF ROI: {output_path.name}"):
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as tw:
            for slab_start in range(0, out_nz, chunk_z):
                if cancel_check and cancel_check():
                    return

                slab_end = min(slab_start + chunk_z, out_nz)

                t0 = time.perf_counter()
                slab = _read_roi_slab(
                    dask_arr, roi, slab_start, slab_end, sz, sy, sx,
                )

                for i in range(slab.shape[0]):
                    tw.write(slab[i], photometric="minisblack", contiguous=True)

                _log_slab_perf(
                    "TIFF-ROI", output_path.name, slab_start, slab_end,
                    slab.nbytes, t0,
                )

                bytes_done += slab.nbytes
                if progress_callback:
                    progress_callback(bytes_done, total_bytes)


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
        Called with (bytes_written, total_bytes) after each chunk.
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

    total_bytes = nz * ny * nx * 2
    bytes_done = 0

    with timed_operation(f"Write TIFF: {output_path.name}"):
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as tw:
            for out_z_start in range(0, nz, chunk_z):
                if cancel_check and cancel_check():
                    return

                out_z_end = min(out_z_start + chunk_z, nz)

                t0 = time.perf_counter()
                slab = _read_full_slab(dask_arr, out_z_start, out_z_end, sz)

                # Apply XY shifts (parallel across planes).
                slab = _shift_slab_xy(slab, sy, sx)

                # Write planes.
                for i in range(slab.shape[0]):
                    tw.write(
                        slab[i],
                        photometric="minisblack",
                        contiguous=True,
                    )

                _log_slab_perf(
                    "TIFF", output_path.name, out_z_start, out_z_end,
                    slab.nbytes, t0,
                )

                bytes_done += slab.nbytes
                if progress_callback:
                    progress_callback(bytes_done, total_bytes)


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
        Called with (bytes_written, total_bytes) accounting for all
        channels.
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
        bytes_per_channel = (rz_e - rz_s) * roi_ny * roi_nx * 2
        suffix = "_corrected_roi"
    else:
        xy_shape = (ref_shape[1], ref_shape[2])
        bytes_per_channel = None  # varies per loader
        suffix = "_corrected"

    chunk_z = compute_chunk_size(xy_shape, n_channels, ram_percent)
    log_event(
        f"TIFF export plan | channels={n_channels} chunk_z={chunk_z} "
        f"xy=({xy_shape[0]},{xy_shape[1]}) ram%={ram_percent} "
        f"roi={roi is not None}"
    )
    log_memory("TIFF export start", level=logging.INFO)

    # Total bytes across all channels for progress reporting.
    if roi is not None:
        total_bytes = bytes_per_channel * n_channels
    else:
        total_bytes = sum(
            loader.shape[0] * loader.shape[1] * loader.shape[2] * 2
            for loader in loaders
        )
    global_bytes_done = 0

    def _channel_progress(done: int, _total: int) -> None:
        if progress_callback:
            progress_callback(global_bytes_done + done, total_bytes)

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
            global_bytes_done += bytes_per_channel
        else:
            global_bytes_done += loader.shape[0] * loader.shape[1] * loader.shape[2] * 2

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
    metadata["bytes_written"] = total_bytes
    metadata["bytes_written_gb"] = round(total_bytes / (1024**3), 3)
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
    write_pyramids: bool = True,
) -> None:
    """Export a single corrected channel to a Luxendo .lux.h5 file.

    Writes the shift-corrected ``Data`` dataset, copies the original
    ``metadata`` dataset verbatim, then (when *write_pyramids* is True)
    regenerates all pyramid levels that existed in the original file.
    Pyramid regeneration re-reads the full ``Data`` volume once per level and
    dominates export time on large volumes, so *write_pyramids=False* skips it.

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
        Called with (bytes_written, total_bytes) after each chunk write and
        after each regenerated pyramid level. Tracking *bytes* rather than
        Z-planes means the (often slow) pyramid-regeneration phase \u2014 which
        happens after ``Data`` is fully written \u2014 still moves the progress
        indicator instead of leaving it stuck at "100%".
    cancel_check : callable, optional
        Called before each chunk; if True, abort.
    roi : tuple, optional
        (z_start, z_end, y_start, y_end, x_start, x_end) crop region.
    """
    import h5py

    from shifter.h5_utils import (
        StreamingPyramidWriter,
        compute_pyramid_level_shape,
        detect_pyramid_levels,
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

    # Empty pyramid list when disabled: this makes both the byte-total below and
    # the regeneration loop further down no-ops, without special-casing either.
    pyramid_levels = detect_pyramid_levels(original_h5) if write_pyramids else []
    total_bytes = out_nz * out_ny * out_nx * 2
    for _name, fw, fh, fd in pyramid_levels:
        pnz, pny, pnx = compute_pyramid_level_shape((out_nz, out_ny, out_nx), fw, fh, fd)
        total_bytes += pnz * pny * pnx * 2
    bytes_done = 0

    with timed_operation(f"Write H5: {output_path.name}"):
        with h5py.File(str(output_path), "w") as out_h5:
            # Create output Data dataset.
            ds = out_h5.create_dataset(
                "Data",
                shape=(out_nz, out_ny, out_nx),
                dtype=np.uint16,
                chunks=h5_chunks,
            )

            # Pyramids are built from each corrected slab while it is still in
            # memory, so the output Data is never read back. Datasets are created
            # here, before the first slab, and fed by the loops below.
            pyramid_writer = None
            if pyramid_levels:
                pyr_chunks = {}
                for level_name, _fw, _fh, _fd in pyramid_levels:
                    src = original_h5.get(level_name)
                    pyr_chunks[level_name] = getattr(src, "chunks", None)
                pyramid_writer = StreamingPyramidWriter(
                    out_h5, pyramid_levels, (out_nz, out_ny, out_nx), pyr_chunks
                )

            if roi is not None:
                # ROI export: read only the needed sub-region.
                for slab_start in range(0, out_nz, chunk_z):
                    if cancel_check and cancel_check():
                        return

                    slab_end = min(slab_start + chunk_z, out_nz)

                    t0 = time.perf_counter()
                    slab = _read_roi_slab(
                        dask_arr, roi, slab_start, slab_end, sz, sy, sx,
                    )
                    ds[slab_start:slab_end] = slab

                    pyramid_bytes = 0
                    if pyramid_writer is not None:
                        pyramid_bytes = pyramid_writer.consume(slab, slab_start)

                    _log_slab_perf(
                        "H5-ROI", output_path.name, slab_start, slab_end,
                        slab.nbytes, t0,
                    )

                    bytes_done += slab.nbytes + pyramid_bytes
                    del slab  # release before the next (multi-GiB) allocation
                    if progress_callback:
                        progress_callback(bytes_done, total_bytes)
            else:
                # Full-volume export.
                for out_z_start in range(0, nz, chunk_z):
                    if cancel_check and cancel_check():
                        return

                    out_z_end = min(out_z_start + chunk_z, nz)

                    t0 = time.perf_counter()
                    slab = _read_full_slab(dask_arr, out_z_start, out_z_end, sz)

                    # Apply XY shifts (parallel across planes).
                    slab = _shift_slab_xy(slab, sy, sx)

                    ds[out_z_start:out_z_end] = slab

                    pyramid_bytes = 0
                    if pyramid_writer is not None:
                        pyramid_bytes = pyramid_writer.consume(slab, out_z_start)

                    _log_slab_perf(
                        "H5", output_path.name, out_z_start, out_z_end,
                        slab.nbytes, t0,
                    )

                    bytes_done += slab.nbytes + pyramid_bytes
                    del slab  # release before the next (multi-GiB) allocation
                    if progress_callback:
                        progress_callback(bytes_done, total_bytes)

            # Copy metadata verbatim.
            if "metadata" in original_h5:
                raw_meta = original_h5["metadata"][()]
                out_h5.create_dataset("metadata", data=raw_meta)

            # Pyramids were built alongside Data, so there is nothing to
            # regenerate here \u2014 just close them out and report.
            if pyramid_writer is not None:
                summary = pyramid_writer.finish()
                _log_pyramid_summary(output_path.name, summary)


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
    write_pyramids: bool = True,
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
        suffix = "_corrected_roi"
    else:
        # Full-volume export keeps the original filenames (no suffix) so
        # that any companion Imaris (.ims) / BigDataViewer (_bdv.h5,
        # _bdv.xml) header files copied alongside continue to resolve
        # their internal references to the per-channel files.
        xy_shape = (ref_shape[1], ref_shape[2])
        suffix = ""

    chunk_z = compute_chunk_size(xy_shape, n_channels, ram_percent)
    log_event(
        f"H5 export plan | channels={n_channels} chunk_z={chunk_z} "
        f"xy=({xy_shape[0]},{xy_shape[1]}) ram%={ram_percent} "
        f"roi={roi is not None}"
    )
    log_memory("H5 export start", level=logging.INFO)

    # Track progress in bytes rather than Z-planes: pyramid regeneration
    # (which runs after "Data" is fully written, per channel) writes no
    # planes of its own, so a plane-based total would leave the progress
    # indicator stuck once Data finishes while pyramids are still churning.
    # _channel_output_bytes counts full-res + every regenerated pyramid level,
    # the same accounting the pre-export estimate uses.
    channel_bytes = [
        _channel_output_bytes(loader, roi, include_pyramids=write_pyramids)
        for loader in loaders
    ]
    total_bytes = sum(channel_bytes)
    global_bytes_done = 0

    def _channel_progress(done: int, _total: int) -> None:
        if progress_callback:
            progress_callback(global_bytes_done + done, total_bytes)

    for i, (loader, transform) in enumerate(
        zip(loaders, shift_manager.transforms)
    ):
        out_name = h5_output_filename(transform.filename, suffix)
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
            write_pyramids=write_pyramids,
        )
        global_bytes_done += channel_bytes[i]

    # Build metadata with H5-specific fields.
    channel_dicts = shift_manager.to_channel_dicts(output_suffix=suffix)
    ref_idx = shift_manager.reference_index or 0

    # Augment channel dicts with H5-specific info.
    for i, cd in enumerate(channel_dicts):
        loader = loaders[i]
        cd["filename_corrected"] = h5_output_filename(cd["filename_original"], suffix)

        # Add channel_description and pyramid info.
        cd["channel_description"] = loader.channel_description
        cd["pyramid_levels_regenerated"] = (
            [lvl[0] for lvl in loader.pyramid_levels] if write_pyramids else []
        )

    vol_shape = ref_shape
    if roi is not None:
        vol_shape = (rz_e - rz_s, ry_e - ry_s, rx_e - rx_s)
    metadata = build_metadata(
        channel_dicts, ref_idx, voxel_xy, voxel_z, vol_shape, ram_percent,
        roi_bounds=roi,
    )
    metadata["input_format"] = "luxendo_h5"
    metadata["voxel_size_source"] = "h5_metadata"
    metadata["pyramids_written"] = write_pyramids
    metadata["bytes_written"] = total_bytes
    metadata["bytes_written_gb"] = round(total_bytes / (1024**3), 3)

    # Write companion Imaris/BigDataViewer header files (.ims, *_bdv.h5,
    # *_bdv.xml) alongside the exported data. These headers describe a
    # MULTI-resolution dataset and reference the per-channel data (and pyramid
    # levels) via HDF5 external links, so they must be reconciled with what was
    # actually written:
    #   * full volume + pyramids  -> copy verbatim (headers already match)
    #   * full volume, no pyramids -> reduce to a single (full-res) level, else
    #     their links to the absent pyramids make viewers read the data as corrupt
    #   * ROI                       -> regenerate with the ROI filenames, cropped
    #     dimensions/extent, and (when pyramids are off) a single level
    from shifter.h5_utils import (
        copy_companion_header_files,
        find_companion_header_files,
        write_roi_headers,
        write_single_resolution_headers,
    )

    input_dir = loaders[0].path.parent
    header_files = find_companion_header_files(input_dir)
    if header_files and roi is not None:
        written = write_roi_headers(
            header_files, output_dir, suffix, roi, write_pyramids
        )
        log_event(
            "Wrote ROI companion headers: "
            + ", ".join(p.name for p in written)
        )
        metadata["companion_header_files"] = [p.name for p in written]
        metadata["companion_headers_roi"] = True
        metadata["companion_headers_single_resolution"] = not write_pyramids
    elif header_files and not write_pyramids:
        # Pyramids weren't written, so the multi-resolution headers would link
        # to absent pyramid levels (which Imaris/BigDataViewer read as corrupt).
        # Rewrite them to reference only the full-resolution Data.
        written = write_single_resolution_headers(
            header_files, output_dir, output_shape_zyx=tuple(ref_shape)
        )
        log_event(
            "Wrote single-resolution companion headers: "
            + ", ".join(p.name for p in written)
        )
        metadata["companion_header_files"] = [p.name for p in written]
        metadata["companion_headers_single_resolution"] = True
    elif header_files:
        copied = copy_companion_header_files(header_files, output_dir)
        log_event(
            "Copied companion header files: "
            + ", ".join(p.name for p in copied)
        )
        metadata["companion_header_files"] = [p.name for p in copied]

    meta_path = save_metadata(metadata, output_dir)
    return meta_path
