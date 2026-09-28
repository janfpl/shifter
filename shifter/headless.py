"""Headless batch processing — register and export Luxendo H5 folders without napari.

For each input folder of ``.lux.h5`` channel files this:

1. Loads every channel (companion ``.ims`` / ``*_bdv.h5`` headers are skipped).
2. Picks the brightest channel (highest mean intensity) as the reference.
3. Builds the automatic registration ROI: centred, full X width, a thin
   Y slab (2 voxels by default) at the Y midpoint, and the full Z depth.
4. Registers every other channel against the reference with Mutual
   Information (default search range: 1 voxel XY, 90 voxels Z).
5. Exports full-volume corrected ``.lux.h5`` files (original filenames,
   regenerated pyramids, companion headers) into a new sibling folder named
   ``<source folder>_<MMDDYY_HHMM>_shifted``.

Usage::

    python -m shifter.headless D:\\data\\sample1
    python -m shifter.headless D:\\data\\sample1 D:\\data\\sample2
    python -m shifter.headless folders.txt

A ``.txt`` file lists folders separated by commas and/or new lines; lines
starting with ``#`` are ignored and relative paths are resolved against the
text file's own folder. Folders are processed one after another; a failure in
one folder is reported and the rest still run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ALGORITHM = "Mutual Information"
DEFAULT_SEARCH_XY = 1
DEFAULT_SEARCH_Z = 90
DEFAULT_ROI_Y = 2  # also the GUI's "Add registration ROI" default height
DEFAULT_RAM_PERCENT = 90


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #

def parse_folder_list(text: str, base_dir: Path) -> list[Path]:
    """Split a comma/newline separated list of folders into paths."""
    folders: list[Path] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for item in line.split(","):
            item = item.strip().strip('"').strip("'").strip()
            if not item:
                continue
            p = Path(item)
            if not p.is_absolute():
                p = base_dir / p
            folders.append(p)
    return folders


def expand_inputs(inputs: list[str]) -> list[Path]:
    """Turn command-line arguments (folders and/or list files) into folders."""
    folders: list[Path] = []
    for raw in inputs:
        p = Path(raw.strip().strip('"'))
        if p.is_file():
            folders.extend(
                parse_folder_list(p.read_text(encoding="utf-8-sig"), p.parent)
            )
        elif not p.exists() and "," in raw:
            # A comma-separated list typed or pasted at the prompt.
            folders.extend(parse_folder_list(raw, Path.cwd()))
        else:
            folders.append(p)
    return folders


# --------------------------------------------------------------------------- #
# Per-dataset steps
# --------------------------------------------------------------------------- #

def find_channel_files(folder: Path) -> list[Path]:
    """Return the ``.lux.h5`` channel files in *folder* (headers excluded)."""
    from shifter.h5_utils import find_companion_header_files, scan_h5_files

    headers = {p.resolve() for p in find_companion_header_files(folder)}
    return [p for p in scan_h5_files(folder) if p.resolve() not in headers]


def auto_roi(
    shape_zyx: tuple[int, int, int], roi_y: int = DEFAULT_ROI_Y
) -> tuple[int, int, int, int, int, int]:
    """Automatic registration ROI: full X, *roi_y* voxels at the Y midpoint, full Z.

    Matches the GUI's "Add registration ROI" default. Returns
    ``(z_start, z_end, y_start, y_end, x_start, x_end)`` with exclusive ends.
    """
    nz, ny, nx = shape_zyx
    size = max(1, min(roi_y, ny))
    y_start = ny // 2 - size // 2
    return 0, nz, y_start, y_start + size, 0, nx


def channel_brightness(
    loaders: list[Any], roi: tuple[int, int, int, int, int, int]
) -> list[float]:
    """Mean intensity per channel.

    Uses each channel's coarsest pyramid level when every channel has one
    (cheap, and covers the whole volume); otherwise falls back to the
    registration ROI so the full-resolution volume is never read in full.
    """
    if all(getattr(ld, "num_levels", 1) > 1 for ld in loaders):
        return [float(ld.multiscale[-1].mean().compute()) for ld in loaders]
    z0, z1, y0, y1, x0, x1 = roi
    return [
        float(ld.dask_array[z0:z1, y0:y1, x0:x1].mean().compute())
        for ld in loaders
    ]


def output_folder_for(source: Path, when: datetime) -> Path:
    """``<parent>/<source name>_<MMDDYY_HHMM>_shifted``, never an existing folder."""
    base = f"{source.name}_{when.strftime('%m%d%y_%H%M')}_shifted"
    out = source.parent / base
    n = 2
    while out.exists():
        out = source.parent / f"{base}_{n}"
        n += 1
    return out


class SkipFolder(Exception):
    """The folder needs no correction (e.g. a single channel); nothing is written."""


class _ProgressPrinter:
    """Single-line console progress that only redraws when the percent changes."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._last = -1

    def __call__(self, done: Any, total: Any, *_: Any) -> None:
        if not total:
            return
        pct = int(100 * done / total)
        if pct != self._last:
            self._last = pct
            print(f"\r    {self.label}: {pct:3d}%", end="", flush=True)

    def finish(self) -> None:
        if self._last >= 0:
            print()


def process_folder(
    folder: Path,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
    search_xy: int = DEFAULT_SEARCH_XY,
    search_z: int = DEFAULT_SEARCH_Z,
    roi_y: int = DEFAULT_ROI_Y,
    ram_percent: int = DEFAULT_RAM_PERCENT,
    write_pyramids: bool = True,
    use_gpu: bool | None = None,
) -> Path:
    """Register and export one folder. Returns the output folder."""
    from shifter.data_loader import H5Loader, validate_channels
    from shifter.export_engine import run_export_h5
    from shifter.h5_utils import H5FileManager
    from shifter.memory import release_memory
    from shifter.perf_logger import log_event, setup_perf_log
    from shifter.registration import CONFIDENCE_LOW, gpu_available
    from shifter.registration_runner import register_channels
    from shifter.shift_manager import ShiftManager
    from shifter.utils import load_metadata, save_metadata

    folder = folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = find_channel_files(folder)
    if not files:
        raise ValueError(f"No .lux.h5 channel files found in {folder}")
    if len(files) < 2:
        raise SkipFolder(
            f"only one channel ({files[0].name}), nothing to register against"
        )

    started = datetime.now()
    out_dir = output_folder_for(folder, started)

    file_manager = H5FileManager()
    try:
        loaders = [H5Loader(p, file_manager) for p in files]
        ok, msg = validate_channels(loaders)
        if not ok:
            raise ValueError(msg)

        shape = (
            min(ld.shape[0] for ld in loaders),
            min(ld.shape[1] for ld in loaders),
            min(ld.shape[2] for ld in loaders),
        )
        roi = auto_roi(shape, roi_y)
        z0, z1, y0, y1, x0, x1 = roi
        if z1 - z0 < 2 * search_z or y1 - y0 < 2 * search_xy or x1 - x0 < 2 * search_xy:
            raise ValueError(
                f"Registration ROI ({x1 - x0}x{y1 - y0}x{z1 - z0} XYZ) is smaller "
                f"than 2x the search range ({2 * search_xy}x{2 * search_xy}x"
                f"{2 * search_z}). Reduce the search range or increase --roi-y."
            )

        out_dir.mkdir(parents=True)
        setup_perf_log(out_dir)
        log_event(f"Headless processing | source={folder}")

        print(f"  Channels ({len(files)}):")
        brightness = channel_brightness(loaders, roi)
        ref = int(np.argmax(brightness))
        for i, (p, b) in enumerate(zip(files, brightness)):
            tag = "  <- reference (brightest)" if i == ref else ""
            print(f"    ch{i} {p.name}  mean={b:.1f}{tag}")
        log_event(
            "Channel brightness: "
            + ", ".join(f"ch{i}={b:.1f}" for i, b in enumerate(brightness))
            + f" | reference=ch{ref}"
        )

        print(
            f"  ROI: X=[{x0},{x1}) Y=[{y0},{y1}) Z=[{z0},{z1})  |  "
            f"{algorithm}, search XY={search_xy} Z={search_z}"
        )
        if use_gpu is None:
            use_gpu = gpu_available()
        moving = [i for i in range(len(loaders)) if i != ref]
        bar = _ProgressPrinter("Registration")
        results = register_channels(
            loaders, ref, moving, algorithm, {},
            search_xy, search_z,
            (y0, y1, x0, x1), z0, z1,
            background_subtraction=False,
            gaussian_smoothing=False,
            use_gpu=use_gpu,
            progress=bar,
        )
        bar.finish()
        release_memory(use_gpu=use_gpu, context="registration")

        shift_manager = ShiftManager()
        shift_manager.init_channels([p.name for p in files], ref, ["gray"] * len(files))
        registration_record = []
        for ch_i, r in results:
            for axis, val in (("x", r.shift_x), ("y", r.shift_y), ("z", r.shift_z)):
                shift_manager.set_shift(ch_i, axis, val)
            notes = []
            if abs(r.shift_x) >= search_xy or abs(r.shift_y) >= search_xy:
                notes.append("XY shift at search limit")
            if abs(r.shift_z) >= search_z:
                notes.append("Z shift at search limit")
            if r.confidence < CONFIDENCE_LOW:
                notes.append("low confidence")
            warn = f"  WARNING: {', '.join(notes)}" if notes else ""
            print(
                f"    ch{ch_i}: X={r.shift_x:+d} Y={r.shift_y:+d} Z={r.shift_z:+d}  "
                f"confidence={r.confidence:.2f}{warn}"
            )
            registration_record.append({
                "channel_index": ch_i,
                "shift_x": r.shift_x,
                "shift_y": r.shift_y,
                "shift_z": r.shift_z,
                "confidence": r.confidence,
                "warnings": notes,
            })

        meta = loaders[ref].h5_metadata
        voxel_xy = meta.get("voxel_size_xy_um") or 1.0
        voxel_z = meta.get("voxel_size_z_um") or 1.0

        print(f"  Exporting to {out_dir}")
        bar = _ProgressPrinter("Export")
        meta_path = run_export_h5(
            loaders, shift_manager, out_dir, ram_percent,
            progress_callback=bar,
            voxel_xy=voxel_xy, voxel_z=voxel_z,
            write_pyramids=write_pyramids,
        )
        bar.finish()
        release_memory(context="export")

        # Record how the shifts were found alongside the shifts themselves.
        metadata = load_metadata(meta_path)
        metadata["headless_registration"] = {
            "source_folder": str(folder),
            "algorithm": algorithm,
            "search_range_xy": search_xy,
            "search_range_z": search_z,
            "roi_zyx": {"z": [z0, z1], "y": [y0, y1], "x": [x0, x1]},
            "reference_selection": "brightest (mean intensity)",
            "channel_mean_intensity": brightness,
            "results": registration_record,
        }
        save_metadata(metadata, out_dir)
        return out_dir
    finally:
        file_manager.close_all()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shifter.headless",
        description="Register and export Luxendo H5 folders without opening napari.",
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Data folder(s), or .txt file(s) listing folders (comma or newline separated).",
    )
    parser.add_argument("--xy-range", type=int, default=DEFAULT_SEARCH_XY,
                        help=f"XY search range in voxels (default {DEFAULT_SEARCH_XY})")
    parser.add_argument("--z-range", type=int, default=DEFAULT_SEARCH_Z,
                        help=f"Z search range in voxels (default {DEFAULT_SEARCH_Z})")
    parser.add_argument("--roi-y", type=int, default=DEFAULT_ROI_Y,
                        help=f"Height of the registration ROI in Y voxels (default {DEFAULT_ROI_Y})")
    parser.add_argument("--ram", type=int, default=DEFAULT_RAM_PERCENT,
                        help=f"Export RAM allocation percent (default {DEFAULT_RAM_PERCENT})")
    parser.add_argument("--no-pyramids", action="store_true",
                        help="Do not write low-resolution pyramid layers")
    args = parser.parse_args(argv)

    from shifter.logging_setup import configure_logging
    configure_logging()

    folders = expand_inputs(args.inputs)
    if not folders:
        print("No folders to process.")
        return 1

    print(f"{len(folders)} folder(s) to process.")
    outcomes: list[tuple[Path, str]] = []
    for i, folder in enumerate(folders, start=1):
        print(f"\n[{i}/{len(folders)}] {folder}")
        t0 = time.monotonic()
        try:
            out = process_folder(
                folder,
                search_xy=args.xy_range,
                search_z=args.z_range,
                roi_y=args.roi_y,
                ram_percent=args.ram,
                write_pyramids=not args.no_pyramids,
            )
            mins = (time.monotonic() - t0) / 60
            print(f"  Done in {mins:.1f} min -> {out}")
            outcomes.append((folder, f"OK      -> {out}"))
        except SkipFolder as exc:
            print(f"  SKIPPED: {exc}")
            outcomes.append((folder, f"SKIPPED {exc}"))
        except Exception as exc:
            logger.debug("Failed processing %s", folder, exc_info=True)
            print(f"  FAILED: {exc}")
            if not isinstance(exc, (FileNotFoundError, ValueError)):
                traceback.print_exc()
            outcomes.append((folder, f"FAILED  {exc}"))

    print("\nSummary:")
    for folder, status in outcomes:
        print(f"  {folder}: {status}")
    return 0 if not any(s.startswith("FAILED") for _, s in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
