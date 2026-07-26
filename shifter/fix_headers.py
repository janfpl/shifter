"""Reduce copied Imaris/BigDataViewer headers to single (full-resolution) level.

Use this on a Luxendo H5 export folder that was written **without** pyramid
layers but still has the multi-resolution companion headers (``*.ims`` /
``*_bdv.h5``) alongside it. Those headers link to pyramid levels
(``Data_2_2_2``, ...) that don't exist, so Imaris / BigDataViewer — and the
Imaris File Converter — read the dataset as corrupt. This rewrites the headers
**in place** to reference only the full-resolution ``Data`` so they resolve
cleanly. It never touches the ``.lux.h5`` data files.

Running it again is harmless (already-single-resolution headers are left as-is).

Usage::

    python -m shifter.fix_headers /path/to/export_folder
"""

from __future__ import annotations

import sys
from pathlib import Path

from shifter.h5_utils import reduce_header_to_single_resolution


def fix_folder(folder: Path | str) -> list[Path]:
    """Reduce every ``*.ims`` / ``*_bdv.h5`` header in *folder* to single-res.

    Returns the list of header files that were reduced.
    """
    folder = Path(folder)
    fixed: list[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".ims") or name.endswith("_bdv.h5"):
            if reduce_header_to_single_resolution(p):
                fixed.append(p)
    return fixed


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m shifter.fix_headers <export_folder>", file=sys.stderr)
        sys.exit(2)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    fixed = fix_folder(folder)
    if fixed:
        print("Reduced to single (full-resolution) level:")
        for p in fixed:
            print(f"  {p.name}")
        print(
            "\nThese headers now reference only the full-resolution 'Data' in the "
            "companion .lux.h5 files and are safe to open / import."
        )
    else:
        print(f"No .ims / *_bdv.h5 headers found in {folder}.")


if __name__ == "__main__":
    main()
