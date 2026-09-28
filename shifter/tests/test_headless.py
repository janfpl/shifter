"""Tests for headless batch processing (``python -m shifter.headless``).

Builds a synthetic Luxendo folder where the *second* channel is the brightest
and the other two are dimmer, shifted copies of it, then checks that headless
mode picks the brightest channel as reference, recovers the shifts, and writes
aligned ``.lux.h5`` output into a new sibling ``*_shifted`` folder.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_headless
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("SHIFTER_DISABLE_GPU", "1")

from shifter import headless
from shifter.tests.test_h5 import _apply_shift, _create_h5_file, _make_blob_volume

# Channel 1 is the brightest; 0 and 2 are dimmer and shifted relative to it.
SHIFTS = {0: (3, 1, -1), 2: (-4, -1, 1)}  # applied (Z, Y, X)
SCALES = {0: 0.5, 1: 1.0, 2: 0.3}


def _make_dataset(folder: Path) -> np.ndarray:
    folder.mkdir(parents=True)
    ref = _make_blob_volume(np.random.default_rng(7))
    for ch, scale in SCALES.items():
        data = (ref.astype(np.float64) * scale).astype(np.uint16)
        if ch in SHIFTS:
            data = _apply_shift(data, SHIFTS[ch])
        _create_h5_file(folder / f"ch{ch}.lux.h5", data, ch)
    return ref


def test_parse_folder_list(tmp_path: Path = None) -> None:
    base = Path("/base")
    text = '# comment\nA, "B"\n\n  /abs/C ,\nD\n'
    got = headless.parse_folder_list(text, base)
    assert got == [base / "A", base / "B", Path("/abs/C"), base / "D"], got


def test_expand_inputs_comma_list() -> None:
    got = headless.expand_inputs(["/nope/a, /nope/b"])
    assert got == [Path("/nope/a"), Path("/nope/b")], got


def test_auto_roi() -> None:
    assert headless.auto_roi((200, 100, 50)) == (0, 200, 49, 51, 0, 50)
    assert headless.auto_roi((200, 1, 50)) == (0, 200, 0, 1, 0, 50)


def test_output_folder_unique(tmp_path: Path = None) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "sample1"
        src.mkdir()
        when = datetime(2026, 9, 28, 14, 30)
        first = headless.output_folder_for(src, when)
        assert first.name == "sample1_092826_1430_shifted"
        first.mkdir()
        assert headless.output_folder_for(src, when).name == "sample1_092826_1430_shifted_2"


def test_process_folder_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "sample1"
        ref = _make_dataset(src)

        # Test volume is 100 planes deep, so the Z search range must stay < 50;
        # a taller Y slab keeps MI well-conditioned on synthetic blobs.
        out = headless.process_folder(
            src, search_xy=3, search_z=8, roi_y=64, use_gpu=False
        )

        assert out.parent == src.parent
        assert out.name.startswith("sample1_") and out.name.endswith("_shifted")
        assert sorted(p.name for p in out.glob("*.lux.h5")) == [
            "ch0.lux.h5", "ch1.lux.h5", "ch2.lux.h5"
        ]

        meta = json.loads((out / "correction_metadata.json").read_text())
        reg = meta["headless_registration"]
        assert int(np.argmax(reg["channel_mean_intensity"])) == 1
        assert reg["source_folder"] == str(src.resolve())

        # Corrected channels line up with the reference in the interior.
        m = 12
        inner = (slice(m, -m),) * 3
        for ch, scale in SCALES.items():
            with h5py.File(out / f"ch{ch}.lux.h5", "r") as f:
                got = f["Data"][inner]
            expected = (ref.astype(np.float64) * scale).astype(np.uint16)[inner]
            assert np.array_equal(got, expected), f"ch{ch} not aligned"


def test_main_batch_continues_after_failure() -> None:
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good"
        _make_dataset(good)
        listing = Path(d) / "folders.txt"
        listing.write_text("missing_folder, good\n")
        rc = headless.main(
            [str(listing), "--xy-range", "3", "--z-range", "8", "--roi-y", "64"]
        )
        assert rc == 1  # one folder failed
        assert len(list(Path(d).glob("good_*_shifted"))) == 1


def test_single_channel_folder_is_skipped() -> None:
    with tempfile.TemporaryDirectory() as d:
        single = Path(d) / "single"
        single.mkdir()
        _create_h5_file(single / "ch0.lux.h5", _make_blob_volume(np.random.default_rng(1)), 0)
        assert headless.main([str(single)]) == 0
        assert list(Path(d).glob("*_shifted")) == []


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    sys.exit(1 if failed else 0)
