"""Tests for the CUDA version detection and compatibility helpers.

These cover the version-preference logic that lets the GPU auto-detection
handle both CUDA 12.x and CUDA 13.x toolkits without importing CuPy or
requiring a GPU.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_gpu_utils
"""

from __future__ import annotations

import sys

from shifter.registration import gpu_utils as g


def test_supported_majors_include_12_and_13() -> None:
    assert 12 in g._SUPPORTED_CUDA_MAJORS
    assert 13 in g._SUPPORTED_CUDA_MAJORS


def test_cuda_path_var_major_parsing() -> None:
    assert g._cuda_path_var_major("CUDA_PATH_V12_6") == 12
    assert g._cuda_path_var_major("CUDA_PATH_V13_0") == 13
    assert g._cuda_path_var_major("CUDA_PATH_V9_2") == 9
    # Unparseable suffixes fall back to 0.
    assert g._cuda_path_var_major("CUDA_PATH_VXX") == 0


def test_major_pref_rank_orders_supported_newest_first() -> None:
    # Supported majors sort before unsupported ones, newest first within
    # each group.
    assert sorted([11, 13, 12, 15], key=g._major_pref_rank) == [13, 12, 15, 11]


def test_cuda_path_var_sort_prefers_newest_supported() -> None:
    keys = ["CUDA_PATH_V11_8", "CUDA_PATH_V13_0", "CUDA_PATH_V12_6"]
    keys.sort(key=lambda k: (g._major_pref_rank(g._cuda_path_var_major(k)), k))
    assert keys == ["CUDA_PATH_V13_0", "CUDA_PATH_V12_6", "CUDA_PATH_V11_8"]


def test_cuda_runtime_major_from_version_int() -> None:
    class _FakeRuntime:
        def __init__(self, version: int) -> None:
            self._version = version

        def runtimeGetVersion(self) -> int:
            return self._version

    class _FakeCuda:
        def __init__(self, version: int) -> None:
            self.runtime = _FakeRuntime(version)

    class _FakeCupy:
        def __init__(self, version: int) -> None:
            self.cuda = _FakeCuda(version)

    assert g._cuda_runtime_major(_FakeCupy(12060)) == 12
    assert g._cuda_runtime_major(_FakeCupy(13000)) == 13
    # A module that raises is treated as "unknown" (0).
    assert g._cuda_runtime_major(object()) == 0


def test_install_hint_lists_supported_wheels() -> None:
    hint = g._cupy_install_hint()
    assert "cupy-cuda12x" in hint
    assert "cupy-cuda13x" in hint


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
