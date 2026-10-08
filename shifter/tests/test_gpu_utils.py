"""Tests for the CUDA version detection and compatibility helpers.

These cover the version-preference logic that lets the GPU auto-detection
handle both CUDA 12.x and CUDA 13.x toolkits without importing CuPy or
requiring a GPU.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_gpu_utils
"""

from __future__ import annotations

import sys
from pathlib import Path, PureWindowsPath

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


def test_normalize_cuda_root_strips_bin_components() -> None:
    root = PureWindowsPath(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
    assert g._normalize_cuda_root(root) == root
    assert g._normalize_cuda_root(root / "bin") == root
    assert g._normalize_cuda_root(root / "BIN") == root
    assert g._normalize_cuda_root(root / "bin" / "x64") == root
    assert g._normalize_cuda_root(root / "lib" / "x64") == root


def _make_toolkit(root: Path, dll_subdir: str) -> Path:
    dll_dir = root / dll_subdir
    dll_dir.mkdir(parents=True)
    (dll_dir / "nvrtc64_130_0.dll").touch()
    return root


def test_has_nvrtc_handles_cuda12_and_cuda13_layouts(tmp_path: Path) -> None:
    assert g._has_nvrtc(_make_toolkit(tmp_path / "v12.6", "bin"))
    assert g._has_nvrtc(_make_toolkit(tmp_path / "v13.2", "bin/x64"))
    assert not g._has_nvrtc(tmp_path / "missing")


def test_use_cuda_root_fixes_cuda_path_pointing_at_bin(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression: CUDA_PATH=<root>\bin made CuPy look for <root>\bin\bin.
    root = _make_toolkit(tmp_path / "v13.2", "bin/x64")
    monkeypatch.setenv("CUDA_PATH", str(root / "bin"))
    monkeypatch.setenv("PATH", "")
    assert g._use_cuda_root(str(root / "bin"), "CUDA_PATH")
    import os
    assert os.environ["CUDA_PATH"] == str(root)
    assert str(root / "bin" / "x64") in os.environ["PATH"]


def test_use_cuda_root_rejects_invalid_paths(tmp_path: Path) -> None:
    assert not g._use_cuda_root(None, "CUDA_PATH")
    assert not g._use_cuda_root(str(tmp_path / "nope"), "CUDA_PATH")


def test_driver_insufficient_detection() -> None:
    # CUDA 12.x driver running a CUDA 13.x CuPy build.
    assert g._driver_insufficient({"runtime": 13000, "driver": 12040}, "")
    # Minor-version compatibility within a major line is fine.
    assert not g._driver_insufficient({"runtime": 12090, "driver": 12060}, "")
    assert not g._driver_insufficient({"runtime": 13020, "driver": 13000}, "")
    # The CUDA error name alone is enough, even without versions.
    assert g._driver_insufficient(
        None,
        "CUDA device detection failed: cudaErrorInsufficientDriver: CUDA "
        "driver version is insufficient for CUDA runtime version",
    )
    assert not g._driver_insufficient(None, "CuPy is not installed")


def _banner_for(monkeypatch, versions, reason: str) -> str:
    lines: list[str] = []
    monkeypatch.setattr(g, "_GPU_CUDA_VERSIONS", versions)
    monkeypatch.setattr(g, "_print_banner", lambda body: lines.extend(body))
    monkeypatch.setattr(g.logger, "warning", lambda *a: lines.append(a[0] % a[1:]))
    g._report_gpu_unavailable(reason)
    return "\n".join(lines)


def test_banner_cuda13_is_supported(monkeypatch) -> None:
    text = _banner_for(monkeypatch, {"runtime": 13020, "driver": 13020},
                       "NVRTC compilation failed: boom")
    assert "unsupported CUDA version" not in text
    assert "cupy-cuda13x" in text


def test_banner_old_driver(monkeypatch) -> None:
    text = _banner_for(monkeypatch, {"runtime": 13020, "driver": 12040},
                       "CUDA device detection failed: cudaErrorInsufficientDriver")
    assert "driver is too old" in text
    assert "CUDA 13.2" in text and "CUDA 12.4" in text


def test_banner_unsupported_major(monkeypatch) -> None:
    text = _banner_for(monkeypatch, {"runtime": 11080, "driver": 12040},
                       "GPU computation test failed: x")
    assert "unsupported CUDA version" in text
    assert "12.x or 13.x" in text


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
