"""Qt backend detection — imported by both __main__.py and main.py."""

import os
import sys


def _purge_qtpy_modules():
    """Remove qtpy and all its sub-modules from sys.modules.

    A simple importlib.reload on the top-level qtpy module is not enough
    because stale sub-module entries (qtpy.QtCore, qtpy._utils, etc.) survive
    and prevent the reloaded qtpy from re-detecting a different backend.
    """
    to_remove = [key for key in sys.modules if key == "qtpy" or key.startswith("qtpy.")]
    for key in to_remove:
        del sys.modules[key]


def _try_direct_import(name):
    """Try importing a Qt backend directly (bypassing qtpy) to get the real error."""
    try:
        __import__(name + ".QtCore")
        return True, None
    except Exception as exc:
        return False, exc


def ensure_qt():
    """Try to import qtpy, cycling through Qt backends if needed.

    Safe to call multiple times — returns immediately once a backend is loaded.
    """
    # Fast path: if qtpy already imported successfully, nothing to do.
    qtpy_mod = sys.modules.get("qtpy")
    if qtpy_mod is not None and hasattr(qtpy_mod, "API_NAME"):
        return

    # If QT_API is already set, try it first — but fall through on failure.
    if "QT_API" in os.environ:
        try:
            import qtpy  # noqa: F401
            return
        except ImportError:
            _purge_qtpy_modules()
            # The pre-set value didn't work; clear it and try all backends.
            del os.environ["QT_API"]

    # Try default detection (qtpy picks the first backend it finds).
    try:
        import qtpy  # noqa: F401
        return
    except ImportError:
        _purge_qtpy_modules()

    # Default failed — try each backend explicitly.
    errors = {}
    for api, pkg in [("pyqt6", "PyQt6"), ("pyqt5", "PyQt5"),
                     ("pyside6", "PySide6"), ("pyside2", "PySide2")]:
        os.environ["QT_API"] = api
        _purge_qtpy_modules()
        try:
            import qtpy  # noqa: F401
            return
        except ImportError:
            # Also try a direct import to capture the real error (e.g. DLL load failure).
            _ok, exc = _try_direct_import(pkg)
            if _ok:
                # The package itself imports fine but qtpy rejects it — unusual.
                errors[pkg] = "importable but rejected by qtpy"
            else:
                errors[pkg] = str(exc) if exc else "not installed"
            continue

    # All failed — clean up and raise with diagnostic info.
    if "QT_API" in os.environ:
        del os.environ["QT_API"]
    _purge_qtpy_modules()

    diag_lines = [f"  QT_API env var was: {os.environ.get('QT_API', '<not set>')}",
                  f"  Python: {sys.version}",
                  f"  Platform: {sys.platform}"]
    for pkg, err in errors.items():
        diag_lines.append(f"  {pkg}: {err}")
    diag = "\n".join(diag_lines)

    raise ImportError(
        f"No Qt bindings found.\n"
        f"\n"
        f"Diagnostic info:\n"
        f"{diag}\n"
        f"\n"
        f"Install a Qt backend with:  pip install PyQt6\n"
        f"Or with conda:              conda install pyqt"
    )
