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


def ensure_qt():
    """Try to import qtpy, cycling through Qt backends if needed.

    Safe to call multiple times — returns immediately once a backend is loaded.
    """
    # Fast path: if qtpy already imported successfully, nothing to do.
    qtpy_mod = sys.modules.get("qtpy")
    if qtpy_mod is not None and hasattr(qtpy_mod, "API_NAME"):
        return

    # If QT_API is already set, respect it.
    if "QT_API" not in os.environ:
        # Try default detection first.
        try:
            import qtpy  # noqa: F401
            return
        except ImportError:
            _purge_qtpy_modules()

        # Default failed — try each backend explicitly.
        for api in ("pyqt6", "pyqt5", "pyside6", "pyside2"):
            os.environ["QT_API"] = api
            _purge_qtpy_modules()
            try:
                import qtpy  # noqa: F401
                return
            except ImportError:
                continue

        # All failed.
        del os.environ["QT_API"]
        _purge_qtpy_modules()
        raise ImportError("No Qt bindings found")
    else:
        # QT_API is set — just try importing.
        import qtpy  # noqa: F401
