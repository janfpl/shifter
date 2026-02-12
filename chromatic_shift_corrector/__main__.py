"""Allow running as `python -m chromatic_shift_corrector`."""

import os
import sys


def _try_import_qt():
    """Try to import qtpy, cycling through Qt backends if needed."""
    # If QT_API is already set, respect it.
    if "QT_API" not in os.environ:
        # Try default detection first.
        try:
            import qtpy  # noqa: F401
            return
        except ImportError:
            pass

        # Default failed — try each backend explicitly.
        for api in ("pyqt6", "pyqt5", "pyside6", "pyside2"):
            os.environ["QT_API"] = api
            try:
                import importlib
                if "qtpy" in sys.modules:
                    importlib.reload(sys.modules["qtpy"])
                else:
                    import qtpy  # noqa: F401
                return
            except ImportError:
                continue

        # All failed.
        del os.environ["QT_API"]
        raise ImportError("No Qt bindings found")


try:
    _try_import_qt()
    from chromatic_shift_corrector.main import main
    main()
except ImportError as e:
    if "Qt" in str(e) or "QtBindingsNotFoundError" in type(e).__name__:
        print(
            "ERROR: No Qt bindings found.\n"
            "\n"
            "This application requires a Qt backend (PyQt5, PyQt6, or PySide2).\n"
            "\n"
            "If you are using conda, install Qt with:\n"
            "    conda install pyqt\n"
            "\n"
            "If you are using pip, install Qt with:\n"
            "    pip install PyQt5\n"
            "\n"
            "If you already installed Qt but still see this error, a conflicting\n"
            "package may be interfering. Try:\n"
            "    pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip\n"
            "    pip install -e .\n",
            file=sys.stderr,
        )
        sys.exit(1)
    raise
