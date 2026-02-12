"""Allow running as `python -m chromatic_shift_corrector`."""

import sys

try:
    from chromatic_shift_corrector._qt_setup import ensure_qt
    ensure_qt()
    from chromatic_shift_corrector.main import main
    main()
except ImportError as e:
    if "Qt" in str(e) or "QtBindingsNotFoundError" in type(e).__name__:
        import os
        # Gather diagnostics so the user can report what's actually installed.
        diag_lines = [f"  QT_API env var: {os.environ.get('QT_API', '<not set>')}"]
        for pkg in ("PyQt6", "PyQt5", "PySide6", "PySide2"):
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", getattr(mod, "PYQT_VERSION_STR", "?"))
                diag_lines.append(f"  {pkg}: {ver}")
            except ImportError:
                diag_lines.append(f"  {pkg}: not installed")
        diag = "\n".join(diag_lines)
        print(
            f"ERROR: No Qt bindings found.\n"
            f"\n"
            f"Diagnostic info:\n"
            f"{diag}\n"
            f"\n"
            f"This application requires a Qt backend (PyQt6, PyQt5, or PySide6).\n"
            f"\n"
            f"If you are using conda, install Qt with:\n"
            f"    conda install pyqt\n"
            f"\n"
            f"If you are using pip, install Qt with:\n"
            f"    pip install PyQt6\n"
            f"\n"
            f"If you already installed Qt but still see this error, a conflicting\n"
            f"package may be interfering. Try creating a fresh environment:\n"
            f"    conda create -n shifter python=3.12\n"
            f"    conda activate shifter\n"
            f"    conda install pyqt\n"
            f"    pip install -e .\n",
            file=sys.stderr,
        )
        sys.exit(1)
    raise
