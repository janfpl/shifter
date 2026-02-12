"""Allow running as `python -m chromatic_shift_corrector`."""

import sys

try:
    from chromatic_shift_corrector._qt_setup import ensure_qt
    ensure_qt()
    from chromatic_shift_corrector.main import main
    main()
except ImportError as e:
    if "Qt" in str(e) or "QtBindingsNotFoundError" in type(e).__name__:
        print(
            "ERROR: No Qt bindings found.\n"
            "\n"
            "This application requires a Qt backend (PyQt6, PyQt5, or PySide6).\n"
            "\n"
            "If you are using conda, install Qt with:\n"
            "    conda install pyqt\n"
            "\n"
            "If you are using pip, install Qt with:\n"
            "    pip install PyQt6\n"
            "\n"
            "If you already installed Qt but still see this error, a conflicting\n"
            "package may be interfering. Try:\n"
            "    pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip PyQt6 PyQt6-Qt6 PyQt6-sip\n"
            "    pip install PyQt6\n",
            file=sys.stderr,
        )
        sys.exit(1)
    raise
