"""Allow running as `python -m chromatic_shift_corrector`."""

try:
    from chromatic_shift_corrector.main import main

    main()
except ImportError as e:
    if "QtBindingsNotFoundError" in type(e).__name__ or "No Qt bindings" in str(e):
        import sys

        print(
            "ERROR: No Qt bindings found.\n"
            "\n"
            "This application requires a Qt backend (PyQt5, PyQt6, or PySide2).\n"
            "\n"
            "If you are using conda, install Qt with:\n"
            "    conda install pyqt\n"
            "\n"
            "If you are using pip, install Qt with:\n"
            "    pip install PyQt5\n",
            file=sys.stderr,
        )
        sys.exit(1)
    raise
