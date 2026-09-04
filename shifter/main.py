"""Entry point — launches napari with the Chromatic Shift Corrector widget."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def _set_default_font() -> None:
    """Set a default application font on Windows to avoid DirectWrite warnings.

    Qt may fall back to "MS Sans Serif" which triggers::

        DirectWrite: CreateFontFaceFromHDC() failed …

    Setting "Segoe UI" (available on all modern Windows) prevents this.
    """
    try:
        if sys.platform != "win32":
            return
        from qtpy.QtWidgets import QApplication
        from qtpy.QtGui import QFont

        app = QApplication.instance()
        if app is None:
            return
        app.setFont(QFont("Segoe UI", 9))
    except Exception:
        pass  # Non-critical — the warning is cosmetic.


def _fit_window_to_screen(viewer) -> None:
    """Constrain the viewer window so its frame fits the available screen area.

    On high-resolution displays (e.g. 4K) Qt may attempt to set a window
    geometry whose frame (title-bar + borders) extends beyond the desktop,
    producing the warning::

        QWindowsWindow::setGeometry: Unable to set geometry …

    This function shrinks the content area just enough so the full frame
    (including decorations) stays within the available screen geometry.
    """
    try:
        from qtpy.QtWidgets import QApplication

        qt_window = viewer.window._qt_window
        screen = qt_window.screen() or QApplication.primaryScreen()
        if screen is None:
            return

        avail = screen.availableGeometry()
        frame = qt_window.frameGeometry()

        # Compute decoration overhead (title-bar, borders).
        margins_w = frame.width() - qt_window.width()
        margins_h = frame.height() - qt_window.height()

        max_content_w = avail.width() - margins_w
        max_content_h = avail.height() - margins_h

        new_w = min(qt_window.width(), max_content_w)
        new_h = min(qt_window.height(), max_content_h)

        if new_w < qt_window.width() or new_h < qt_window.height():
            qt_window.resize(new_w, new_h)
            logger.debug(
                "Resized viewer window to %dx%d to fit available screen "
                "area %dx%d",
                new_w,
                new_h,
                avail.width(),
                avail.height(),
            )
    except Exception:
        pass  # Non-critical — the Qt warning is cosmetic.


def main() -> None:
    # Configure terminal logging and crash diagnostics first, so that anything
    # going wrong during startup (or later inside the Qt event loop) surfaces a
    # traceback instead of silently aborting the process.
    from shifter.logging_setup import configure_logging, install_diagnostics
    configure_logging()
    install_diagnostics()

    from shifter._qt_setup import ensure_qt
    ensure_qt()

    # Qt is importable now — capture its own warnings/fatals too.
    install_diagnostics()

    try:
        import napari

        from shifter.widget import ChromaticShiftWidget

        logger.info("Creating napari viewer")
        viewer = napari.Viewer(title="Chromatic Shift Corrector", show=False)
        _set_default_font()
        _fit_window_to_screen(viewer)
        viewer.show()

        logger.info("Building Chromatic Shift Corrector widget")
        widget = ChromaticShiftWidget(viewer)
        viewer.window.add_dock_widget(
            widget, name="Chromatic Shift Corrector", area="right"
        )
        logger.info("Startup complete — entering napari event loop")
        napari.run()
    except Exception:
        logger.exception("Fatal error during application startup")
        raise


if __name__ == "__main__":
    main()
