"""Entry point — launches napari with the Image Processing Pipeline widget."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
    from chromatic_shift_corrector._qt_setup import ensure_qt
    ensure_qt()

    import napari

    from chromatic_shift_corrector.widget import ImageProcessingWidget

    viewer = napari.Viewer(title="H5 Image Processor", show=False)
    _fit_window_to_screen(viewer)
    viewer.show()

    widget = ImageProcessingWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Image Processor", area="right")
    napari.run()


if __name__ == "__main__":
    main()
