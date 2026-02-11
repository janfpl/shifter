"""Entry point — launches napari with the Chromatic Shift Corrector widget."""

from __future__ import annotations


def main() -> None:
    import napari

    from chromatic_shift_corrector.widget import ChromaticShiftWidget

    viewer = napari.Viewer(title="Chromatic Shift Corrector")
    widget = ChromaticShiftWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Chromatic Shift Corrector", area="right")
    napari.run()


if __name__ == "__main__":
    main()
