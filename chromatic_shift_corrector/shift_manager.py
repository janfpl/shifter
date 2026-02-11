"""Stores and applies per-channel shift parameters.

Designed to be extensible — v1 stores integer translations only, but the
data structure can later accommodate scale factors and affine transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChannelTransform:
    """Transform parameters for a single channel.

    Attributes
    ----------
    channel_index : int
        Ordered index (0-based).
    filename : str
        Original filename associated with this channel.
    shift_x : int
        Translation in X (voxels). Positive = right.
    shift_y : int
        Translation in Y (voxels). Positive = down.
    shift_z : int
        Translation in Z (voxels). Positive = higher Z index.
    is_reference : bool
        If True, shifts are locked to (0, 0, 0).
    colormap : str
        napari colormap name for display.
    """

    channel_index: int = 0
    filename: str = ""
    shift_x: int = 0
    shift_y: int = 0
    shift_z: int = 0
    is_reference: bool = False
    colormap: str = "green"

    # -- Future extension slots (not used in v1) --
    # scale_x: float = 1.0
    # scale_y: float = 1.0
    # scale_z: float = 1.0

    @property
    def shift_zyx(self) -> tuple[int, int, int]:
        return (self.shift_z, self.shift_y, self.shift_x)

    @property
    def shift_yx(self) -> tuple[int, int]:
        return (self.shift_y, self.shift_x)

    def reset(self) -> None:
        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_index": self.channel_index,
            "filename_original": self.filename,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
            "shift_z": self.shift_z,
            "is_reference": self.is_reference,
        }


@dataclass
class ShiftManager:
    """Container for all channel transforms.

    Provides convenience methods for querying/modifying transforms and
    serializing to the metadata format expected by :mod:`utils`.
    """

    transforms: list[ChannelTransform] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def init_channels(
        self,
        filenames: list[str],
        reference_index: int,
        colormaps: list[str],
    ) -> None:
        """(Re-)initialize transforms for the given channel list."""
        self.transforms.clear()
        for i, fname in enumerate(filenames):
            self.transforms.append(
                ChannelTransform(
                    channel_index=i,
                    filename=fname,
                    is_reference=(i == reference_index),
                    colormap=colormaps[i] if i < len(colormaps) else "gray",
                )
            )

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, index: int) -> ChannelTransform:
        return self.transforms[index]

    @property
    def reference_index(self) -> int | None:
        for t in self.transforms:
            if t.is_reference:
                return t.channel_index
        return None

    def set_reference(self, index: int) -> None:
        for t in self.transforms:
            t.is_reference = (t.channel_index == index)
            if t.is_reference:
                t.reset()

    def set_shift(self, channel_index: int, axis: str, value: int) -> None:
        """Set a single axis shift for a channel. *axis* is 'x', 'y', or 'z'."""
        t = self.transforms[channel_index]
        if t.is_reference:
            return  # reference channel shifts are locked
        setattr(t, f"shift_{axis}", value)

    def reset_all(self) -> None:
        for t in self.transforms:
            t.reset()

    def has_any_shift(self) -> bool:
        return any(
            (t.shift_x != 0 or t.shift_y != 0 or t.shift_z != 0)
            for t in self.transforms
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_channel_dicts(self, output_suffix: str = "_corrected") -> list[dict[str, Any]]:
        """Return list of dicts suitable for :func:`utils.build_metadata`."""
        result = []
        for t in self.transforms:
            stem = t.filename.rsplit(".", 1)[0] if "." in t.filename else t.filename
            ext = t.filename.rsplit(".", 1)[1] if "." in t.filename else "tif"
            result.append(
                {
                    "filename_original": t.filename,
                    "filename_corrected": f"{stem}{output_suffix}.{ext}",
                    "channel_index": t.channel_index,
                    "shift_x": t.shift_x,
                    "shift_y": t.shift_y,
                    "shift_z": t.shift_z,
                }
            )
        return result
