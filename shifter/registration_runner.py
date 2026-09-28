"""Qt-free registration loop shared by the GUI worker and headless mode."""

from __future__ import annotations

import logging
from typing import Any, Callable

from shifter.perf_logger import log_event, log_memory, timed_operation
from shifter.preview_engine import extract_subvolume
from shifter.registration import ALGORITHM_REGISTRY, RegistrationResult, preprocess


def register_channels(
    loaders: list[Any],
    reference_index: int,
    channels_to_register: list[int],
    algorithm_name: str,
    algorithm_kwargs: dict,
    search_range_xy: int,
    search_range_z: int,
    roi_bounds: tuple[int, int, int, int],
    z_start: int,
    z_end: int,
    background_subtraction: bool,
    gaussian_smoothing: bool,
    use_gpu: bool,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[tuple[int, RegistrationResult]]:
    """Register each channel in *channels_to_register* against the reference.

    Parameters
    ----------
    roi_bounds : tuple
        (y_start, y_end, x_start, x_end) of the registration sub-volume.
    z_start, z_end : int
        Z range of the sub-volume (inclusive start, exclusive end).
    progress : callable, optional
        Called as ``progress(done, total, description)``.

    Returns
    -------
    list of (channel_index, RegistrationResult)
    """
    y_start, y_end, x_start, x_end = roi_bounds

    log_event(f"Registration started | algo={algorithm_name} "
              f"channels={channels_to_register} "
              f"search_xy={search_range_xy} search_z={search_range_z} "
              f"gpu={use_gpu}")
    log_memory("registration start", level=logging.INFO)

    # Extract reference sub-volume.
    ref_loader = loaders[reference_index]
    ref_vol = extract_subvolume(
        ref_loader.dask_array,
        z_start, z_end,
        y_start, y_end, x_start, x_end,
    )

    # Preprocess reference.
    ref_vol = preprocess(
        ref_vol,
        background_subtraction=background_subtraction,
        gaussian_smoothing=gaussian_smoothing,
        use_gpu=use_gpu,
    )

    # Instantiate algorithm.
    algo_cls = ALGORITHM_REGISTRY[algorithm_name]
    algo = algo_cls(**algorithm_kwargs)

    n = len(channels_to_register)
    results = []

    # Progress is tracked at sub-channel resolution: each channel spans one
    # unit, and the algorithm reports a fraction within it, so the bar keeps
    # moving during a single (possibly long) mutual-information search rather
    # than jumping once per channel. ``scale`` gives the bar smooth steps.
    scale = 1000

    def _emit(idx: int, frac: float) -> None:
        if progress is None:
            return
        frac = min(1.0, max(0.0, frac))
        progress(
            int((idx + frac) * scale),
            n * scale,
            f"Registering channel {idx + 1} of {n}...",
        )

    for idx, ch_i in enumerate(channels_to_register):
        loader = loaders[ch_i]
        _emit(idx, 0.0)

        # Extract moving sub-volume.
        mov_vol = extract_subvolume(
            loader.dask_array,
            z_start, z_end,
            y_start, y_end, x_start, x_end,
        )

        # Preprocess moving.
        mov_vol = preprocess(
            mov_vol,
            background_subtraction=background_subtraction,
            gaussian_smoothing=gaussian_smoothing,
            use_gpu=use_gpu,
        )

        # Advance the bar within this channel as the algorithm searches.
        channel_cb = lambda frac, _idx=idx: _emit(_idx, frac)

        # Run registration with GPU OOM fallback.
        with timed_operation(f"Registration channel {ch_i} ({algorithm_name})"):
            try:
                result = algo.register(
                    ref_vol, mov_vol,
                    search_range_xy, search_range_z,
                    use_gpu=use_gpu,
                    progress_callback=channel_cb,
                )
            except Exception:
                # If GPU fails (e.g. OOM), retry on CPU.
                if use_gpu:
                    result = algo.register(
                        ref_vol, mov_vol,
                        search_range_xy, search_range_z,
                        use_gpu=False,
                        progress_callback=channel_cb,
                    )
                else:
                    raise

        _emit(idx, 1.0)
        log_event(f"Registration channel {ch_i} result: "
                  f"shift=({result.shift_z},{result.shift_y},{result.shift_x}) "
                  f"confidence={result.confidence:.3f}")
        results.append((ch_i, result))

    if progress is not None:
        progress(n * scale, n * scale, "Registration complete.")
    return results
