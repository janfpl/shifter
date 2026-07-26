# Chromatic Shift Corrector

Napari-based axial and lateral chromatic shift correction for light sheet microscopy. Provides interactive 3D visualization, automatic registration, manual shift adjustment, and chunked full-volume export.

## Installation

Clone the repository and create a conda environment with Python 3.12, Qt, and numba:

```bash
git clone https://github.com/janfpl/shifter.git
cd shifter
conda create -n shifter python=3.12 pyqt numba -y
conda activate shifter
pip install -e .
```

> **Install `numba` — it is strongly recommended, not cosmetic.** It parallelises two
> hot paths across CPU cores. On a measured 431 GiB two-channel export, pyramid
> generation took **103 s per channel with numba versus 631 s without (6.1×)**, and
> mutual-information registration is likewise far slower on the pure-NumPy fallback.
> Results are bit-identical either way — only the speed differs.
>
> Install it via conda (as in the command above) because conda ships pre-built binaries
> for `numba` and its dependency `llvmlite`; installing via pip may fail on macOS and
> other platforms due to build toolchain incompatibilities. The pip extra
> `pip install -e ".[numba]"` also works but may require additional build dependencies.
>
> To confirm it is active in the environment you actually run the app from:
>
> ```bash
> conda activate shifter
> python -c "import numba; print(numba.__version__)"
> ```
>
> Every export also logs the backend in use — look for
> `Pyramid reduction backend: numba (N threads)` near the top of
> `performance_log.txt`. If it instead reads `numpy (single-threaded fallback …)`, the
> line names the underlying error.

For GPU acceleration (optional, requires **CUDA Toolkit 12.6**):

```bash
pip install -e ".[gpu]"
```

> **Note:** CUDA Toolkit **12.6** is the recommended and tested version. CUDA 10.x and 13.x are **not compatible**.

Requires Python 3.12.

## Usage

Launch the application:

```bash
python -m shifter
```

This opens a napari viewer with the Chromatic Shift Corrector widget docked on the right.

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| BigTIFF | `.tif`, `.tiff` | Single-channel 3D volumes, one file per channel |
| Luxendo H5 | `.lux.h5`, `.h5` | Flat HDF5 structure with optional resolution pyramids |

All data is loaded lazily via Dask arrays to avoid loading entire volumes into memory.

### Luxendo H5 Details

- Reads the `Data` dataset as the full-resolution volume
- Detects and displays resolution pyramid levels (`Data_W_H_D` naming convention) as napari multiscale layers
- Parses embedded JSON metadata for voxel sizes and channel descriptions
- On export, pyramids are regenerated for corrected volumes using block averaging (on by default; can be disabled — see Export)

## Workflow

### 1. Load Data

- Select the input format (BigTIFF or Luxendo H5)
- Choose a directory containing your channel files
- Select which files to load and assign channel order, reference channel, and colormaps
- For H5 files, voxel sizes are auto-populated from embedded metadata if available
- For BigTIFF directories containing a `.xml` sidecar, voxel sizes are extracted automatically

### 2. Register Channels

Draw a rectangle ROI on the napari viewer and specify a Z sub-range to define the registration volume. Select which channels to register against the reference, choose an algorithm (Mutual Information is the default), and run.

Results populate the shift table with X/Y/Z voxel shifts and a confidence score per channel. Confidence is color-coded in the table (green = high, red = low).

The progress bar advances at sub-channel resolution — for Mutual Information it moves through the coarse and fine search passes rather than jumping once per channel — and the status text shows which channel is being registered.

**Preprocessing options:**
- Background subtraction (percentile-based)
- Gaussian smoothing

When registration finishes it releases the working memory it allocated (several full-precision copies of the sub-volume) back to the OS — running a garbage collection, freeing GPU/pinned memory pools when the GPU path was used, and trimming the process heap. This keeps the resident footprint from lingering at its peak, which also matters for a subsequent export: slab sizing is based on *available* RAM, so memory the process is still hoarding would otherwise shrink the export's budget. The performance log records a before/after memory snapshot (written to the output directory if one is selected, otherwise the input data directory) so you can see how much was reclaimed.

### 3. Adjust Shifts

Shifts can be edited manually via spinboxes in the shift table. Use the preview button to visualize the corrected sub-volume in napari before committing to a full export.

### 4. Export

Select an output directory and RAM allocation (50-95% of system memory). Choose whether to export the **full volume** or **ROI only** (crops to the current ROI rectangle and Z range). The export streams corrected volumes in Z-slab chunks, writing one file per channel. Progress is reported in actual bytes written (not Z-planes), so for Luxendo H5 output the indicator keeps moving through pyramid regeneration instead of appearing to finish early. A `correction_metadata.json` sidecar is written alongside the output files containing all shift parameters, voxel sizes, processing details, and the total bytes written (`bytes_written_gb`). ROI exports include the crop bounds in the metadata and use a `_corrected_roi` filename suffix.

The number of Z-planes per slab is sized from the *currently available* system RAM (scaled by the RAM allocation slider) and capped so a single slab never exceeds a few GiB — reading and materializing one slab transiently holds several full-size copies at once (the source chunks, dask's concatenated buffer, and the output slab). This keeps peak memory bounded even for very large (hundreds-of-GB) volumes: export throughput is limited by disk I/O, not slab size, so batching more planes into one slab only increases memory pressure without exporting any faster. If an export runs out of memory, lower the RAM allocation slider, close other applications, or export a smaller ROI.

**Low-resolution pyramid layers** (Luxendo H5 only) are controlled by the *"Write low-resolution pyramid layers"* checkbox and are **on by default**. Every level is built from each corrected slab while it is still in memory — the written `Data` is **never read back** — and the reduction is parallelised across CPU cores, so the pyramid phase is now a modest addition rather than the dominant cost.

Measured on a 431 GiB two-channel export (3099 × 6979 × 5347, five pyramid levels, 32-core machine):

| Build | Pyramid compute / channel | Total export |
|---|---|---|
| Original (re-read `Data` once per level) | 3 h 52 m | **8 h 53 m** |
| Streaming, single-threaded numpy | 631 s | **44.8 m** |
| Streaming + numba (current) | **103 s** | **28.6 m** |
| Pyramids disabled (floor) | — | 21.3 m |

That is **18.7× faster than the original build**, and pyramids now cost ~7 min on top of the 21.3 min pyramids-off floor (they used to cost 23.5 min). Roughly 80% of the remaining runtime is disk I/O.

Untick the box for the fastest possible export when only full resolution is needed — the output `.lux.h5` then contains just `Data`, the size estimate and `bytes_written_gb` reflect full resolution only, and the companion headers are rewritten to describe a single resolution level.

Three properties make this exact and fast:

- Sums are accumulated as **integers** rather than `float64` (integer floor division of the block sum is identical to truncating the float mean for uint16 input).
- Where one level's factors divide another's (the usual 2/4/8 ladder), the coarser level is derived from the finer level's **unrounded sums** instead of from the full-resolution slab again.
- The XY reduction is **parallelised across CPU cores with numba** when it is installed. This is safe precisely because the sums are integers — addition is associative and commutative and the accumulator cannot overflow, so evaluation order does not change the result. **Without numba the reduction falls back to a single-threaded numpy path measured 6.1× slower** (results are identical either way), so check the log line `Pyramid reduction backend:` — it states which backend is in use and, when numba is missing, why.

Setting `CSC_PYRAMID_GPU=1` runs the XY reduction on the GPU via CuPy instead. This copies each slab to the device, so it only helps when the GPU is otherwise idle and the CPU is the bottleneck; numba is the better default. The chosen backend is recorded in `performance_log.txt` (`backend=numba|gpu|numpy`).

### CPU usage

Parallel work — pyramid generation, per-plane XY shifts, FFT-based registration, and the mutual-information grid search — uses **all logical cores except four**, which are left free so the OS and the napari UI stay responsive. The reservation is capped at half the machine, so smaller systems still get useful parallelism (32 cores → 28 workers, 16 → 12, 8 → 4, 4 → 2). On Linux the count respects the process's CPU affinity mask, so a restricted core set is honoured.

Two environment variables override this:

```bash
CSC_MAX_WORKERS=16     # use exactly this many worker threads
CSC_RESERVED_CORES=8   # leave this many cores free instead of 4
```

The count in effect is recorded at the top of `performance_log.txt` (`CPU cores: N available, using M worker threads`).

Companion Imaris (`.ims`) and BigDataViewer (`*_bdv.h5`) headers describe a *multi-resolution* dataset and link to the pyramid levels. With pyramids **on** they are copied verbatim; with pyramids **off** they are **rewritten to a single (full-resolution) level** so they still resolve against the pyramid-less output — e.g. for import into the Imaris File Converter, which builds its own pyramids from the full-resolution data. (Copying the original multi-resolution headers next to pyramid-less data would make Imaris/BigDataViewer read the dataset as corrupt.)

To repair a folder that was exported by an older build (multi-resolution headers copied next to pyramid-less data), reduce the headers in place without re-exporting:

```bash
python -m shifter.fix_headers /path/to/export_folder
```

### Export diagnostics

Every export writes a `performance_log.txt` into the output directory with timestamped start/end markers and elapsed times for each phase. By default the log is written at **DEBUG** level, which also records the chunk-size decision (and which limit bound it), a memory snapshot at export start, and a per-slab line with timing, throughput (MiB/s), and memory usage — useful for tracking down slow or memory-hungry exports. The log is rewritten from scratch on each export (it does not accumulate across runs), and the per-slab overhead is negligible against the disk I/O each slab performs.

To keep only the INFO-level phase markers and suppress the extra detail, set `CSC_DEBUG` to a falsy value before launching:

```bash
# Windows (cmd) — disable the extra debug detail
set CSC_DEBUG=0
python -m shifter

# macOS / Linux
CSC_DEBUG=0 python -m shifter
```

Setting `CSC_DEBUG=1` (or leaving it unset) keeps the debug diagnostics on.

Output format matches the input format:
- BigTIFF input produces BigTIFF output, using a `_corrected` filename suffix
- Luxendo H5 input produces H5 output with preserved metadata and, when the pyramid checkbox is enabled, regenerated resolution pyramids

**Luxendo H5 full-volume exports keep the original filenames unchanged** (no suffix), so that companion Imaris/BigDataViewer header files continue to work. If the input directory contains an Imaris `.ims` header and/or a BigDataViewer `*_bdv.h5` / `*_bdv.xml` pair (these reference the per-channel `.lux.h5` files by their literal filenames via HDF5 external links / relative XML paths), they are written into the output directory alongside the corrected data (copied verbatim with pyramids on, or reduced to a single level with pyramids off — see above).

**ROI exports** use the `_corrected_roi` suffix and now also get companion headers, **regenerated** for the crop: the `.ims` / `*_bdv.h5` external links are repointed to the `_corrected_roi` files, and the Imaris `.ims` is given the ROI's voxel dimensions and a cropped physical extent (voxel size preserved). The BigDataViewer `*_bdv.xml` is regenerated too — see below. As with full-volume, pyramid levels are included only when the pyramid checkbox is ticked.

### How links are repointed, and how headers are checked

Every rewritten header must point its external links at the **corrected output**. That target is taken from a mapping the export records as it writes each channel (source path → output filename), not inferred from the old filename. The difference matters as soon as links are not bare filenames: a real acquisition nests them, e.g. `raw/stack_1_channel_0-561_obj_left/Cam_left_00000.lux.h5`. Carried unchanged into a flat output folder, such a link either dangles — loud and harmless — or **resolves back to the original raw data**, and the corrected export silently displays uncorrected pixels. Lookup tries the exact path, then normalised separators, then the path resolved against the header's own directory, then the bare basename — but only if that basename is unique across the sources. Two channels ending in the same filename are refused with a clear message rather than guessed, and a link that cannot be mapped fails its header instead of being left pointing at the source.

Headers are then **validated by dereference**, not by inspection: an `h5py.ExternalLink` is only a `(filename, dataset path)` pair and proves nothing about what is on the other end. Validation opens each header, follows every link from the header's own directory exactly as a viewer would, and checks that the target exists, holds a 3-D dataset at the named path, and has the shape the header claims — per level, for the `.ims` `ImageSizeX/Y/Z` and `DataSetInfo/Image`, and for the `*_bdv.h5` that each level's shape equals the full shape divided by that level's own downsample factors and that `resolutions` never advertises a level that is not there. Imaris text attributes are confirmed to be `uint8`.

This runs at the very end of the export, since links cannot resolve until the data files are closed. Nothing is reported as a written header before it passes, and the result is recorded in `correction_metadata.json` under `companion_header_validation`. If validation fails the image data is kept untouched, the bad headers are renamed to `*.invalid` so no viewer will open them, and the export raises with the specific problems listed. A header that looks fine but points somewhere wrong is never reported as good.

One consequence worth knowing: with pyramids **on** and a full volume the headers are still copied verbatim, because they already describe exactly what was written — but if that copy then fails validation (which is what a nested link produces), it is discarded and the header is rebuilt through the recorded mapping instead.

### BigDataViewer XML

A BigDataViewer dataset is a **pair**: the `*_bdv.h5` holds the resolution tables and links to the pixel data, and the `*_bdv.xml` is the entry point Fiji/BigDataViewer actually opens. The two are therefore written together or not at all — an `.h5` without its `.xml` cannot be opened, so it is never left behind on its own.

The XML is rewritten from the source rather than authored from scratch; only what is unambiguous is touched and everything else (voxel size, channel/tile/angle attributes, timepoints, extra transforms) is carried through verbatim:

* **`ViewSetup/size`** is set to the exported voxel dimensions — unchanged for a full-volume export, the crop's `X Y Z` for an ROI. `voxelSize` is left alone: a crop resamples nothing.
* **`ImageLoader/hdf5`** is repointed at the `*_bdv.h5` as written, as a bare relative name, so a source that referenced its H5 by a nested path cannot end up pointing back at the original raw data.
* **`ViewRegistration`** gains a pure-translation `ViewTransform` of `(x0, y0, z0)` voxels for an ROI, so the crop keeps its true position inside the specimen instead of being drawn at the full volume's origin. The offset is in voxel units and is appended as the *last* transform, which is the one SpimData applies *first* — matching the Luxendo spec's `affine_to_sample` convention, where the transform maps image (voxel) coordinates into sample space and the voxel size is treated as `(1, 1, 1)`. The composition is verified numerically before the file is committed.

If any of that cannot be done safely — the document is not SpimData, has no `ViewSetup`/`ViewRegistration` entries, or declares a size that does not match the exported volume — **neither** file is written and the reason is logged. A header that looks plausible but misplaces the data is worse than no header.

The rewrite is checked against a genuine Luxendo BDV XML (`shifter/tests/data/main_st-0-x00-y00-0-x00-y01_bdv.xml`, a two-channel 3099 × 6979 × 5347 acquisition) as well as synthetic fixtures. In that sample the registration *is* the calibration (2.925 × 2.925 × 3 µm per voxel), which pins the transform ordering down: cropping at voxel (1000, 2000, 500) moves the world origin by 2925 / 5850 / 1500 µm, not by 1000 / 2000 / 500. Getting that backwards would displace a crop by roughly 2 mm while still producing a volume that looks entirely reasonable, so it is covered by a test.

## Registration Algorithms

Three algorithms are available for automatic shift detection. All operate on integer voxel shifts and support configurable XY and Z search ranges.

### Phase Cross-Correlation

FFT-based phase correlation using `skimage.registration.phase_cross_correlation`.

| Aspect | Detail |
|--------|--------|
| Speed | Fast |
| Best for | High-SNR data with similar intensity distributions across channels |
| Limitations | Sensitive to noise; can produce spurious results on low-contrast data |
| Parameters | Normalization mode (`phase` or `None`) |
| GPU | Supported via CuPy |

Use `normalization=None` when channels have very different intensity profiles or when the default `phase` normalization produces unreliable results.

### Zero-Normalized Cross-Correlation (ZNCC)

Normalizes both volumes to zero mean and unit variance before FFT-based cross-correlation. Confidence is derived directly from the ZNCC peak value.

| Aspect | Detail |
|--------|--------|
| Speed | Fast |
| Best for | General-purpose use; robust across varying intensity levels |
| Limitations | Assumes linear intensity relationship between channels |
| Parameters | None (beyond search range) |
| GPU | Supported via CuPy |

A fast, robust general-purpose option when channels have similar intensity profiles.

### Mutual Information

Coarse-to-fine exhaustive search maximizing mutual information via joint histograms. Coarse pass uses a step size of 5 voxels; fine pass refines within a 5-voxel radius.

| Aspect | Detail |
|--------|--------|
| Speed | Slow (exhaustive search over 3D shift space) |
| Best for | Channels with non-linear intensity relationships (e.g., different fluorophores, modalities) |
| Limitations | Significantly slower than FFT-based methods |
| Parameters | None (beyond search range) |
| GPU | Supported via CuPy (accelerates histogram computation) |

This is the default algorithm. It is the most robust across dissimilar intensity distributions between channels (e.g. different fluorophores), at the cost of speed; install `numba` (recommended) for a large parallel speed-up.

## GPU Acceleration

All registration algorithms support optional GPU acceleration via CuPy. The widget displays the detected GPU name or indicates CPU-only mode. If a GPU computation fails (e.g., out of memory), it falls back to CPU automatically.

Install GPU support:

```bash
pip install -e ".[gpu]"
```

Requires CUDA Toolkit 12.6 and compatible hardware/drivers. CUDA 10.x and 13.x are not supported.

**Troubleshooting: GPU not detected**

If the widget shows CPU-only mode despite having a CUDA-capable GPU, the `CUDA_PATH` environment variable may not be visible inside your conda environment. Verify by running:

```cmd
echo %CUDA_PATH%
```

If this prints nothing, set it manually for your session:

```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

To make this permanent for the conda environment, create an activation script:

```cmd
mkdir "%CONDA_PREFIX%\etc\conda\activate.d"
echo set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6 > "%CONDA_PREFIX%\etc\conda\activate.d\cuda_path.bat"
```

Ensure the path points to your CUDA 12.6 installation.

## Dependencies

- napari (>=0.4.18)
- tifffile (>=2023.2.3)
- dask (>=2023.1.0)
- numpy (>=1.23)
- scipy (>=1.10)
- scikit-image (>=0.20)
- h5py (>=3.7)
- psutil (>=5.9)
- qtpy (>=2.3)
- matplotlib (>=3.5)
- numba (recommended: parallelises pyramid generation and mutual-information registration; install via conda)
- CuPy (optional, for GPU acceleration)
