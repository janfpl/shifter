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

> **Note:** Installing `numba` via conda is recommended because it provides pre-built
> binaries for `numba` and its dependency `llvmlite`. Installing these via pip may fail
> on macOS and other platforms due to build toolchain incompatibilities. If you skip the
> conda numba install, mutual-information registration will still work but will use a
> slower pure-NumPy fallback. You can also install numba as a pip extra with
> `pip install -e ".[numba]"`, though this may require additional build dependencies.

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
- On export, pyramids can optionally be regenerated for corrected volumes using block averaging (off by default — see Export)

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

**Low-resolution pyramid layers** (Luxendo H5 only) are controlled by the *"Write low-resolution pyramid layers"* checkbox and are **off by default**. When enabled, every level is built from each corrected slab while it is still in memory — the written `Data` is **never read back** — so the pyramid phase costs a little extra CPU rather than a full re-read of the volume per level. (The original implementation re-read the whole corrected volume once per level, which on a ~215 GiB/channel volume was ~85% of total export time; on that dataset the rewrite took a full export from 8h 53m to 44m.) When disabled, the output `.lux.h5` contains only the full-resolution `Data`, and the size estimate / `bytes_written_gb` reflect full-resolution only.

Three properties make this exact and fast:

- Sums are accumulated as **integers** rather than `float64` (integer floor division of the block sum is identical to truncating the float mean for uint16 input).
- Where one level's factors divide another's (the usual 2/4/8 ladder), the coarser level is derived from the finer level's **unrounded sums** instead of from the full-resolution slab again.
- The XY reduction is **parallelised across CPU cores with numba** when it is installed. This is safe precisely because the sums are integers — addition is associative and commutative and the accumulator cannot overflow, so evaluation order does not change the result. Install numba (see above) for a multi-core speed-up; without it the code falls back to numpy and results are unchanged.

Setting `CSC_PYRAMID_GPU=1` runs the XY reduction on the GPU via CuPy instead. This copies each slab to the device, so it only helps when the GPU is otherwise idle and the CPU is the bottleneck; numba is the better default. The chosen backend is recorded in `performance_log.txt` (`backend=numba|gpu|numpy`).

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

**ROI exports** use the `_corrected_roi` suffix and now also get companion headers, **regenerated** for the crop: the `.ims` / `*_bdv.h5` external links are repointed to the `_corrected_roi` files, and the Imaris `.ims` is given the ROI's voxel dimensions and a cropped physical extent (voxel size preserved). The BigDataViewer `*_bdv.xml` is not regenerated for ROI (its dimensions can't be rewritten reliably here); Imaris — which uses the `.ims` — is unaffected. As with full-volume, pyramid levels are included only when the pyramid checkbox is ticked.

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
