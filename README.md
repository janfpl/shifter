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

**ROI exports** use the `_corrected_roi` suffix and now also get companion headers, **regenerated** for the crop: the `.ims` / `*_bdv.h5` external links are repointed to the `_corrected_roi` files, and the Imaris `.ims` is given the ROI's voxel dimensions and a cropped physical extent (voxel size preserved). The BigDataViewer `*_bdv.xml` is not regenerated for ROI (its dimensions can't be rewritten reliably here); Imaris — which uses the `.ims` — is unaffected. As with full-volume, pyramid levels are included only when the pyramid checkbox is ticked.

## Registration Algorithms

Six algorithms are available for automatic shift detection. All operate on integer voxel shifts and support configurable XY and Z search ranges.

Every algorithm estimates **one global integer translation per channel** — Shifter corrects rigid chromatic shift, not local deformation. Where an algorithm is named after an upstream package that does more than that (currently deedsBCV), only the part that produces a translation is implemented; the per-algorithm scope notes below say exactly what was and was not taken from the original.

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

### Mutual Information (Brent)

The same mutual-information metric as above, but the exhaustive *fine* search is replaced with **Brent's method** — the bounded one-dimensional optimizer from `scipy.optimize.minimize_scalar` (`method="bounded"`), applied per axis in a cyclic coordinate-descent loop. A cheap coarse grid pass (step 5) first locates the correct basin — mutual information is multimodal over a translation, so a purely local optimizer would otherwise get trapped — and Brent then refines it. The integer part of each candidate shift is evaluated by exact overlap slicing (as in the grid method); the sub-voxel remainder is applied by linear interpolation so Brent sees a smooth objective, and the converged shift is rounded to the nearest voxel.

| Aspect | Detail |
|--------|--------|
| Speed | Faster than grid Mutual Information — Brent reaches the optimum in far fewer metric evaluations than the exhaustive fine grid (roughly 5–8× faster in practice) |
| Best for | The mutual-information use case (dissimilar/non-linear intensity relationships) when the exhaustive fine grid is unnecessarily slow |
| Limitations | Local refinement — relies on the coarse pass to seed the right basin; result is still integer-rounded |
| Parameters | None (beyond search range) |
| GPU | Not used — Brent is a sequential optimizer, so this method runs on CPU regardless of the GPU toggle |

Recovers the same shifts as grid Mutual Information on well-structured data, at a fraction of the run time; install `numba` for a fast coarse pass.

### deedsBCV (MIND-SSC)

> **Scope: translation only — this is not the full deedsBCV.** deedsBCV proper is a *deformable* registration: dense discrete displacements on a control-point grid, regularized with a minimum spanning tree. Shifter applies one global integer shift per channel, so what is implemented here is deeds' similarity core — the MIND-SSC descriptor plus the discrete data-cost search — in the translation-only role that `linearBCV` plays before deeds' deformable pass. The regularization and the deformable field are not implemented, as this pipeline has nowhere to apply them. Expect deeds-quality *shift detection*, not deeds-quality non-rigid alignment: local warping, and any residual misalignment that varies across the field of view, is out of reach for this (and every other) algorithm in Shifter.

Registration on **MIND-SSC** descriptors — the modality-independent self-similarity descriptor from [deedsBCV](https://github.com/mattiaspaul/deedsBCV) (Mattias P. Heinrich, MIT-licensed). Each voxel is described by 12 values measuring how its local patch differs from patches at neighbouring offsets, so the descriptor encodes *structure* rather than intensity: an arbitrary brightness/contrast change leaves it unchanged. Shifts are then found by a discrete displacement search over a 4× / 2× / 1× downsampling pyramid, minimizing the descriptor sum-of-squared-differences over a strided grid of sample points.

| Aspect | Detail |
|--------|--------|
| Speed | Moderate (pyramid search; roughly a few seconds per channel for a 160³ ROI) |
| Best for | Channels whose intensity relationship is non-linear or inverted — the mutual-information use case, at a fraction of the cost |
| Limitations | Translation only (see the scope note above); descriptor computation holds two 12-channel `float32` volumes in RAM |
| Parameters | Descriptor quantisation step (default 1) and refinement radius (default 3) |
| GPU | Supported via CuPy (whole pipeline, descriptors and search) |

The descriptors follow `src/MINDSSCbox.h` of the reference implementation, with two deviations: descriptor entries are kept as `float32` (the `exp(-x)` form the reference leaves commented out) and compared by SSD rather than quantized into a 64-bit word and compared by Hamming distance, and box filtering uses a mean rather than a running sum — the constant cancels in the per-voxel noise normalization that follows.

Confidence is how far the best candidate stands out from the coarsest level's cost distribution, `(median − min) / (max − min)`, the minimization counterpart of the mutual-information confidence.

### deedsBCV (MIND-SSC, Brent)

The MIND-SSC descriptor above, but the finer pyramid grid searches are replaced with **Brent's method** — the same bounded per-axis optimizer used by Mutual Information (Brent). Descriptors are computed once at full resolution; the coarsest pyramid level grid-searches the whole range for a seed, then Brent refines it, evaluating the descriptor cost at continuous (sub-voxel) shifts by linear interpolation of the descriptor field (`scipy.ndimage.map_coordinates`) so the objective is smooth. The converged shift is rounded to the nearest voxel.

| Aspect | Detail |
|--------|--------|
| Speed | Comparable to grid deedsBCV — the descriptor grid search is already cheap, so Brent is a modest saving rather than a large one (unlike the Mutual Information pair, where it replaces an expensive fine grid) |
| Best for | The MIND-SSC use case when you specifically want a gradient-free continuous optimizer over the descriptor cost rather than a discrete grid |
| Limitations | Local refinement seeded by the coarse pyramid level; translation-only (as with grid deedsBCV); result is integer-rounded |
| Parameters | Descriptor quantisation step (default 1) |
| GPU | Not used — Brent is a sequential optimizer, so this method runs on CPU regardless of the GPU toggle |

Included mainly to complete the pairing of both similarity metrics (mutual information and MIND-SSC) with both search strategies (grid and Brent). For MIND-SSC the grid search is already fast, so the grid variant remains the better default; the speed win from Brent is real for Mutual Information, where the exhaustive fine grid is the bottleneck.

## GPU Acceleration

All registration algorithms support optional GPU acceleration via CuPy. The widget displays the detected GPU name or indicates CPU-only mode. If a GPU computation fails (e.g., out of memory), it falls back to CPU automatically.

On startup the app checks for a usable GPU by JIT-compiling a small test kernel with CuPy/NVRTC. Because a CuPy build that does not match the installed CUDA driver/toolkit can make that compile fault at the native level (a Windows *access violation*), the check runs in a **separate subprocess** — if it crashes, the app reports CPU mode and keeps running rather than going down with it.

The probe is attempted twice, each in its own subprocess. The first attempt is **isolated**: it removes any system CUDA-toolkit directory from `PATH` so CuPy loads only its own bundled CUDA libraries. This fixes the most common failure on Windows, where a system CUDA toolkit's `nvrtc` DLL (already on `PATH` from the CUDA installer) shadows the different-version one bundled with `cupy-cuda12x` and crashes the compile. The second attempt injects the **system** CUDA toolkit path, for setups (e.g. conda `cudatoolkit`) whose CuPy relies on the system libraries. Whichever succeeds, that same `PATH` arrangement is applied to the app process for the real GPU work. Two environment variables control the check:

- `SHIFTER_DISABLE_GPU=1` — skip the GPU probe entirely and run on CPU (fastest startup; use this if the probe is slow or unreliable on your machine).
- `SHIFTER_GPU_PROBE=inprocess` — run the probe in-process (the old behaviour), for debugging only; a native CuPy fault will crash the app.

Install GPU support:

```bash
pip install -e ".[gpu]"
```

Any **CUDA 12.x** runtime is supported (this is what `cupy-cuda12x` targets); the app is tested against CUDA 12.6. The CuPy wheel bundles its own CUDA 12.x libraries, so it may report a runtime version (e.g. 12.9) different from a separately installed toolkit — that is expected and fine. CUDA 11.x and 13.x are not supported.

**Troubleshooting: GPU shows CPU mode with a "could not compile a test kernel" banner**

If the startup banner reports that CuPy could not compile a test kernel on an otherwise-supported CUDA 12.x runtime, a system CUDA toolkit's `nvrtc` DLL is most likely shadowing CuPy's bundled one. Try, in order:

1. Refresh the bundled CUDA libraries: `pip install -U cupy-cuda12x`.
2. Update your NVIDIA driver.
3. Diagnose directly with the probe, which takes a `--strategy` and prints its JSON result (and, thanks to faulthandler, any native crash stack):

   ```cmd
   python -m shifter.registration._gpu_probe --strategy isolated
   python -m shifter.registration._gpu_probe --strategy system
   python -m shifter.registration._gpu_probe --strategy bundled
   ```

   `isolated` is what the app tries first (it drops system CUDA-toolkit dirs from `PATH` so CuPy uses its bundled libraries); `bundled` leaves `PATH` untouched. If `isolated` prints `"available": true`, the app will use the GPU on the next launch.

**Troubleshooting: GPU not detected (conda-installed CUDA toolkit)**

If you rely on a conda-installed `cudatoolkit` rather than the bundled CuPy libraries, and the widget shows CPU-only mode, the `CUDA_PATH` environment variable may not be visible inside your conda environment. Verify by running:

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

**Troubleshooting: app closes immediately on startup**

The GPU probe now runs out-of-process, so a native CuPy/NVRTC crash should no longer take the app down — it falls back to CPU and prints a banner explaining why. If you are on an older build, or the app still exits during "Building Chromatic Shift Corrector widget" with a *Windows fatal exception: access violation* traceback pointing into `cupy`/NVRTC, start with the probe disabled:

```cmd
set SHIFTER_DISABLE_GPU=1
python -m shifter
```

The app will run on CPU. To restore GPU acceleration, reinstall CuPy to match your CUDA version (`pip install cupy-cuda12x` for CUDA 12.x) and update your NVIDIA driver, then unset the variable.

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
