# Chromatic Shift Corrector

Napari-based axial and lateral chromatic shift correction for light sheet microscopy. Provides interactive 3D visualization, automatic registration, manual shift adjustment, and chunked full-volume export.

## Installation

Requires Python 3.12. Clone the repository and create a conda environment:

```bash
git clone https://github.com/janfpl/shifter.git
cd shifter
conda create -n shifter python=3.12 pyqt numba -y
conda activate shifter
pip install -e .
```

`numba` is strongly recommended (it is several times faster for pyramid generation and mutual-information registration). Install it via conda as above; pip builds can fail on some platforms.

Optional GPU acceleration, matching the `CUDA Version` that `nvidia-smi` reports (install only one):

```bash
pip install -e ".[gpu-cuda12]"   # CUDA 12.x (tested with 12.6)
pip install -e ".[gpu-cuda13]"   # CUDA 13.x (NVIDIA driver R580 or newer)
```

To switch, `pip uninstall -y` the other CuPy build (`cupy-cuda12x` or `cupy-cuda13x`) first. Step-by-step Windows instructions (ZIP download instead of git, CUDA Toolkit, `CUDA_PATH`, checking the GPU) are in [Detailed installation (Windows)](#detailed-installation-windows).

## Usage

Open a new Anaconda Prompt (or terminal), activate the environment, go to the install folder and launch the application:

```bash
conda activate shifter
cd path/to/shifter
python -m shifter
```

This opens a napari viewer with the Chromatic Shift Corrector widget docked on the right.

### One-click launch on Windows

Double-click `launch_shifter.bat` in the repository folder. It activates the conda
environment and starts shifter, so users don't need to open Anaconda Prompt. To make a
desktop shortcut, right-click the file → *Send to* → *Desktop (create shortcut)*.

It should work unchanged if the environment is called `shifter` and conda is on `PATH`
or in a standard install location. Otherwise, open the file in Notepad and edit the two
settings at the top:

| Setting | What to put there | How to find it |
|---------|-------------------|----------------|
| `ENV_NAME` | Environment name (default `shifter`) or full path to it | `conda env list` in Anaconda Prompt |
| `CONDA_ROOT` | Anaconda/Miniconda install folder, e.g. `C:\Users\you\anaconda3` (leave empty to auto-detect) | `where conda` in Anaconda Prompt; the root is the folder above `condabin` or `Scripts` |

If something goes wrong, the window stays open and shows the error.

### Headless batch processing (no napari window)

`headless_process.bat` registers and exports one or more Luxendo `.lux.h5` folders
without opening napari. Drag a data folder (or several) onto it, drag a `.txt` file
listing folders onto it, or double-click it and paste a path. In a `.txt` list, separate
folders with commas and/or new lines. Lines starting with `#` are ignored, and relative
paths are resolved against the `.txt` file's folder. The settings it uses are described
under [Changing the headless defaults](#changing-the-headless-defaults) below.

For each folder, one after another:

1. **Load** every `.lux.h5` channel. Companion `.ims` / `*_bdv.h5` headers and `main*`
   files are skipped.
2. **Reference = brightest channel.** This is the highest mean intensity, measured on
   each channel's coarsest pyramid level (or on the registration ROI if a channel has no
   pyramids).
3. **Automatic ROI.** The same as the GUI's *Add registration ROI* default: full X width,
   2 voxels in Y at the Y midpoint, full Z depth.
4. **Mutual Information** registration of every other channel against the reference.
   The shifts are applied without prompting. Channels whose shift hits the search limit,
   or whose confidence is low, are flagged in the console but still exported.
5. **Export** full-volume corrected `.lux.h5` files: original filenames, regenerated
   pyramids, and companion headers. They go into a new folder next to the source,
   `<folder>_MMDDYY_HHMM_shifted` (e.g. `sample1_092826_1430_shifted`). The name gets a
   `_2`, `_3`, … suffix rather than overwriting an existing folder.

`correction_metadata.json` in the output also records the reference choice, channel
brightness, ROI, and per-channel shifts and confidence (`headless_registration`).
`performance_log.txt` covers both registration and export. If one folder fails, the rest
still run, and the summary at the end lists each folder's result. A folder with only one
channel is skipped (nothing is written) and isn't counted as a failure.

The same thing from a terminal:

```bash
python -m shifter.headless D:\data\sample1 D:\data\sample2
python -m shifter.headless folders.txt --xy-range 0 --z-range 90
python -m shifter.headless --help
```

#### Changing the headless defaults

Open `headless_process.bat` in Notepad (right-click → *Edit*) and change the lines in the
**USER SETTINGS** block at the top. Keep the `set "NAME=value"` form, with no spaces
around `=`, and save. The new values apply from the next run.

| Setting | Default | What it does | Command-line equivalent |
|---------|---------|--------------|-------------------------|
| `ENV_NAME` | `shifter` | Conda environment to activate (name or full path) | — |
| `CONDA_ROOT` | empty (auto-detect) | Anaconda/Miniconda install folder, see [One-click launch](#one-click-launch-on-windows) | — |
| `XY_RANGE` | `0` | Registration search range in X and Y, in voxels. `0` means no XY search: only the Z shift is registered and X/Y stay at 0. Raise it (e.g. `1`–`5`) if the channels are also offset laterally | `--xy-range` |
| `Z_RANGE` | `90` | Registration search range in Z, in voxels. The stack must be at least 2 × `Z_RANGE` planes deep; a folder that isn't fails with a message saying so | `--z-range` |
| `ROI_Y` | `2` | Height of the automatic registration ROI in Y, in voxels. Must be at least 2 × `XY_RANGE`, so raise it when you raise `XY_RANGE` | `--roi-y` |
| `RAM_PERCENT` | `90` | Percent of system RAM the export may use. Lower it if other programs need memory during a run | `--ram` |
| `PYRAMIDS` | `yes` | `no` skips writing low-resolution pyramid layers (faster export, but viewers such as Imaris or BigDataViewer lose their overview levels) | `--no-pyramids` |

For example, a wider search with a taller ROI and no pyramids:

```bat
set "XY_RANGE=5"
set "Z_RANGE=120"
set "ROI_Y=16"
set "PYRAMIDS=no"
```

To keep several configurations, copy the file (e.g. `headless_process_wide.bat`) and edit
the copy; each copy has its own settings.

Other environment variables can go in the same block, e.g. `set "SHIFTER_DISABLE_GPU=1"`
to run on the CPU only, or `set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"`
if `CUDA_PATH` isn't configured in the conda environment ([Detailed installation step 3](#3-point-the-conda-environment-at-the-cuda-toolkit-cuda_path)).

When running `python -m shifter.headless` from a terminal, pass the options shown in the
last column; anything you leave out uses the built-in default. The built-in defaults are
the `DEFAULT_*` constants at the top of `shifter/headless.py`. The `.bat` file always
passes its own values, so editing the `.bat` is enough for double-click use.
The GUI takes its starting values from the same constants: `DEFAULT_SEARCH_XY` and
`DEFAULT_SEARCH_Z` for the search range, and `DEFAULT_ROI_Y` for the *Add registration ROI*
height. Changing them there changes both the GUI and terminal runs.

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

Draw a rectangle ROI on the napari viewer and specify a Z sub-range to define the registration volume. Or click **Add registration ROI** to add a centred ROI and matching Z range; by default it spans the full X width, 2 voxels in Y at the midpoint, and the full Z depth, and the sizes can be changed under *Registration ROI size*. Select which channels to register against the reference, choose an algorithm (Mutual Information is the default), and run.

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

Coarse-to-fine exhaustive search maximizing mutual information via joint histograms. Coarse pass uses a step size of 5 voxels; fine pass refines within a 5-voxel radius. Both are capped per axis at that axis' search range, so a small range (e.g. 1 voxel in XY) is searched exhaustively in both directions.

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

## Deformable Registration (deedsBCV)

The six algorithms above all estimate a single **global integer translation** per channel. For **local, non-rigid** chromatic distortion that a global shift cannot correct, there is a separate **deformable** path: a faithful numpy/cupy port of the full deedsBCV algorithm that solves a dense, sub-voxel **displacement field** and warps the moving channel with trilinear interpolation.

It is deliberately kept **off** the integer transform model — the shift table, the manual spin boxes, and the normal "Apply & Export" all stay integer-only and untouched. Two dedicated buttons in the Export section drive it for the registration-selected channels over the drawn ROI:

- **"Preview Deformable (deedsBCV, ROI)"** solves the field and adds the corrected ROI channels to the napari viewer as layers (nothing is written), for visual QC — the reference channel is shown alongside for comparison.
- **"Export Deformable (deedsBCV, ROI)"** does the same solve and writes corrected BigTIFF volumes directly (the reference and unselected channels are written unchanged; output is always `.tif`, even for H5 inputs).

The solver ports the reference pipeline (`deedsBCV0.cpp`): a 5-level control-point grid (`grid_spacing = 8,7,6,5,4`), MIND-SSC descriptors, a discrete per-control-point data cost over a `(2·L+1)^3` displacement label space, minimum-spanning-tree **belief-propagation** regularization (`primsMST` + separable squared-L2 `messageDT`), and forward+backward **symmetric inverse-consistent** composition (`consistentMappingCL`). The displacement field warps the moving volume as `corrected(p) = moving(p + field(p))`.

| Aspect | Detail |
|--------|--------|
| Output | Warped corrected BigTIFF volumes for the ROI (a dense field per channel, applied with sub-voxel interpolation) |
| Best for | Local, spatially-varying misalignment that a single translation leaves behind |
| Scope (v1) | ROI / downsampled volumes held in RAM. Uses the GPU (CuPy) when available with automatic per-solve CPU fallback. Native full-size slab+halo streaming warp is a planned follow-up |
| Fidelity | Faithful to the reference apart from the inherited descriptor deviation (float32 `exp(-x)` MIND-SSC + SSD instead of quantised popcount Hamming), and no affine pre-stage — assumes roughly pre-aligned inputs |

On synthetic data with a known smooth deformation, the full schedule reduces fixed↔moving SSD by ~98%. Because it solves a per-voxel field over a multi-level label search, it is substantially slower than the translation methods (tens of seconds for a modest ROI on CPU) — it is an export-time correction, not an interactive one.

## Detailed installation (Windows)

These steps are written for Windows with Anaconda or Miniconda. The same `conda` and `pip`
commands work on macOS and Linux, but the GPU part is Windows/Linux only.

### 1. Create the conda environment

Open **Anaconda Prompt** and run:

```cmd
git clone https://github.com/janfpl/shifter.git
cd shifter
conda create -n shifter python=3.12 pyqt numba -y
conda activate shifter
pip install -e .
```

> **No git?** You can download the code instead of cloning it. On the
> [repository page](https://github.com/janfpl/shifter), click the green **Code** button →
> **Download ZIP**, then extract the ZIP (right-click → *Extract All*). In Anaconda Prompt,
> `cd` into the extracted folder (the one containing `README.md`, e.g.
> `cd C:\Users\you\Downloads\shifter-main\shifter-main`) and run the commands above from
> `conda create` onwards. To install a branch other than `main`, pick it in the branch
> menu on the repository page before clicking **Code**.
>
> `pip install -e .` runs shifter directly from that folder, so extract it somewhere
> permanent (not *Downloads*, which tends to get cleaned up) and don't move or delete it
> afterwards. A ZIP doesn't update itself: to get a newer version, download and extract
> it again, then rerun `pip install -e .` from the new folder.

The environment is named **`shifter`**. The double-click launchers
(`launch_shifter.bat` and `headless_process.bat`) activate an environment by that name. If
you pick a different name (`conda create -n myname ...`), open both `.bat` files in Notepad
and set `ENV_NAME=myname` at the top. `conda env list` shows the environments you have.

Always install into the activated `shifter` environment. `pip` from another Python (for
example the Microsoft Store Python in `C:\Users\...\AppData\Local\Packages\PythonSoftwareFoundation...`)
installs packages that the launchers never see. `where python` should list
`...\envs\shifter\python.exe` first.

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

### 2. GPU acceleration (optional): choose CUDA 12 or CUDA 13

The GPU is optional; without it everything runs on the CPU. It needs an NVIDIA GPU with
compute capability 8.6 or newer (GeForce RTX 30-series, RTX A-series workstation cards,
or newer).

CuPy, the GPU library, comes in one build per CUDA major version, and the build must
match both your NVIDIA driver and your CUDA Toolkit. Run `nvidia-smi` and read
**`CUDA Version`** in the top-right corner. That is the newest CUDA your driver supports.

| `nvidia-smi` shows | Use | CUDA Toolkit to install | pip extra | Installs |
|---|---|---|---|---|
| `CUDA Version: 13.x` (driver R580 or newer) | CUDA 13 | 13.x | `.[gpu-cuda13]` | `cupy-cuda13x` |
| `CUDA Version: 12.x` | CUDA 12 | 12.x (12.6 is the tested version) | `.[gpu-cuda12]` | `cupy-cuda12x` |
| lower than 12.0 | update the NVIDIA driver first | | | |

A driver that supports CUDA 13 also runs CUDA 12, so CUDA 12 works on either kind of
machine. Within one major version the minor versions don't need to match (a 13.4 toolkit
runs on a driver that reports 13.0).

1. Install the NVIDIA CUDA Toolkit of that major version from
   [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   (older versions: *CUDA Toolkit Archive*). It installs to
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>`.
2. Install the matching CuPy build into the `shifter` environment. Only one CuPy build may
   be installed at a time, so remove the other one first:

   **CUDA 12:**
   ```cmd
   conda activate shifter
   pip uninstall -y cupy-cuda13x
   pip install -e ".[gpu-cuda12]"
   ```

   **CUDA 13:**
   ```cmd
   conda activate shifter
   pip uninstall -y cupy-cuda12x
   pip install -e ".[gpu-cuda13]"
   ```

   The older `.[gpu]` extra still works and is the same as `.[gpu-cuda12]`.

### 3. Point the conda environment at the CUDA Toolkit (`CUDA_PATH`)

shifter looks for the toolkit automatically (`CUDA_PATH`, `CUDA_PATH_V13_4`-style
variables set by the NVIDIA installer, `nvcc` on `PATH`, then `C:\Program Files\NVIDIA GPU
Computing Toolkit\CUDA`), so this step is often unnecessary. Set it explicitly if you have
several toolkits installed, the GPU check fails, or the toolkit is in a non-standard place.

Store the variable in the conda environment itself, so it is set whenever the
environment is activated (including by the launchers):

```cmd
conda activate shifter
conda env config vars set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
conda deactivate
conda activate shifter
echo %CUDA_PATH%
```

Use your own toolkit folder, e.g. `...\CUDA\v12.6` for CUDA 12. Point it at the toolkit
folder itself, **not** its `bin` subfolder. `conda env config vars list` shows what is
set, and `conda env config vars unset CUDA_PATH` removes it.

If your conda is too old for `conda env config vars` (before 4.8), use an activation
script instead:

```cmd
conda activate shifter
mkdir "%CONDA_PREFIX%\etc\conda\activate.d"
echo set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"> "%CONDA_PREFIX%\etc\conda\activate.d\cuda_path.bat"
```

Alternatively set `CUDA_PATH` machine-wide under *Windows Settings → System → About →
Advanced system settings → Environment Variables*, then open a new Anaconda Prompt.

### 4. Check the GPU

```cmd
conda activate shifter
python -c "from shifter.registration.gpu_utils import gpu_available, gpu_fail_reason, gpu_name; print(gpu_available(), gpu_name(), gpu_fail_reason())"
```

`True <your GPU name>` means the GPU will be used. `False` is followed by the reason; see
[GPU Acceleration](#gpu-acceleration) for the common ones. This runs the same check the
app runs at startup.

## GPU Acceleration

All registration algorithms support optional GPU acceleration via CuPy. The widget displays the detected GPU name or indicates CPU-only mode. If a GPU computation fails (e.g., out of memory), it falls back to CPU automatically.

On startup the app checks for a usable GPU by JIT-compiling a small test kernel with CuPy/NVRTC. Because a CuPy build that does not match the installed CUDA driver/toolkit can make that compile fault at the native level (a Windows *access violation*), the check runs in a **separate subprocess** — if it crashes, the app reports CPU mode and keeps running rather than going down with it.

The probe is attempted twice, each in its own subprocess. The first attempt is **isolated**: it removes any system CUDA-toolkit directory from `PATH` so CuPy loads only its own bundled CUDA libraries. This fixes the most common failure on Windows, where a system CUDA toolkit's `nvrtc` DLL (already on `PATH` from the CUDA installer) shadows the different-version one used by `cupy-cuda12x`/`cupy-cuda13x` and crashes the compile. The second attempt injects the **system** CUDA toolkit path, for setups (e.g. conda `cudatoolkit`) whose CuPy relies on the system libraries. Whichever succeeds, that same `PATH` arrangement is applied to the app process for the real GPU work. Two environment variables control the check:

- `SHIFTER_DISABLE_GPU=1` — skip the GPU probe entirely and run on CPU (fastest startup; use this if the probe is slow or unreliable on your machine).
- `SHIFTER_GPU_PROBE=inprocess` — run the probe in-process (the old behaviour), for debugging only; a native CuPy fault will crash the app.

To install GPU support, see [Installation](#installation), or [Detailed installation step 2](#2-gpu-acceleration-optional-choose-cuda-12-or-cuda-13) for choosing between CUDA 12 and 13.

Any **CUDA 12.x** or **CUDA 13.x** runtime is supported; the installed CuPy wheel must match the major version (`cupy-cuda12x` for 12.x, `cupy-cuda13x` for 13.x). The app is tested against CUDA 12.6. The CuPy wheel may report a runtime version (e.g. 12.9) different from a separately installed toolkit of the same major version — that is expected and fine. CUDA 11.x and older are not supported. CUDA 13.x requires an NVIDIA driver from the R580 series or newer; an older driver fails with `cudaErrorInsufficientDriver` and the startup banner says so.

**Troubleshooting: GPU shows CPU mode with a "could not compile a test kernel" banner**

If the startup banner reports that CuPy could not compile a test kernel on an otherwise-supported CUDA 12.x or 13.x runtime, a system CUDA toolkit's `nvrtc` DLL is most likely shadowing CuPy's bundled one. Try, in order:

1. Refresh CuPy for your CUDA version: `pip install -U cupy-cuda12x` (CUDA 12.x) or `pip install -U cupy-cuda13x` (CUDA 13.x).
2. Update your NVIDIA driver.
3. Diagnose directly with the probe, which takes a `--strategy` and prints its JSON result (and, thanks to faulthandler, any native crash stack):

   ```cmd
   python -m shifter.registration._gpu_probe --strategy isolated
   python -m shifter.registration._gpu_probe --strategy system
   python -m shifter.registration._gpu_probe --strategy bundled
   ```

   `isolated` is what the app tries first (it drops system CUDA-toolkit dirs from `PATH` so CuPy uses its bundled libraries); if it fails, the app tries `system` (it adds the CUDA Toolkit from `CUDA_PATH`). `bundled` leaves `PATH` untouched. If **either** `isolated` or `system` prints `"available": true`, the app will use the GPU on the next launch. With a system CUDA 13 toolkit it is normal for `isolated` to fail with `failed to open nvrtc-builtins64_1xx.dll` and `system` to succeed: CuPy uses the toolkit's NVRTC, whose companion DLL the isolated check hides on purpose.

**Troubleshooting: GPU not detected / "NVRTC not found"**

If the reason mentions NVRTC or missing CUDA libraries, `CUDA_PATH` is probably unset
inside the conda environment, or points at the wrong toolkit. Check it with
`echo %CUDA_PATH%` in the activated environment and set it as described in
[Detailed installation step 3](#3-point-the-conda-environment-at-the-cuda-toolkit-cuda_path).

**Troubleshooting: "NVIDIA driver is too old" banner**

The installed CuPy build needs a newer CUDA than your driver supports (typically
`cupy-cuda13x` on a driver older than R580). Update the NVIDIA driver, or switch to the
CUDA 12 build: `pip uninstall -y cupy-cuda13x` then `pip install -e ".[gpu-cuda12]"`.

**Troubleshooting: app closes immediately on startup**

The GPU probe now runs out-of-process, so a native CuPy/NVRTC crash should no longer take the app down — it falls back to CPU and prints a banner explaining why. If you are on an older build, or the app still exits during "Building Chromatic Shift Corrector widget" with a *Windows fatal exception: access violation* traceback pointing into `cupy`/NVRTC, start with the probe disabled:

```cmd
set SHIFTER_DISABLE_GPU=1
python -m shifter
```

The app will run on CPU. To restore GPU acceleration, reinstall CuPy to match your CUDA version (`pip install cupy-cuda12x` for CUDA 12.x, `pip install cupy-cuda13x` for CUDA 13.x) and update your NVIDIA driver, then unset the variable.

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
