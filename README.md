# Chromatic Shift Corrector

Napari-based axial and lateral chromatic shift correction for light sheet microscopy. Provides interactive 3D visualization, automatic registration, manual shift adjustment, and chunked full-volume export.

## Installation

Clone the repository and create a conda environment with Python 3.12 and Qt:

```bash
git clone https://github.com/janfpl/shifter.git
cd shifter
conda create -n shifter python=3.12 pyqt -y
conda activate shifter
pip install -e .
```

For GPU acceleration (optional):

```bash
pip install -e ".[gpu]"
```

Requires Python 3.12.

## Usage

Launch the application:

```bash
python -m chromatic_shift_corrector
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
- On export, pyramids are regenerated for corrected volumes using block averaging

## Workflow

### 1. Load Data

- Select the input format (BigTIFF or Luxendo H5)
- Choose a directory containing your channel files
- Select which files to load and assign channel order, reference channel, and colormaps
- For H5 files, voxel sizes are auto-populated from embedded metadata if available
- For BigTIFF directories containing a `.xml` sidecar, voxel sizes are extracted automatically

### 2. Register Channels

Draw a rectangle ROI on the napari viewer and specify a Z sub-range to define the registration volume. Select which channels to register against the reference, choose an algorithm, and run.

Results populate the shift table with X/Y/Z voxel shifts and a confidence score per channel. Confidence is color-coded in the table (green = high, red = low).

**Preprocessing options:**
- Background subtraction (percentile-based)
- Gaussian smoothing

### 3. Adjust Shifts

Shifts can be edited manually via spinboxes in the shift table. Use the preview button to visualize the corrected sub-volume in napari before committing to a full export.

### 4. Export

Select an output directory and RAM allocation (50-95% of system memory). The export streams corrected volumes in Z-slab chunks, writing one file per channel. A `correction_metadata.json` sidecar is written alongside the output files containing all shift parameters, voxel sizes, and processing details.

Output format matches the input format:
- BigTIFF input produces BigTIFF output
- Luxendo H5 input produces H5 output with regenerated resolution pyramids and preserved metadata

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

Recommended as the default algorithm for most datasets.

### Mutual Information

Coarse-to-fine exhaustive search maximizing mutual information via joint histograms. Coarse pass uses a step size of 5 voxels; fine pass refines within a 5-voxel radius.

| Aspect | Detail |
|--------|--------|
| Speed | Slow (exhaustive search over 3D shift space) |
| Best for | Channels with non-linear intensity relationships (e.g., different fluorophores, modalities) |
| Limitations | Significantly slower than FFT-based methods |
| Parameters | None (beyond search range) |
| GPU | Supported via CuPy (accelerates histogram computation) |

Use when Phase Cross-Correlation and ZNCC fail due to dissimilar intensity distributions between channels.

## GPU Acceleration

All registration algorithms support optional GPU acceleration via CuPy. The widget displays the detected GPU name or indicates CPU-only mode. If a GPU computation fails (e.g., out of memory), it falls back to CPU automatically.

Install GPU support:

```bash
pip install -e ".[gpu]"
```

Requires CUDA 12.x compatible hardware and drivers.

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
- CuPy (optional, for GPU acceleration)
