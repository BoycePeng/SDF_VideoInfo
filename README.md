# SDF Microcirculation Video Analysis

An automated analysis pipeline for Sublingual Dark Field (SDF) microcirculation videos. The pipeline computes key physiological indicators from video input: **PBR** (Perfused Boundary Region), **TVD** (Total Vessel Density), **De Backer Score**, and **Blood Flow Velocity**.

## Features

- **Video Stabilization** — Standard single-pass and advanced 3-pass split-restabilize mode to remove motion artifacts
- **PBR Calculation** — Perfused Boundary Region measurement with per-frame tracking
- **Blood Flow Velocity** — Dense optical flow (Farneback) with grid-based regional speed analysis
- **De Backer Score & TVD** — Automated vessel density and crossing analysis
- **Batch Processing** — Process entire directories of videos with nested folder support
- **Configurable** — All parameters (calibration, thresholds, grid size) are centralized in `config/default_config.py`

## Project Structure

```
SDFcode/
├── src/                           # Source code package
│   ├── __init__.py                # Package init; exports PBRCalculator, DeBackerAnalyzer, FlowSpeedAnalyzer, VideoStabilizer
│   ├── core/                      # Core algorithm modules
│   │   ├── __init__.py            # Exports core classes
│   │   ├── pbr.py                 # PBR (Perfused Boundary Region) calculation
│   │   ├── debacker.py            # De Backer Score & TVD analysis
│   │   ├── flow_speed.py          # Blood flow velocity via dense optical flow
│   │   └── video_stable.py        # Video stabilization (standard + advanced 3-pass)
│   └── utils/                     # Utility functions
│       ├── __init__.py            # Exports VideoReader, VideoWriter, image utilities
│       ├── video_utils.py         # Video I/O helpers (VideoReader, VideoWriter)
│       └── image_utils.py         # Image processing helpers (morphology, thresholding)
├── config/                        # Configuration module
│   ├── __init__.py                # Config package init; exports all parameters
│   └── default_config.py          # All default parameters (calibration, processing, output paths)
├── scripts/                       # Executable CLI scripts
│   ├── single_process.py          # Single video processing pipeline
│   ├── batch_process.py           # Batch video processing pipeline
│   └── combine_results.py         # Merge per-video results into a combined CSV
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_pbr.py                # PBR calculator unit tests
│   ├── test_debacker.py           # De Backer analyzer unit tests
│   └── test_flow_speed.py         # Flow speed analyzer unit tests
├── tools/                         # Utility scripts
│   ├── check_video.py             # Quick video file info checker
│   └── test_import.py             # Import verification script
├── data/                          # [Placeholder] Input video files
├── output/                        # [Placeholder] Analysis results
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup script (pip installable)
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.10
- pip

### Install via pip (recommended)

```bash
# Clone the repository
git clone <repo-url>
cd SDFcode

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Install via conda

```bash
conda create -n sdf python=3.10
conda activate sdf
pip install -r requirements.txt
pip install -e .
```

## Usage

### Single Video Analysis

```bash
python scripts/single_process.py --input video.mp4 --output ./results
```

With options:

```bash
python scripts/single_process.py \
    --input video.mp4 \
    --output ./results \
    --pixel-ratio 1.1851 \
    --grid-rows 6 --grid-cols 6 \
    --enhance --method hist_eq \
    --advanced
```

### Batch Processing

```bash
python scripts/batch_process.py --input ./data --output ./results
```

The batch processor auto-detects folder structure:
- Flat: `data/video1.mp4`, `data/video2.mp4`
- Nested: `data/patient1/video1.mp4`, `data/patient2/video2.mp4`

### Combine Results

After batch processing, merge all per-video results into a single CSV:

```bash
python scripts/combine_results.py
```

This reads from `results_advanced/结果.txt` and speed CSV files, producing `results_advanced/all_results_combined.csv`.

### Python API

```python
from src.core import PBRCalculator, DeBackerAnalyzer, FlowSpeedAnalyzer, VideoStabilizer

# Video stabilization
stabilizer = VideoStabilizer()
stabilizer.stabilize_video('input.mp4', 'output_stable.mp4')

# PBR calculation
pbr = PBRCalculator()
avg_pbr, std_pbr, _ = pbr.process_video('stable.mp4', './output', pixel_to_um_ratio=1.1851, video_name='sample')

# Flow velocity
flow = FlowSpeedAnalyzer(pixel_to_um_ratio=1.1851, grid_rows=6, grid_cols=6)
speeds = flow.process_video('stable.mp4', 'flow.mp4', 'mask.mp4', enhance=True)

# De Backer Score & TVD
debacker = DeBackerAnalyzer()
tvd, score = debacker.analyze_video('mask.mp4', pixel_to_um_ratio=1.1851, output_dir='./output')
```

## Configuration

All parameters are centralized in `config/default_config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PIXEL_TO_UM_RATIO` | 1.1851 | Calibration factor (μm/pixel) |
| `FPS` | 30 | Input video frame rate |
| `GRID_ROWS` / `GRID_COLS` | 6 | Grid division for flow analysis |
| `MIN_AREA` | 1500 | Minimum vessel area in pixels |
| `RBC_THRESHOLD` | 120 | RBC extraction threshold |
| `SMOOTHING_RADIUS` | 1 | Stabilization trajectory smoothing window |
| `FLOW_WINDOW_SIZE` | 15 | Optical flow averaging window |
| `DEBACKER_GRID_SIZE` | 4 | De Backer analysis grid size |

## Processing Pipeline

The analysis pipeline processes each video through four sequential steps:

1. **Video Stabilization** — Removes camera shake using feature-based motion estimation and trajectory smoothing. The advanced mode (`--advanced`) splits the video into segments, stabilizes each independently, and re-stabilizes the concatenated result for long recordings.

2. **PBR Calculation** — Extracts the Perfused Boundary Region by combining vessel masks with RBC (red blood cell) detection, measuring the boundary region width in micrometers.

3. **Blood Flow Velocity** — Uses Farneback dense optical flow to compute per-region velocity with configurable grid division and contrast enhancement (CLAHE / histogram equalization / gamma correction).

4. **De Backer Score & TVD** — Computes Total Vessel Density and De Backer Score from skeletonized vessel masks using grid-based crossing analysis.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | >=4.5.0 | Video I/O, image processing, optical flow |
| numpy | >=1.20.0 | Numerical operations |
| scikit-image | >=0.18.0 | Skeletonization (morphology) |
| scipy | >=1.7.0 | Distance transform (ndimage) |
| Pillow | >=8.0.0 | Image enhancement (PIL.Image, ImageEnhance) |

### Optional (development)

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=6.0 | Running unit tests |
| black | >=21.0 | Code formatting |
| flake8 | >=3.9 | Linting |

## Output Files

After processing, the following output structure is generated:

```
results/
├── pbr/                          # PBR per-frame CSV results
├── debacker/                     # De Backer & TVD analysis CSVs
└── videos/
    └── <video_name>/
        ├── <name>_stable.mp4    # Stabilized video
        ├── <name>_compare.mp4   # Side-by-side comparison
        ├── <name>_flow.mp4      # Flow visualization
        ├── <name>_mask.mp4      # Vessel mask video
        └── <name>_speeds.csv    # Regional speed data
```

## Running Tests

```bash
pytest tests/
```

## Author

Boyuan Peng — burrypeng@gmail.com
