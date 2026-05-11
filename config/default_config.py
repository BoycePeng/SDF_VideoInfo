# SDF Microcirculation Analysis Configuration
# Default parameters for video analysis

# =============================================================================
# Calibration Parameters
# =============================================================================

# Pixel to micron conversion ratio (μm/pixel)
# This value should be calibrated based on your imaging system
PIXEL_TO_UM_RATIO = 1.1851

# =============================================================================
# Video Processing Parameters
# =============================================================================

# Frame rate of input videos (fps)
FPS = 30

# Video codec for output (as tuple for cv2.VideoWriter_fourcc)
VIDEO_CODEC = ('m', 'p', '4', 'v')

# =============================================================================
# Image Processing Parameters
# =============================================================================

# Morphological operations
MORPH_KERNEL_SIZE = (5, 5)
MORPH_KERNEL_SHAPE = 'ellipse'  # 'ellipse', 'rect', 'cross'

# Gaussian blur
BLUR_KERNEL_SIZE = (11, 11)
BLUR_SIGMA = 0

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_SIZE = (8, 8)

# Thresholding
THRESHOLD_VALUE = 127
OTSU_BINARIZATION = True

# Adaptive threshold
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2

# Connected components filtering
MIN_AREA = 1500  # Minimum vessel area in pixels

# =============================================================================
# Blood Flow Analysis Parameters
# =============================================================================

# Grid division for spatial analysis
GRID_ROWS = 6
GRID_COLS = 6

# Optical flow parameters (Farneback method)
FLOW_PYRAMID_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINDOW_SIZE = 15
FLOW_POLYNOMIAL_ITERATIONS = 3
FLOW_POLYNOMIAL_SIZE = 5
FLOW_SIGMA_GAUSSIAN = 1.2

# Flow visualization
FLOW_STEP = 16  # Sampling step for visualization
FLOW_SCALE = 3  # Arrow scale factor

# Image enhancement methods
ENHANCE_METHODS = ['clahe', 'hist_eq', 'gamma']
DEFAULT_ENHANCE_METHOD = 'hist_eq'

# Gamma correction
GAMMA_VALUE = 1.5

# =============================================================================
# De Backer Analysis Parameters
# =============================================================================

# Grid size for De Backer score calculation
DEBACKER_GRID_SIZE = 4

# =============================================================================
# Video Stabilization Parameters
# =============================================================================

# Feature detection
MAX_CORNERS = 200
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 30
BLOCK_SIZE = 3
USE_HARRIS = True
HARRIS_K = 0.04

# Trajectory smoothing
SMOOTHING_RADIUS = 1  # Moving average window radius

# Video cropping
CROP_PERCENT = 0.1  # Border crop percentage

# Advanced stabilization (split-re-stabilize)
FRAME_INTERVAL = 5      # 分视频间隔（原版默认值）
ADVANCED_STABILIZE = False  # 是否启用高级稳定模式
TEMP_DIR = 'temp_stabilize'  # 临时文件目录

# =============================================================================
# Output Parameters
# =============================================================================

# Output directories
OUTPUT_PBR = 'output/pbr'
OUTPUT_DEBACKER = 'output/debacker'
OUTPUT_SPEEDS = 'output/speeds'
OUTPUT_VIDEOS = 'output/videos'

# CSV column headers
CSV_PBR_HEADERS = ['Frame', 'PBR (μm)']
CSV_DEBACKER_HEADERS = ['Frame Index', 'TVD (μm)', 'De Backer Score']
CSV_SPEED_HEADERS = ['Region', 'Average Speed (μm/s)', 'Standard Deviation (μm/s)']

# =============================================================================
# PBR Calculation Parameters
# =============================================================================

# RBC extraction threshold
RBC_THRESHOLD = 120

# Morphological closing kernel for RBC mask
RBC_MORPH_KERNEL_SIZE = (10, 10)

# Combined mask weights
VESSEL_WEIGHT = 0.5
RBC_WEIGHT = 0.5
