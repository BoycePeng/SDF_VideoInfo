# config/__init__.py
"""
Configuration module for SDF Microcirculation Analysis.

Contains default parameters and settings for video processing,
image analysis, and microcirculation metrics calculation.
"""

from .default_config import *

__all__ = [
    # Calibration
    'PIXEL_TO_UM_RATIO',
    # Video processing
    'FPS', 'VIDEO_CODEC',
    # Image processing
    'MIN_AREA', 'BLUR_KERNEL_SIZE', 'CLAHE_CLIP_LIMIT', 'CLAHE_TILE_SIZE',
    'ADAPTIVE_BLOCK_SIZE', 'ADAPTIVE_C', 'THRESHOLD_VALUE', 'OTSU_BINARIZATION',
    # Flow analysis
    'GRID_ROWS', 'GRID_COLS',
    # De Backer
    'DEBACKER_GRID_SIZE',
    # Output
    'OUTPUT_PBR', 'OUTPUT_DEBACKER', 'OUTPUT_SPEEDS', 'OUTPUT_VIDEOS',
]
