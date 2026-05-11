# src/utils/image_utils.py
"""
Image processing utilities.

Provides helper functions for common image processing operations
used in microcirculation analysis.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def apply_morphology(
    binary_image: np.ndarray,
    operation: str = 'close',
    kernel_size: Tuple[int, int] = (5, 5),
    kernel_shape: str = 'ellipse'
) -> np.ndarray:
    """
    Apply morphological operation to binary image.
    
    Args:
        binary_image: Input binary image
        operation: Operation type ('open', 'close', 'erode', 'dilate')
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element ('ellipse', 'rect', 'cross')
        
    Returns:
        np.ndarray: Processed binary image
    """
    kernel_map = {
        'ellipse': cv2.MORPH_ELLIPSE,
        'rect': cv2.MORPH_RECT,
        'cross': cv2.MORPH_CROSS
    }
    
    kernel = cv2.getStructuringElement(kernel_map[kernel_shape], kernel_size)
    
    op_map = {
        'open': cv2.MORPH_OPEN,
        'close': cv2.MORPH_CLOSE,
        'erode': cv2.MORPH_ERODE,
        'dilate': cv2.MORPH_DILATE
    }
    
    return cv2.morphologyEx(binary_image, op_map[operation], kernel)


def filter_by_area(
    binary_image: np.ndarray,
    min_area: int = 0,
    max_area: int = None,
    connectivity: int = 8
) -> np.ndarray:
    """
    Filter connected components by area.
    
    Args:
        binary_image: Input binary image
        min_area: Minimum component area (0 = no minimum)
        max_area: Maximum component area (None = no maximum)
        connectivity: Connectivity (4 or 8)
        
    Returns:
        np.ndarray: Filtered binary image
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=connectivity
    )
    
    filtered = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if min_area <= area if max_area is None else min_area <= area <= max_area:
            filtered[labels == i] = 255
    
    return filtered


def enhance_contrast(
    gray_image: np.ndarray,
    method: str = 'clahe',
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        gray_image: Input grayscale image
        method: Enhancement method ('clahe', 'hist_eq', 'adaptive')
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile grid size
        
    Returns:
        np.ndarray: Enhanced grayscale image
    """
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(gray_image)
    
    elif method == 'hist_eq':
        return cv2.equalizeHist(gray_image)
    
    elif method == 'adaptive':
        return cv2.equalizeHist(gray_image)
    
    return gray_image


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    fill_value: int = 0
) -> np.ndarray:
    """
    Apply binary mask to image.
    
    Args:
        image: Input image (grayscale or BGR)
        mask: Binary mask (255 = keep, 0 = mask)
        fill_value: Value to fill masked areas
        
    Returns:
        np.ndarray: Masked image
    """
    if len(image.shape) == 2:
        result = image.copy()
        result[mask == 0] = fill_value
    else:
        result = cv2.bitwise_and(image, image, mask=mask)
        if fill_value != 0:
            result[mask == 0] = fill_value
    
    return result


def binarize(
    gray_image: np.ndarray,
    method: str = 'otsu',
    threshold: int = 127,
    invert: bool = False
) -> np.ndarray:
    """
    Binarize grayscale image.
    
    Args:
        gray_image: Input grayscale image
        method: Binarization method ('otsu', 'adaptive', 'fixed')
        threshold: Fixed threshold value
        invert: Invert result (white <-> black)
        
    Returns:
        np.ndarray: Binary image
    """
    if method == 'otsu':
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            gray_image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:  # fixed
        _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    if invert:
        binary = cv2.bitwise_not(binary)
    
    return binary


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    padding_color: Tuple[int, ...] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        padding_color: Padding color
        
    Returns:
        np.ndarray: Resized and padded image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas with padding
    if len(image.shape) == 2:
        result = np.full((target_h, target_w), padding_color[0], dtype=np.uint8)
    else:
        result = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    
    # Center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result
