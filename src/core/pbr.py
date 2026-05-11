"""
PBR (Perfused Boundary Region) Calculator

This module calculates the PBR metric from SDF (Sublingual Dark Field) microcirculation videos.
PBR measures the functional perfusion capability of microvasculature by analyzing the distance
between vessel walls and the red blood cell column.

Algorithm:
    1. Apply Gaussian blur and CLAHE enhancement to improve contrast
    2. Combine OTSU global thresholding with adaptive thresholding
    3. Apply morphological operations (closing, opening) to remove noise
    4. Filter connected components by minimum area threshold
    5. Extract vessel mask and RBC mask separately
    6. Apply skeletonization and distance transform to both masks
    7. PBR = mean_distance(vessel) - mean_distance(RBC)

Reference:
    - PBR concept: Microvascular perfusion assessment in critical illness
"""

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import csv

from config.default_config import (
    MIN_AREA, BLUR_KERNEL_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C, RBC_THRESHOLD, RBC_MORPH_KERNEL_SIZE,
    VESSEL_WEIGHT, RBC_WEIGHT, PIXEL_TO_UM_RATIO, FPS, VIDEO_CODEC
)


class PBRCalculator:
    """
    Calculator for Perfused Boundary Region (PBR) metric.
    
    PBR is a key indicator of microvascular functional perfusion, representing
    the ratio of perfused vessel area to total vessel area.
    
    Attributes:
        min_area (int): Minimum area threshold for connected component filtering
        kernel (np.ndarray): Morphological structuring element
        clahe (cv2.CLAHE): CLAHE object for contrast enhancement
    """
    
    def __init__(self, min_area: int = MIN_AREA):
        """
        Initialize the PBR calculator.
        
        Args:
            min_area: Minimum area (pixels) for vessel detection.
                     Components smaller than this are filtered out.
        """
        self.min_area = min_area
        # Create elliptical morphological kernel for vessel morphology
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Initialize CLAHE for adaptive contrast enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE
        )
    
    def preprocess_image(self, gray_image: np.ndarray) -> tuple:
        """
        Preprocess image to extract vessel mask.
        
        Processing pipeline:
            1. Gaussian blur to reduce noise
            2. CLAHE enhancement to improve local contrast
            3. Combine OTSU and adaptive thresholding
            4. Morphological operations to clean up
            5. Connected component filtering
        
        Args:
            gray_image: Grayscale input image (2D array)
            
        Returns:
            tuple: (filled_mask, masked_image)
                - filled_mask: Binary vessel mask (255 for vessel, 0 for background)
                - masked_image: Original image masked by vessel region
        """
        # Step 1: Noise reduction
        blurred = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)
        
        # Step 2: Contrast enhancement
        enhanced = self.clahe.apply(blurred)
        
        # Step 3: Dual thresholding for robust segmentation
        # OTSU's method finds optimal global threshold automatically
        _, global_thresh = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Adaptive threshold handles non-uniform illumination
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )
        # Combine both thresholds for robustness
        combined_thresh = cv2.bitwise_and(global_thresh, adaptive_thresh)
        
        # Step 4: Morphological operations
        # Closing fills small holes inside vessels
        binary_image = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, self.kernel)
        # Opening removes small noise outside vessels
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, self.kernel)
        # Re-add original threshold to preserve thin connections
        binary_image = cv2.bitwise_or(binary_image, combined_thresh)
        # Invert to get vessel as white
        binary_image = cv2.bitwise_not(binary_image)
        
        # Step 5: Connected component analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        # Filter small regions (likely noise)
        filtered_image = np.zeros_like(binary_image)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                filtered_image[labels == i] = 255
        
        # Final morphological closing to smooth vessel boundaries
        filled_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, self.kernel)
        # Apply mask to original image
        masked_image = cv2.bitwise_and(gray_image, gray_image, mask=filled_image)
        
        return filled_image, masked_image
    
    def extract_rbc_mask(self, enhanced_masked_image: np.ndarray) -> np.ndarray:
        """
        Extract red blood cell mask from the masked image.
        
        RBCs appear darker than plasma in SDF imaging. This method uses
        inverse thresholding to capture RBC regions.
        
        Args:
            enhanced_masked_image: CLAHE-enhanced masked image
            
        Returns:
            np.ndarray: Binary RBC mask
        """
        # Inverse threshold: RBCs are dark (low intensity)
        _, rbc_mask = cv2.threshold(
            enhanced_masked_image, RBC_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )
        
        # Small morphological kernel for RBC cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rbc_mask = cv2.morphologyEx(rbc_mask, cv2.MORPH_CLOSE, kernel)
        rbc_mask = cv2.morphologyEx(rbc_mask, cv2.MORPH_OPEN, kernel)
        
        return rbc_mask
    
    def process_binary_mask(self, binary_mask: np.ndarray) -> tuple:
        """
        Perform skeletonization and distance transform on binary mask.
        
        Skeletonization reduces vessels to 1-pixel wide centerlines.
        Distance transform computes distance from each pixel to nearest zero-pixel.
        
        Args:
            binary_mask: Binary vessel or RBC mask
            
        Returns:
            tuple: (skeleton, mean_distance, distance_transform)
        """
        # Ensure mask is binary (0 and 1)
        skeleton = skeletonize(binary_mask // 255)
        # Calculate Euclidean distance to background
        distance_transform = distance_transform_edt(binary_mask)
        # Get distances only on skeleton pixels
        skeleton_distances = distance_transform[skeleton]
        mean_distance = np.mean(skeleton_distances) if len(skeleton_distances) > 0 else 0
        
        return skeleton, mean_distance, distance_transform
    
    def calculate_pbr(self, vessel_mask: np.ndarray, rbc_mask: np.ndarray) -> float:
        """
        Calculate PBR from vessel and RBC masks.
        
        PBR formula:
            PBR = mean_distance_to_vessel_wall - mean_distance_to_RBC_column
            
        A lower PBR indicates better perfusion (RBCs can access vessel edges).
        Higher PBR suggests impaired perfusion or endothelial swelling.
        
        Args:
            vessel_mask: Binary vessel mask
            rbc_mask: Binary RBC mask
            
        Returns:
            float: PBR value (in pixels, should be multiplied by PIXEL_TO_UM_RATIO)
        """
        # Get mean distances for vessel and RBC skeletons
        _, mean_distance_vessel, _ = self.process_binary_mask(vessel_mask)
        _, mean_distance_rbc, _ = self.process_binary_mask(rbc_mask)
        
        # PBR is the difference between vessel and RBC mean distances
        pbr = mean_distance_vessel - mean_distance_rbc
        
        return pbr
    
    def enhance_contrast(self, masked_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE contrast enhancement within the vessel mask.
        
        Args:
            masked_image: Original image masked by vessel region
            mask: Binary vessel mask
            
        Returns:
            np.ndarray: Contrast-enhanced image
        """
        enhanced = self.clahe.apply(masked_image)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        return enhanced
    
    def process_video(
        self,
        video_path: str,
        output_folder: str,
        pixel_to_um_ratio: float = PIXEL_TO_UM_RATIO,
        file_name: str = None
    ) -> tuple:
        """
        Process a video file and calculate PBR for all frames.
        
        Args:
            video_path: Path to input video file
            output_folder: Directory for output CSV files
            pixel_to_um_ratio: Calibration factor (μm/pixel)
            file_name: Base name for output files (derived from video if None)
            
        Returns:
            tuple: (average_pbr, std_pbr, combined_masks)
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Derive file name from video path if not provided
        if file_name is None:
            file_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbr_values = []
        combined_masks = []
        
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Preprocess and extract vessel mask
            filled_image, masked_image = self.preprocess_image(gray_frame)
            
            # Enhance contrast and extract RBC mask
            enhanced_masked_image = self.enhance_contrast(masked_image, filled_image)
            rbc_mask = self.extract_rbc_mask(enhanced_masked_image)
            
            # Refine RBC mask
            inverted_rbc_mask = cv2.bitwise_not(rbc_mask)
            inverted_filled_mask = cv2.bitwise_not(filled_image)
            rbc_mask = rbc_mask - inverted_filled_mask
            
            # Morphological cleanup of RBC mask
            kernel = np.ones(RBC_MORPH_KERNEL_SIZE, np.uint8)
            filled_rbc_mask = cv2.morphologyEx(rbc_mask, cv2.MORPH_CLOSE, kernel)
            filled_rbc_mask = cv2.morphologyEx(filled_rbc_mask, cv2.MORPH_OPEN, kernel)
            rbc_mask = cv2.bitwise_or(rbc_mask, filled_rbc_mask)
            
            # Create combined mask for visualization
            combined_mask = cv2.addWeighted(
                filled_image, VESSEL_WEIGHT, rbc_mask, RBC_WEIGHT, 0
            )
            combined_masks.append(combined_mask)
            
            # Calculate PBR for this frame
            pbr = self.calculate_pbr(filled_image, rbc_mask) * pixel_to_um_ratio
            pbr_values.append(pbr)
        
        cap.release()
        
        # Calculate statistics
        average_pbr = np.mean(pbr_values)
        std_pbr = np.std(pbr_values)
        
        # Save results to CSV
        csv_path = os.path.join(output_folder, f'{file_name}_pbr_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Average PBR (μm)', 'Standard Deviation (μm)'])
            writer.writerow([f'{average_pbr:.2f}', f'{std_pbr:.2f}'])
        
        # Save visualization example
        example_idx = len(combined_masks) // 2
        example_path = os.path.join(output_folder, f'{file_name}_combined_mask_example.png')
        cv2.imwrite(example_path, combined_masks[example_idx])
        
        return average_pbr, std_pbr, combined_masks
