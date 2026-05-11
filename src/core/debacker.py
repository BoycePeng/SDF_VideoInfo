"""
De Backer Score and TVD (Total Vessel Density) Analyzer

This module calculates standard microcirculation indices from SDF videos:
    - TVD: Total Vessel Density (vessel area / total area)
    - De Backer Score: Grid intersection-based vascular density score

These metrics are internationally recognized standards for microcirculation assessment.

Algorithm:
    1. Load video and extract vessel masks
    2. Apply OTSU thresholding for binarization
    3. Skeletonize vessel mask to get centerlines
    4. Count intersections with grid lines
    5. TVD = vessel_pixels / total_pixels
    6. De Backer Score = intersections / grid_length

Reference:
    - De Backer et al. "Microcirculatory oxygenation and shunting"
    - ICSH guidelines for microcirculation assessment
"""

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import csv

from config.default_config import (
    PIXEL_TO_UM_RATIO, THRESHOLD_VALUE, DEBACKER_GRID_SIZE, FPS, VIDEO_CODEC
)


class DeBackerAnalyzer:
    """
    Analyzer for De Backer Score and Total Vessel Density (TVD).
    
    These are standard microcirculation function indices used in clinical research.
    
    Attributes:
        grid_size (int): Number of grid divisions (e.g., 4x4 grid)
        threshold_value (int): Binary threshold for vessel segmentation
    """
    
    def __init__(self, grid_size: int = DEBACKER_GRID_SIZE, threshold_value: int = THRESHOLD_VALUE):
        """
        Initialize the De Backer analyzer.
        
        Args:
            grid_size: Number of divisions for each dimension (default: 4)
            threshold_value: Binary threshold value for vessel detection
        """
        self.grid_size = grid_size
        self.threshold_value = threshold_value
    
    def calculate_tvd(self, mask: np.ndarray, pixel_to_um_ratio: float = PIXEL_TO_UM_RATIO) -> float:
        """
        Calculate Total Vessel Density (TVD).
        
        TVD represents the proportion of image area occupied by vessels.
        
        Formula:
            TVD = vessel_area / total_area × pixel_to_um_ratio
        
        Args:
            mask: Binary vessel mask (255 for vessel, 0 for background)
            pixel_to_um_ratio: Calibration factor for unit conversion
            
        Returns:
            float: TVD value in μm
        """
        vessel_area = np.sum(mask == 255)
        total_area = mask.size
        tvd = vessel_area / total_area
        return tvd * pixel_to_um_ratio
    
    def calculate_de_backer_score(
        self,
        mask: np.ndarray,
        pixel_to_um_ratio: float = PIXEL_TO_UM_RATIO
    ) -> float:
        """
        Calculate De Backer Score based on grid intersection counting.
        
        The De Backer Score counts how many times vessel centerlines intersect
        with imaginary grid lines, normalized by grid dimensions.
        
        Formula:
            De_Backer = intersection_count / ((grid_size-1) × (grid_height + grid_width))
        
        Args:
            mask: Binary vessel mask
            pixel_to_um_ratio: Calibration factor for unit conversion
            
        Returns:
            float: De Backer Score in μm
        """
        # Step 1: Skeletonize to get vessel centerlines
        skeleton = skeletonize(mask // 255)
        height, width = skeleton.shape
        
        # Calculate grid cell dimensions
        grid_height = height // self.grid_size
        grid_width = width // self.grid_size
        
        intersection_count = 0
        vertical_intersections = []
        horizontal_intersections = []
        
        # Step 2: Count vertical line intersections
        # Check each vertical grid line (excluding first and last)
        for i in range(1, self.grid_size):
            grid_x = i * grid_width
            if grid_x < width:
                # Count skeleton pixels along this vertical line
                line_count = np.sum(skeleton[:, grid_x])
                intersection_count += line_count
                vertical_intersections.append(line_count)
        
        # Step 3: Count horizontal line intersections
        for i in range(1, self.grid_size):
            grid_y = i * grid_height
            if grid_y < height:
                # Count skeleton pixels along this horizontal line
                line_count = np.sum(skeleton[grid_y, :])
                intersection_count += line_count
                horizontal_intersections.append(line_count)
        
        # Step 4: Normalize by grid dimensions
        # Denominator: (grid_lines) × (avg_grid_height + avg_grid_width)
        # grid_lines = grid_size - 1 for each orientation
        normalization_factor = (self.grid_size - 1) * (grid_height + grid_width)
        de_backer_score = intersection_count / normalization_factor
        
        return de_backer_score * pixel_to_um_ratio
    
    def calculate_metrics(self, mask: np.ndarray, pixel_to_um_ratio: float = PIXEL_TO_UM_RATIO) -> tuple:
        """
        Calculate both TVD and De Backer Score.
        
        Args:
            mask: Binary vessel mask
            pixel_to_um_ratio: Calibration factor
            
        Returns:
            tuple: (tvd, de_backer_score)
        """
        tvd = self.calculate_tvd(mask, pixel_to_um_ratio)
        de_backer_score = self.calculate_de_backer_score(mask, pixel_to_um_ratio)
        return tvd, de_backer_score
    
    def create_annotated_image(
        self,
        mask: np.ndarray,
        show_grid: bool = True
    ) -> np.ndarray:
        """
        Create annotated visualization with skeleton and grid.
        
        Args:
            mask: Binary vessel mask
            show_grid: Whether to draw grid lines and intersections
            
        Returns:
            np.ndarray: Annotated BGR image
        """
        skeleton = skeletonize(mask // 255).astype(np.uint8) * 255
        annotated = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        height, width = skeleton.shape
        
        if show_grid:
            grid_height = height // self.grid_size
            grid_width = width // self.grid_size
            
            # Draw grid lines
            for i in range(1, self.grid_size):
                # Vertical lines (green)
                cv2.line(
                    annotated,
                    (i * grid_width, 0),
                    (i * grid_width, height),
                    (0, 255, 0), 1
                )
                # Horizontal lines (green)
                cv2.line(
                    annotated,
                    (0, i * grid_height),
                    (width, i * grid_height),
                    (0, 255, 0), 1
                )
            
            # Mark intersections with red circles
            for i in range(1, self.grid_size):
                # Vertical intersections
                for y in range(height):
                    x = i * grid_width
                    if x < width and skeleton[y, x] > 0:
                        cv2.circle(annotated, (x, y), 2, (0, 0, 255), -1)
                
                # Horizontal intersections
                for x in range(width):
                    y = i * grid_height
                    if y < height and skeleton[y, x] > 0:
                        cv2.circle(annotated, (x, y), 2, (0, 0, 255), -1)
        
        return annotated
    
    def analyze_video(
        self,
        video_path: str,
        pixel_to_um_ratio: float = PIXEL_TO_UM_RATIO,
        output_dir: str = "output"
    ) -> tuple:
        """
        Analyze entire video and calculate TVD/De Backer for each frame.
        
        Args:
            video_path: Path to input video (with vessel masks)
            pixel_to_um_ratio: Calibration factor
            output_dir: Directory for output files
            
        Returns:
            tuple: (avg_tvd, avg_de_backer_score)
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        annotated_dir = os.path.join(output_dir, f"{video_name}_annotated_frames")
        os.makedirs(annotated_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        tvd_values = []
        de_backer_scores = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Adaptive binarization: auto-detect vessel/background
            # Check image histogram to determine if vessel is dark or light
            white_ratio = np.sum(gray > 127) / gray.size
            
            if white_ratio > 0.5:
                # Light background (white), dark vessels (black)
                # Use OTSU for automatic thresholding
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Dark background, light vessels (white)
                # Invert after binarization to get vessel as 255
                _, mask = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_not(mask)
            
            # Calculate metrics
            tvd, de_backer_score = self.calculate_metrics(mask, pixel_to_um_ratio)
            tvd_values.append(tvd)
            de_backer_scores.append(de_backer_score)
            
            # Create and save annotated frame
            annotated = self.create_annotated_image(mask)
            annotated_path = os.path.join(
                annotated_dir,
                f"frame_{frame_idx:04d}_annotated.png"
            )
            cv2.imwrite(annotated_path, annotated)
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate statistics
        avg_tvd = np.mean(tvd_values)
        std_tvd = np.std(tvd_values)
        avg_de_backer = np.mean(de_backer_scores)
        std_de_backer = np.std(de_backer_scores)
        
        # Save results to CSV
        csv_path = os.path.join(output_dir, f"{video_name}_debacker_analysis_results.csv")
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Frame Index", "TVD (μm)", "De Backer Score"])
            
            for idx, (tvd, score) in enumerate(zip(tvd_values, de_backer_scores)):
                writer.writerow([idx, f"{tvd:.2f}", f"{score:.2f}"])
            
            writer.writerow([])
            writer.writerow(["Average TVD (μm)", f"{avg_tvd:.2f}"])
            writer.writerow(["Standard Deviation TVD (μm)", f"{std_tvd:.2f}"])
            writer.writerow(["Average De Backer Score", f"{avg_de_backer:.2f}"])
            writer.writerow(["Standard Deviation De Backer Score", f"{std_de_backer:.2f}"])
        
        print(f"Analysis complete. Results saved to {csv_path}")
        print(f"Average TVD: {avg_tvd:.2f} μm ± {std_tvd:.2f} μm")
        print(f"Average De Backer Score: {avg_de_backer:.2f} ± {std_de_backer:.2f}")
        
        return avg_tvd, avg_de_backer


# Backward compatibility alias
def debacker_analysis(video_path, pixel_to_um_ratio, grid_size=DEBACKER_GRID_SIZE, output_dir="output"):
    """
    Convenience function for backward compatibility.
    
    Args:
        video_path: Path to input video
        pixel_to_um_ratio: Calibration factor
        grid_size: Grid division count
        output_dir: Output directory
        
    Returns:
        tuple: (avg_tvd, avg_de_backer_score)
    """
    analyzer = DeBackerAnalyzer(grid_size=grid_size)
    return analyzer.analyze_video(video_path, pixel_to_um_ratio, output_dir)
