"""
Blood Flow Velocity Analyzer

This module calculates spatial blood flow velocity distribution from SDF videos
using dense optical flow methods. The image is divided into a grid, and velocity
is computed for each region.

Algorithm:
    1. Extract vessel mask using thresholding
    2. Calculate dense optical flow between consecutive frames
    3. Filter flow by vessel mask (only measure flow inside vessels)
    4. Convert flow magnitude to velocity units
    5. Compute mean velocity per grid region

Optical Flow Method:
    Uses Farneback algorithm for dense optical flow estimation, which provides
    sub-pixel accuracy and handles large displacements.

Formula:
    velocity = mean(flow_magnitude) × pixel_to_um_ratio × fps

Reference:
    - Farneback, G. "Two-Frame Motion Estimation Based on Polynomial Expansion"
"""

import os
import cv2
import numpy as np
import csv

from config.default_config import (
    PIXEL_TO_UM_RATIO, FPS, GRID_ROWS, GRID_COLS, VIDEO_CODEC,
    FLOW_PYRAMID_SCALE, FLOW_LEVELS, FLOW_WINDOW_SIZE,
    FLOW_POLYNOMIAL_ITERATIONS, FLOW_POLYNOMIAL_SIZE, FLOW_SIGMA_GAUSSIAN,
    FLOW_STEP, FLOW_SCALE, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    BLUR_KERNEL_SIZE, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    MORPH_KERNEL_SIZE, ENHANCE_METHODS, DEFAULT_ENHANCE_METHOD, GAMMA_VALUE
)


class FlowSpeedAnalyzer:
    """
    Analyzer for blood flow velocity using optical flow methods.
    
    Divides the image into a grid and calculates mean velocity for each region.
    
    Attributes:
        pixel_to_um_ratio (float): Calibration factor (μm/pixel)
        fps (int): Video frame rate
        grid_rows (int): Number of grid rows
        grid_cols (int): Number of grid columns
    """
    
    def __init__(
        self,
        pixel_to_um_ratio: float = PIXEL_TO_UM_RATIO,
        fps: int = FPS,
        grid_rows: int = GRID_ROWS,
        grid_cols: int = GRID_COLS
    ):
        """
        Initialize the flow speed analyzer.
        
        Args:
            pixel_to_um_ratio: Calibration factor (μm/pixel)
            fps: Video frame rate
            grid_rows: Number of grid rows for spatial analysis
            grid_cols: Number of grid columns for spatial analysis
        """
        self.pixel_to_um_ratio = pixel_to_um_ratio
        self.fps = fps
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
    
    def enhance_blood_flow(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        method: str = None
    ) -> np.ndarray:
        """
        Apply contrast enhancement to blood flow visualization.
        
        Available methods:
            - 'clahe': Contrast Limited Adaptive Histogram Equalization
            - 'hist_eq': Histogram Equalization
            - 'gamma': Gamma Correction
        
        Args:
            image: Input BGR image
            mask: Binary vessel mask
            method: Enhancement method (None for no enhancement)
            
        Returns:
            np.ndarray: Enhanced BGR image
        """
        if method is None:
            return image
        
        # Apply mask to get only vessel regions
        result = cv2.bitwise_and(image, image, mask=mask)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        if method == 'clahe':
            # CLAHE: Good for local contrast preservation
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=CLAHE_TILE_SIZE
            )
            enhanced = clahe.apply(result)
        
        elif method == 'hist_eq':
            # Global histogram equalization
            enhanced = cv2.equalizeHist(result)
        
        elif method == 'gamma':
            # Gamma correction for brightness adjustment
            inv_gamma = 1.0 / GAMMA_VALUE
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype("uint8")
            enhanced = cv2.LUT(result, table)
        
        else:
            return image
        
        # Convert back to BGR and reapply mask
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        combined = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=mask)
        
        return combined
    
    def extract_vessel_mask(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract vessel mask from grayscale image.
        
        Processing pipeline:
            1. Gaussian blur for noise reduction
            2. CLAHE enhancement for contrast
            3. Dual thresholding (OTSU + adaptive)
            4. Morphological operations
            5. Inversion to get vessel mask
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            np.ndarray: Binary vessel mask (255 = vessel, 0 = background)
        """
        # Noise reduction
        blurred = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        enhanced = clahe.apply(blurred)
        
        # Dual thresholding
        _, global_thresh = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )
        combined_thresh = cv2.bitwise_and(global_thresh, adaptive_thresh)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        binary_image = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        binary_image = cv2.bitwise_or(binary_image, combined_thresh)
        
        # Invert to get vessel as white
        binary_image = cv2.bitwise_not(binary_image)
        
        # Connected component filtering (no area filter, min_area=0)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        filtered_image = np.zeros_like(binary_image)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 0:
                filtered_image[labels == i] = 255
        
        return filtered_image
    
    def calculate_optical_flow(
        self,
        prev_gray: np.ndarray,
        next_gray: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Calculate dense optical flow using Farneback method.
        
        Args:
            prev_gray: Previous frame (grayscale)
            next_gray: Next frame (grayscale)
            mask: Vessel mask to filter flow
            
        Returns:
            np.ndarray: Flow vectors (h, w, 2) where [:,:,0] is x-flow, [:,:,1] is y-flow
        """
        # Farneback dense optical flow
        # Parameters tuned for microcirculation imaging
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=FLOW_PYRAMID_SCALE,       # Pyramid scale factor
            levels=FLOW_LEVELS,                   # Number of pyramid layers
            winsize=FLOW_WINDOW_SIZE,             # Window size for averaging
            iterations=FLOW_POLYNOMIAL_ITERATIONS, # Polynomial expansion iterations
            poly_n=FLOW_POLYNOMIAL_SIZE,         # Polynomial neighborhood size
            poly_sigma=FLOW_SIGMA_GAUSSIAN,      # Gaussian std dev
            flags=0
        )
        
        # Zero out flow outside vessel mask
        flow[mask == 0] = 0
        
        return flow
    
    def calculate_region_velocity(
        self,
        flow: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Calculate mean flow velocity from optical flow field.
        
        Args:
            flow: Optical flow vectors (h, w, 2)
            mask: Binary mask for valid region
            
        Returns:
            float: Mean velocity in μm/s
        """
        # Convert cartesian flow to polar (magnitude, angle)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Apply mask
        mask_bool = mask.astype(bool)
        valid_magnitudes = magnitude[mask_bool]
        
        if len(valid_magnitudes) > 0:
            # Mean magnitude × calibration × frame rate
            speed = np.mean(valid_magnitudes) * self.pixel_to_um_ratio * self.fps
        else:
            speed = 0
        
        return speed
    
    def draw_flow_visualization(
        self,
        gray_image: np.ndarray,
        flow: np.ndarray,
        step: int = FLOW_STEP
    ) -> np.ndarray:
        """
        Create flow field visualization with arrows.
        
        Args:
            gray_image: Grayscale base image
            flow: Optical flow field
            step: Sampling step for arrow grid
            
        Returns:
            np.ndarray: BGR visualization image
        """
        h, w = gray_image.shape[:2]
        
        # Create sampling grid
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        fx, fy = fx * FLOW_SCALE, fy * FLOW_SCALE  # Scale arrows for visibility
        
        # Create line endpoints
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        # Create base image
        vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # Draw arrows
        for (x1, y1), (x2, y2) in lines:
            # Only draw non-zero length arrows
            if (x1 != x2) or (y1 != y2):
                cv2.polylines(vis, [np.array([[x1, y1], [x2, y2]])], 0, (0, 255, 0), 2)
                cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        
        return vis
    
    def draw_grid_overlay(
        self,
        image: np.ndarray,
        cell_width: int,
        cell_height: int,
        speeds: np.ndarray = None
    ) -> np.ndarray:
        """
        Draw grid lines and velocity labels on image.
        
        Args:
            image: Base image
            cell_width: Grid cell width in pixels
            cell_height: Grid cell height in pixels
            speeds: Velocity array (grid_rows, grid_cols) for labeling
            
        Returns:
            np.ndarray: Image with grid overlay
        """
        result = image.copy()
        
        # Draw horizontal lines
        for i in range(1, self.grid_rows):
            cv2.line(
                result,
                (0, i * cell_height),
                (image.shape[1], i * cell_height),
                (255, 255, 255), 1
            )
        
        # Draw vertical lines
        for j in range(1, self.grid_cols):
            cv2.line(
                result,
                (j * cell_width, 0),
                (j * cell_width, image.shape[0]),
                (255, 255, 255), 1
            )
        
        # Add region labels and velocity values
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                label_pos = (j * cell_width + 5, i * cell_height + 15)
                speed_pos = (j * cell_width + 5, i * cell_height + 30)
                
                # Region number
                region_num = i * self.grid_cols + j + 1
                cv2.putText(
                    result,
                    f"R{region_num}",
                    label_pos,
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1
                )
                
                # Velocity value
                if speeds is not None:
                    cv2.putText(
                        result,
                        f"{speeds[i, j]:.1f}",
                        speed_pos,
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
                    )
        
        return result
    
    def save_results_to_csv(
        self,
        region_speeds: np.ndarray,
        region_std: np.ndarray,
        file_path: str
    ):
        """
        Save flow speed results to CSV file.
        
        Args:
            region_speeds: Mean velocity array (grid_rows, grid_cols)
            region_std: Standard deviation array (grid_rows, grid_cols)
            file_path: Output CSV path
        """
        overall_mean = np.mean(region_speeds)
        overall_std = np.std(region_speeds)
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Region", "Average Speed (μm/s)", "Standard Deviation (μm/s)"])
            
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    region_index = i * self.grid_cols + j + 1
                    writer.writerow([
                        f"Region {region_index}",
                        f"{region_speeds[i, j]:.2f}",
                        f"{region_std[i, j]:.2f}"
                    ])
            
            writer.writerow([])
            writer.writerow(["Overall Mean Speed (μm/s)", f"{overall_mean:.2f}"])
            writer.writerow(["Overall Speed Std (μm/s)", f"{overall_std:.2f}"])
        
        print(f"Results saved to {file_path}")
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        mask_output_path: str,
        enhance: bool = False,
        method: str = DEFAULT_ENHANCE_METHOD,
        file_name: str = "results"
    ) -> np.ndarray:
        """
        Process video and calculate blood flow velocity distribution.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video (flow visualization)
            mask_output_path: Path to output video (vessel masks)
            enhance: Whether to apply contrast enhancement
            method: Enhancement method
            file_name: Base name for CSV output
            
        Returns:
            np.ndarray: Mean velocity array (grid_rows, grid_cols)
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writers
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        mask_out = cv2.VideoWriter(mask_output_path, fourcc, fps, (width, height))
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate grid dimensions
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # Initialize accumulators
        region_speeds = np.zeros((self.grid_rows, self.grid_cols))
        region_speed_frames = []  # Store per-frame speeds for std calculation
        frame_count = 0
        
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract vessel mask
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = self.extract_vessel_mask(gray)
            
            # Apply enhancement if requested
            if enhance:
                enhanced_frame = self.enhance_blood_flow(frame, mask, method=method)
            else:
                enhanced_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Calculate optical flow
            next_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            flow = self.calculate_optical_flow(prev_gray, next_gray, mask)
            
            # Create visualization
            flow_vis = self.draw_flow_visualization(next_gray, flow)
            
            # Calculate per-region velocities
            frame_speeds = np.zeros((self.grid_rows, self.grid_cols))
            
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    y0, y1 = i * cell_height, (i + 1) * cell_height
                    x0, x1 = j * cell_width, (j + 1) * cell_width
                    
                    cell_mask = mask[y0:y1, x0:x1]
                    
                    if np.count_nonzero(cell_mask) > 0:
                        cell_flow = flow[y0:y1, x0:x1]
                        cell_speed = self.calculate_region_velocity(cell_flow, cell_mask)
                        region_speeds[i, j] += cell_speed
                        frame_speeds[i, j] = cell_speed
            
            region_speed_frames.append(frame_speeds)
            frame_count += 1
            prev_gray = next_gray
            
            # Add grid overlay to visualization
            avg_speeds = region_speeds / frame_count
            flow_vis = self.draw_grid_overlay(flow_vis, cell_width, cell_height, avg_speeds)
            
            # Write output videos
            out.write(flow_vis)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_out.write(mask_bgr)
        
        cap.release()
        out.release()
        mask_out.release()
        
        # Calculate statistics
        region_speeds /= frame_count
        region_speed_frames = np.array(region_speed_frames)
        region_std = np.std(region_speed_frames, axis=0)
        
        # Print results
        print(f"\nVideo processing complete. Blood flow velocity per region:")
        print("-" * 60)
        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                region_idx = i * self.grid_cols + j + 1
                avg_speed = region_speeds[i, j]
                std_speed = region_std[i, j]
                print(f"Region {region_idx} ({i},{j}): {avg_speed:.2f} ± {std_speed:.2f} μm/s")
        
        print("-" * 60)
        print(f"Overall Mean: {np.mean(region_speeds):.2f} μm/s")
        print(f"Overall Std: {np.std(region_speeds):.2f} μm/s")
        
        # Save to CSV
        csv_path = f"{file_name}_speeds.csv"
        self.save_results_to_csv(region_speeds, region_std, csv_path)
        
        return region_speeds


# Backward compatibility alias
def flowSpeed(pixel_to_um_ratio=PIXEL_TO_UM_RATIO, fps=FPS, grid_rows=GRID_ROWS, grid_cols=GRID_COLS):
    """Convenience function for backward compatibility."""
    return FlowSpeedAnalyzer(pixel_to_um_ratio, fps, grid_rows, grid_cols)
