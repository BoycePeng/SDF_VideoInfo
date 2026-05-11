"""
Video Stabilization Module

This module stabilizes shaky video recordings using optical flow-based
feature tracking and trajectory smoothing.

Algorithms:
    1. Feature Detection: Shi-Tomasi corner detection
    2. Optical Flow Tracking: Lucas-Kanade method
    3. Transform Estimation: Affine transformation between frames
    4. Trajectory Smoothing: Moving average filter
    5. Compensation: Apply smoothed transforms
    6. Advanced Mode: Split-restabilize (3-pass stabilization)

The advanced mode splits the video into segments by frame index modulo,
stabilizes each segment separately, concatenates, then stabilizes again.
This provides better results for complex jitter patterns.

Reference:
    - Bouguet, J.Y. "Pyramidal Implementation of the Lucas Kanade Feature Tracker"
    - Shi, J. and Tomasi, C. "Good Features to Track"
"""

import os
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from config.default_config import (
    MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, BLOCK_SIZE,
    USE_HARRIS, HARRIS_K, SMOOTHING_RADIUS, CROP_PERCENT,
    FPS, VIDEO_CODEC, FRAME_INTERVAL, ADVANCED_STABILIZE, TEMP_DIR
)


class VideoStabilizer:
    """
    Video stabilizer using optical flow-based motion estimation.

    Supports both standard single-pass stabilization and advanced
    split-restabilize mode (3-pass) for better results.

    Attributes:
        max_corners: Maximum features to track
        quality_level: Minimum feature quality
        min_distance: Minimum distance between features
        smoothing_radius: Window size for trajectory smoothing
        crop_percent: Border percentage to crop after stabilization
        frame_interval: Interval for split-stabilize (advanced mode)
        advanced_stabilize: Whether to use advanced mode by default
        temp_dir: Temporary directory for intermediate files
    """

    def __init__(
        self,
        max_corners: int = MAX_CORNERS,
        quality_level: float = QUALITY_LEVEL,
        min_distance: int = MIN_DISTANCE,
        smoothing_radius: int = SMOOTHING_RADIUS,
        crop_percent: float = CROP_PERCENT,
        frame_interval: int = FRAME_INTERVAL,
        advanced_stabilize: bool = ADVANCED_STABILIZE,
        temp_dir: str = TEMP_DIR
    ):
        """
        Initialize video stabilizer.

        Args:
            max_corners: Maximum number of features to detect
            quality_level: Feature quality threshold (0.01 = 1%)
            min_distance: Minimum pixel distance between features
            smoothing_radius: Moving average window radius
            crop_percent: Border crop percentage after stabilization
            frame_interval: Frame interval for split-stabilize (advanced mode)
            advanced_stabilize: Default advanced mode flag
            temp_dir: Temporary directory path for intermediate files
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.smoothing_radius = smoothing_radius
        self.crop_percent = crop_percent
        self.frame_interval = frame_interval
        self.advanced_stabilize = advanced_stabilize
        self.temp_dir = temp_dir

    def detect_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Detect good features to track using Shi-Tomasi algorithm.

        Args:
            gray_image: Grayscale image

        Returns:
            np.ndarray: Feature points array (N, 1, 2)
        """
        features = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=BLOCK_SIZE,
            useHarrisDetector=USE_HARRIS,
            k=HARRIS_K
        )
        return features

    def estimate_transform(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray
    ) -> np.ndarray:
        """
        Estimate affine transformation between two point sets.

        Args:
            prev_pts: Previous frame feature points
            curr_pts: Current frame feature points

        Returns:
            np.ndarray: 2x3 affine transformation matrix
        """
        transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

        if transform is None:
            transform = np.eye(2, 3, dtype=np.float32)

        return transform

    def decompose_transform(self, transform: np.ndarray) -> tuple:
        """
        Decompose affine transform into translation and rotation.

        Args:
            transform: 2x3 affine transformation matrix

        Returns:
            tuple: (dx, dy, da) - translations and rotation angle
        """
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])

        return dx, dy, da

    def _smoothing_kernel(self, curve: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing to a 1D trajectory curve.

        Args:
            curve: Input 1D trajectory array

        Returns:
            np.ndarray: Smoothed trajectory array (same length as input)
        """
        window_size = 2 * self.smoothing_radius + 1
        kernel = np.ones(window_size) / window_size

        curve_padded = np.pad(
            curve, (self.smoothing_radius, self.smoothing_radius), 'edge'
        )
        smoothed = np.convolve(curve_padded, kernel, mode='same')
        smoothed = smoothed[self.smoothing_radius:-self.smoothing_radius]

        return smoothed

    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Smooth entire trajectory (x, y, angle components).

        Args:
            trajectory: Cumulative trajectory array (N, 3)

        Returns:
            np.ndarray: Smoothed trajectory (N, 3)
        """
        smoothed = np.copy(trajectory)

        for i in range(3):
            smoothed[:, i] = self._smoothing_kernel(trajectory[:, i])

        return smoothed

    def _fix_border(self, frame: np.ndarray) -> np.ndarray:
        """
        Fix border artifacts from affine transformation.

        Applies a minimal rotation+scale warp to crop black borders
        that appear after stabilization transforms.

        Args:
            frame: Input frame (BGR or grayscale)

        Returns:
            np.ndarray: Frame with fixed borders
        """
        s = frame.shape
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))

        return frame

    def _stabilize_once(
        self,
        input_path: str,
        output_path: str,
        compare: bool = False,
        compare_output_path: str = None
    ) -> str:
        """
        Perform a single-pass video stabilization.

        This is the core stabilization logic, extracted so it can be
        reused by both standard and advanced (split-restabilize) modes.

        Args:
            input_path: Path to input video
            output_path: Path to output stabilized video
            compare: If True, also output side-by-side comparison video
            compare_output_path: Custom path for compare video

        Returns:
            str: Path to stabilized video
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        _, prev = cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # Step 1: Estimate transforms between consecutive frames
        transforms = np.zeros((n_frames - 1, 3), np.float32)

        for i in range(n_frames - 1):
            prev_pts = self.detect_features(prev_gray)

            success, curr = cap.read()
            if not success:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None
            )

            idx = np.where(status == 1)[0]
            if len(idx) == 0:
                # No tracked points, use identity transform
                transforms[i] = [0.0, 0.0, 0.0]
                prev_gray = curr_gray
                continue

            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            m = self.estimate_transform(prev_pts, curr_pts)
            dx, dy, da = self.decompose_transform(m)

            transforms[i] = [dx, dy, da]
            prev_gray = curr_gray

        cap.release()

        # Step 2: Smooth trajectory
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = self._smooth_trajectory(trajectory)
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference

        # Step 3: Apply smoothed transforms
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

        if compare:
            if compare_output_path is None:
                compare_output_path = output_path.replace('.mp4', '_compare.mp4')
            compare_writer = cv2.VideoWriter(
                compare_output_path, fourcc, fps, (2 * width, height)
            )

        stable_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(n_frames - 1):
            success, frame = cap.read()
            if not success:
                break

            dx = transforms_smooth[i, 0]
            dy = transforms_smooth[i, 1]
            da = transforms_smooth[i, 2]

            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            frame_stabilized = cv2.warpAffine(frame, m, (width, height))
            frame_stabilized = self._fix_border(frame_stabilized)

            if compare:
                compare_frame = cv2.hconcat([frame, frame_stabilized])
                if compare_frame.shape[1] > 1920:
                    compare_frame = cv2.resize(
                        compare_frame,
                        (compare_frame.shape[1] // 2, compare_frame.shape[0] // 2)
                    )
                compare_writer.write(compare_frame)

            stable_writer.write(frame_stabilized)

        # Handle last frame
        if n_frames > 1:
            _, last_frame = cap.read()
            if success:
                m = np.zeros((2, 3), np.float32)
                m[0, 0] = np.cos(transforms_smooth[n_frames - 2, 2])
                m[0, 1] = -np.sin(transforms_smooth[n_frames - 2, 2])
                m[1, 0] = np.sin(transforms_smooth[n_frames - 2, 2])
                m[1, 1] = np.cos(transforms_smooth[n_frames - 2, 2])
                m[0, 2] = transforms_smooth[n_frames - 2, 0]
                m[1, 2] = transforms_smooth[n_frames - 2, 1]

                last_frame_stabilized = cv2.warpAffine(
                    last_frame, m, (width, height)
                )
                last_frame_stabilized = self._fix_border(last_frame_stabilized)

                if compare:
                    compare_writer.write(
                        cv2.hconcat([last_frame, last_frame_stabilized])
                    )
                stable_writer.write(last_frame_stabilized)

        cap.release()
        stable_writer.release()
        if compare:
            compare_writer.release()

        return output_path

    def _split_video(self, input_path: str, output_dir: str):
        """
        Split video into segments by frame index modulo frame_interval.

        Frames are distributed as:
            segment 0: frames 0, interval, 2*interval, ...
            segment 1: frames 1, interval+1, 2*interval+1, ...
            etc.

        Each segment is written as a separate video file.

        Args:
            input_path: Path to input video
            output_dir: Directory to write segment videos

        Returns:
            list: Paths to segment video files
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = self.frame_interval

        # Initialize writers for each segment
        writers = []
        seg_paths = []
        for i in range(interval):
            seg_path = os.path.join(output_dir, f'segment_{i}.mp4')
            writer = cv2.VideoWriter(seg_path, fourcc, fps, (width, height))
            writers.append(writer)
            seg_paths.append(seg_path)

        # Read and distribute frames
        frame_count = 0
        success, frame = cap.read()
        while success and frame_count < min(n_frames, 320):  # Match original: cap at 320
            seg_idx = frame_count % interval
            writers[seg_idx].write(frame)

            success, frame = cap.read()
            frame_count += 1

        # Release all
        cap.release()
        for w in writers:
            w.release()

        return seg_paths

    def _concat_videos(self, seg_paths: list, output_path: str,
                       apply_enhance: bool = True) -> str:
        """
        Concatenate multiple video segments into one video.

        Frames are interleaved: all segment 0 frame 0, then all segment 1
        frame 0, etc., then all segment 0 frame 1, etc.

        Args:
            seg_paths: List of paths to segment videos
            output_path: Path for output concatenated video
            apply_enhance: Whether to apply image enhancement

        Returns:
            str: Path to concatenated video
        """
        # Read all frames from all segments
        all_frames = []
        for seg_path in seg_paths:
            cap = cv2.VideoCapture(seg_path)
            frames = []
            success, frame = cap.read()
            while success:
                frames.append(frame)
                success, frame = cap.read()
            cap.release()
            all_frames.append(frames)

        # Interleave frames: for each frame index, take from each segment
        max_len = max(len(f) for f in all_frames)
        concatenated = []
        for frame_idx in range(max_len):
            for seg_idx in range(len(all_frames)):
                if frame_idx < len(all_frames[seg_idx]):
                    concatenated.append(all_frames[seg_idx][frame_idx])

        # Write concatenated video
        if not concatenated:
            raise RuntimeError("No frames to concatenate")

        height, width = concatenated[0].shape[:2]
        fps = 30  # Default, will be overridden by caller if needed

        # Get fps from first segment
        cap = cv2.VideoCapture(seg_paths[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in concatenated:
            if apply_enhance:
                # Apply sharpening enhancement (matching original logic)
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_img = self._apply_enhancement(pil_img, sharpness=1.5)
                frame = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

            out.write(frame)

        out.release()
        return output_path

    def _crop_video(self, input_path: str, output_path: str) -> str:
        """
        Crop video borders by crop_percent.

        Args:
            input_path: Path to input video
            output_path: Path for cropped output video

        Returns:
            str: Path to cropped video
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        crop_pixels_h = int(height * self.crop_percent)
        crop_pixels_w = int(width * self.crop_percent)
        new_width = width - 2 * crop_pixels_w
        new_height = height - 2 * crop_pixels_h

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (new_width, new_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cropped = frame[
                crop_pixels_h:height - crop_pixels_h,
                crop_pixels_w:width - crop_pixels_w
            ]
            out.write(cropped)

        cap.release()
        out.release()

        return output_path

    def _apply_enhancement(self, pil_image, brightness=1, color=1,
                           contrast=1, sharpness=1):
        """
        Apply PIL-based image enhancement.

        Args:
            pil_image: PIL Image object
            brightness: Brightness enhancement factor
            color: Color enhancement factor
            contrast: Contrast enhancement factor
            sharpness: Sharpness enhancement factor

        Returns:
            PIL.Image: Enhanced image
        """
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)

        if color != 1:
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(color)

        if contrast != 1:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)

        if sharpness != 1:
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(sharpness)

        return pil_image

    def stabilize_video(
        self,
        input_path: str,
        output_path: str,
        compare: bool = False,
        compare_output_path: str = None,
        advanced: bool = None
    ) -> str:
        """
        Stabilize a video file.

        Args:
            input_path: Path to input video
            output_path: Path to output stabilized video
            compare: If True, also output side-by-side comparison video
            compare_output_path: Custom path for compare video
            advanced: If True, use advanced split-restabilize mode.
                      If None, use self.advanced_stabilize.

        Returns:
            str: Path to stabilized video
        """
        use_advanced = advanced if advanced is not None else self.advanced_stabilize

        if use_advanced:
            return self.stabilize_video_advanced(input_path, output_path, compare, compare_output_path)
        else:
            return self._stabilize_once(
                input_path, output_path, compare, compare_output_path
            )

    def stabilize_video_advanced(
        self,
        input_path: str,
        output_path: str,
        compare: bool = False,
        compare_output_path: str = None
    ) -> str:
        """
        Advanced 3-pass stabilization with split-restabilize.

        Pipeline (matching original SDF_Paper implementation):
            1. First pass: stabilize entire video
            2. Split: divide stabilized video into frame_interval segments
            3. Second pass: stabilize each segment separately
            4. Concat: interleave segments back into one video
            5. Third pass: stabilize concatenated video
            6. Crop: crop borders of final video

        Args:
            input_path: Path to input video
            output_path: Path for final stabilized output video
            compare: If True, output comparison video (first pass only)
            compare_output_path: Custom path for compare video

        Returns:
            str: Path to final stabilized video
        """
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix='sdf_stab_')
        seg_dir = os.path.join(temp_dir, 'segments')
        os.makedirs(seg_dir, exist_ok=True)

        try:
            # Step 1: First pass stabilization
            stable_first = os.path.join(temp_dir, 'stable_first.mp4')
            self._stabilize_once(input_path, stable_first, compare, compare_output_path)

            # Step 2: Split video into segments
            seg_paths = self._split_video(stable_first, seg_dir)

            # Step 3: Second pass - stabilize each segment
            seg_stable_paths = []
            for i, seg_path in enumerate(seg_paths):
                seg_stable = os.path.join(temp_dir, f'stable_seg_{i}.mp4')
                self._stabilize_once(seg_path, seg_stable)
                seg_stable_paths.append(seg_stable)

            # Step 4: Concatenate stabilized segments
            concat_path = os.path.join(temp_dir, 'concat.mp4')
            self._concat_videos(seg_stable_paths, concat_path, apply_enhance=True)

            # Step 5: Third pass stabilization
            stable_third = os.path.join(temp_dir, 'stable_third.mp4')
            self._stabilize_once(concat_path, stable_third)

            # Step 6: Crop borders
            self._crop_video(stable_third, output_path)

        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        return output_path


# Backward compatibility function
def stabilize_video(input_video_path, output_video_path):
    """
    Convenience function for backward compatibility.

    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
    """
    stabilizer = VideoStabilizer()
    return stabilizer.stabilize_video(input_video_path, output_video_path)
