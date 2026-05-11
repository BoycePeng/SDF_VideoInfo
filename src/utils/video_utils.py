# src/utils/video_utils.py
"""
Video I/O and processing utilities.

Provides helper classes and functions for video reading, writing,
and common video processing operations.
"""

import cv2
import numpy as np
from typing import Optional, Generator, Tuple


class VideoReader:
    """
    Video reader with frame iteration support.
    
    Provides a convenient interface for iterating through video frames
    with automatic resource management.
    
    Example:
        with VideoReader('video.mp4') as reader:
            for frame in reader:
                process(frame)
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterate through video frames."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
    
    def __len__(self) -> int:
        """Get total frame count."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def fps(self) -> float:
        """Get video frame rate."""
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get video frame size (width, height)."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read single frame.
        
        Returns:
            tuple: (success, frame)
        """
        return self.cap.read()
    
    def seek(self, frame_number: int):
        """
        Seek to specific frame.
        
        Args:
            frame_number: Target frame index (0-based)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def close(self):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()


class VideoWriter:
    """
    Video writer with context manager support.
    
    Example:
        with VideoWriter('output.mp4', fps=30, size=(640, 480)) as writer:
            for frame in frames:
                writer.write(frame)
    """
    
    def __init__(self, output_path: str, fps: float, size: Tuple[int, int],
                 codec: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video path
            fps: Frames per second
            size: Frame size (width, height)
            codec: Video codec ('mp4v', 'XVID', etc.)
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def write(self, frame: np.ndarray):
        """Write a frame."""
        self.writer.write(frame)
    
    def close(self):
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()


def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    """
    Extract the first frame from a video.
    
    Args:
        video_path: Path to video file
        
    Returns:
        np.ndarray: First frame, or None if extraction fails
    """
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None


def extract_frames(video_path: str, frame_indices: list) -> list:
    """
    Extract specific frames from a video.
    
    Args:
        video_path: Path to video file
        frame_indices: List of frame indices to extract
        
    Returns:
        list: List of extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get video file information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    cap.release()
    return info
