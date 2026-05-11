#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Processing Script for SDF Microcirculation Videos

This script processes all video files in the input directory and generates
complete analysis reports including PBR, TVD, De Backer Score, and
blood flow velocity.

Usage:
    python scripts/batch_process.py --input ./data --output ./results
    
    python scripts/batch_process.py --input ./data --output ./results \
        --pixel-ratio 1.1851 --grid-rows 6 --grid-cols 6
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    PBRCalculator,
    DeBackerAnalyzer,
    FlowSpeedAnalyzer,
    VideoStabilizer
)
from config.default_config import (
    PIXEL_TO_UM_RATIO, GRID_ROWS, GRID_COLS,
    OUTPUT_PBR, OUTPUT_DEBACKER, OUTPUT_SPEEDS, OUTPUT_VIDEOS
)


class SDFBatchProcessor:
    """
    Batch processor for SDF microcirculation analysis.
    
    Processes multiple video files sequentially, generating comprehensive
    analysis results for each video.
    """
    
    # Supported video formats
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    
    def __init__(
        self,
        pixel_to_um_ratio: float = PIXEL_TO_UM_RATIO,
        grid_rows: int = GRID_ROWS,
        grid_cols: int = GRID_COLS,
        verbose: bool = True,
        advanced: bool = False
    ):
        """
        Initialize batch processor.
        
        Args:
            pixel_to_um_ratio: Calibration factor (μm/pixel)
            grid_rows: Grid rows for flow analysis
            grid_cols: Grid columns for flow analysis
            verbose: Print detailed progress
            advanced: Enable advanced stabilization mode
        """
        self.pixel_to_um_ratio = pixel_to_um_ratio
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.verbose = verbose
        self.advanced = advanced  # 是否使用 advanced 模式
        
        # Initialize analyzers
        self.stabilizer = VideoStabilizer()
        self.pbr_calc = PBRCalculator()
        self.flow_analyzer = FlowSpeedAnalyzer(
            pixel_to_um_ratio=pixel_to_um_ratio,
            grid_rows=grid_rows,
            grid_cols=grid_cols
        )
        self.debacker_analyzer = DeBackerAnalyzer()
    
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def get_video_files(self, input_dir: str) -> list:
        """
        Get list of video files in directory (auto-detect folder structure).
        
        Supports both flat directories and nested structures:
        - Flat: /data/*.mp4
        - Nested: /data/patient1/*.mp4, /data/patient2/*.mp4
        
        Args:
            input_dir: Input directory path
            
        Returns:
            list: List of tuples (video_path, relative_output_path)
        """
        video_files = []
        input_path = Path(input_dir)
        
        # Check if directory has video files directly
        direct_videos = [
            (str(f), str(f.stem))  # (full_path, video_name)
            for f in input_path.iterdir()
            if f.suffix.lower() in self.VIDEO_EXTENSIONS and f.is_file()
        ]
        
        if direct_videos:
            # Flat structure: videos directly in input_dir
            return direct_videos
        
        # Nested structure: recurse into subdirectories
        for subdir in sorted(input_path.iterdir()):
            if not subdir.is_dir():
                continue
            
            for video_file in sorted(subdir.iterdir()):
                if video_file.suffix.lower() in self.VIDEO_EXTENSIONS and video_file.is_file():
                    # Preserve subdirectory structure in output
                    relative_path = f"{subdir.name}/{video_file.stem}"
                    video_files.append((str(video_file), relative_path))
        
        return video_files
    
    def process_single_video(
        self,
        video_path: str,
        output_dir: str,
        relative_path: str = None
    ) -> dict:
        """
        Process a single video and return results.
        
        Args:
            video_path: Path to input video
            output_dir: Base output directory
            relative_path: Relative path for preserving folder structure
            
        Returns:
            dict: Processing results
        """
        video_name = Path(video_path).stem
        results = {'video_name': video_name, 'status': 'pending'}
        
        # Determine output subdirectory (preserve folder structure if nested)
        if relative_path and '/' in relative_path:
            # Extract patient/folder name from relative path
            patient_folder = relative_path.split('/')[0]
            video_output_dir = os.path.join(output_dir, 'videos', patient_folder, video_name)
        else:
            video_output_dir = os.path.join(output_dir, 'videos', video_name)
        
        try:
            self.log(f"\n{'='*60}")
            self.log(f"Processing: {video_name}")
            if relative_path:
                self.log(f"Source: {relative_path}")
            self.log(f"{'='*60}")
            
            # Create output subdirectories
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Step 1: Video Stabilization
            self.log("\n[1/4] Video Stabilization...")
            stable_path = os.path.join(video_output_dir, f'{video_name}_stable.mp4')
            compare_path = os.path.join(video_output_dir, f'{video_name}_compare.mp4')
            start_time = time.time()
            self.stabilizer.stabilize_video(video_path, stable_path, compare=True, compare_output_path=compare_path, advanced=self.advanced)
            results['stabilize_time'] = time.time() - start_time
            self.log(f"      Done: {results['stabilize_time']:.1f}s")
            
            # Step 2: PBR Calculation
            self.log("\n[2/4] Calculating PBR...")
            pbr_output_dir = os.path.join(output_dir, 'pbr')
            os.makedirs(pbr_output_dir, exist_ok=True)
            start_time = time.time()
            avg_pbr, std_pbr, _ = self.pbr_calc.process_video(
                stable_path, pbr_output_dir, self.pixel_to_um_ratio, video_name
            )
            results['pbr'] = {'mean': avg_pbr, 'std': std_pbr}
            results['pbr_time'] = time.time() - start_time
            self.log(f"      Average PBR: {avg_pbr:.2f} ± {std_pbr:.2f} μm")
            
            # Step 3: Blood Flow Velocity
            self.log("\n[3/4] Calculating Blood Flow Velocity...")
            flow_output_dir = os.path.join(video_output_dir)
            output_video = os.path.join(flow_output_dir, f'{video_name}_flow.mp4')
            mask_video = os.path.join(flow_output_dir, f'{video_name}_mask.mp4')
            start_time = time.time()
            region_speeds = self.flow_analyzer.process_video(
                stable_path, output_video, mask_video,
                enhance=True, method='hist_eq',
                file_name=os.path.join(flow_output_dir, video_name)
            )
            results['flow_mean'] = float(np.mean(region_speeds))
            results['flow_time'] = time.time() - start_time
            self.log(f"      Average Flow: {results['flow_mean']:.2f} μm/s")
            
            # Step 4: De Backer Score & TVD
            self.log("\n[4/4] Calculating De Backer Score & TVD...")
            debacker_output_dir = os.path.join(output_dir, 'debacker')
            os.makedirs(debacker_output_dir, exist_ok=True)
            start_time = time.time()
            avg_tvd, avg_debacker = self.debacker_analyzer.analyze_video(
                mask_video, self.pixel_to_um_ratio, debacker_output_dir
            )
            results['tvd'] = avg_tvd
            results['debacker'] = avg_debacker
            results['debacker_time'] = time.time() - start_time
            self.log(f"      TVD: {avg_tvd:.2f} μm")
            self.log(f"      De Backer Score: {avg_debacker:.2f}")
            
            results['status'] = 'success'
            results['total_time'] = (
                results.get('stabilize_time', 0) +
                results.get('pbr_time', 0) +
                results.get('flow_time', 0) +
                results.get('debacker_time', 0)
            )
            
            self.log(f"\n[OK] Complete! Total time: {results['total_time']:.1f}s")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.log(f"\n[ERROR] Failed: {e}")
        
        return results
    
    def process_batch(
        self,
        input_dir: str,
        output_dir: str
    ) -> list:
        """
        Process all videos in directory.
        
        Args:
            input_dir: Input directory with video files
            output_dir: Output directory for results
            
        Returns:
            list: List of processing results for each video
        """
        video_files = self.get_video_files(input_dir)
        
        if not video_files:
            print(f"No video files found in {input_dir}")
            return []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Print summary
        print("\n" + "="*60)
        print("SDF Microcirculation Batch Processing")
        print("="*60)
        print(f"Input Directory: {input_dir}")
        print(f"Output Directory: {output_dir}")
        print(f"Video Files Found: {len(video_files)}")
        print(f"Calibration Factor: {self.pixel_to_um_ratio} μm/pixel")
        print(f"Grid Size: {self.grid_rows} × {self.grid_cols}")
        print("="*60)
        
        # Process each video
        results = []
        start_time = time.time()
        
        for idx, (video_path, relative_path) in enumerate(video_files, 1):
            self.log(f"\n\n[{idx}/{len(video_files)}] Processing...")
            result = self.process_single_video(video_path, output_dir, relative_path)
            results.append(result)
        
        # Print summary
        total_elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = len(results) - success_count
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Videos: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Total Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print("="*60)
        
        # Print per-video summary
        print("\nPer-Video Results:")
        print("-"*80)
        print(f"{'Video':<40} {'PBR':<15} {'TVD':<10} {'DeBacker':<10} {'Status':<5}")
        print("-"*80)
        
        for r in results:
            pbr_str = f"{r.get('pbr', {}).get('mean', 0):.2f}±{r.get('pbr', {}).get('std', 0):.2f}"
            tvd_str = f"{r.get('tvd', 0):.2f}"
            db_str = f"{r.get('debacker', 0):.2f}"
            status_str = "[OK]" if r['status'] == 'success' else "[ERROR]"
            # Use full video_name for display (includes folder path if nested)
            name_display = r['video_name']
            print(f"{name_display:<40} {pbr_str:<15} {tvd_str:<10} {db_str:<10} {status_str:<5}")
        
        print("-"*80)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch process SDF microcirculation videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/batch_process.py --input ./data --output ./results
    
    # With custom parameters
    python scripts/batch_process.py --input ./data --output ./results \\
        --pixel-ratio 1.1851 --grid-rows 6 --grid-cols 6
    
    # Quiet mode (minimal output)
    python scripts/batch_process.py --input ./data --output ./results --quiet
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing video files'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--pixel-ratio',
        type=float,
        default=PIXEL_TO_UM_RATIO,
        help=f'Pixel to micron ratio (default: {PIXEL_TO_UM_RATIO})'
    )
    
    parser.add_argument(
        '--grid-rows',
        type=int,
        default=GRID_ROWS,
        help=f'Number of grid rows for flow analysis (default: {GRID_ROWS})'
    )
    
    parser.add_argument(
        '--grid-cols',
        type=int,
        default=GRID_COLS,
        help=f'Number of grid columns for flow analysis (default: {GRID_COLS})'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed progress output'
    )
    
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Enable advanced stabilization mode (3-pass with segmentation)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    # Create processor and run
    processor = SDFBatchProcessor(
        pixel_to_um_ratio=args.pixel_ratio,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        verbose=not args.quiet,
        advanced=args.advanced
    )
    
    processor.process_batch(args.input, args.output)


if __name__ == '__main__':
    main()
