#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Video Processing Script for SDF Microcirculation Analysis

This script processes a single video file and generates complete analysis
reports including PBR, TVD, De Backer Score, and blood flow velocity.

Usage:
    python scripts/single_process.py --input video.mp4 --output ./results
    
    python scripts/single_process.py --input video.mp4 --output ./results \\
        --pixel-ratio 1.1851 --enhance
"""

import argparse
import os
import sys
import time
from pathlib import Path

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
    PIXEL_TO_UM_RATIO, GRID_ROWS, GRID_COLS
)


def process_video(
    input_path: str,
    output_dir: str,
    pixel_to_um_ratio: float = PIXEL_TO_UM_RATIO,
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    enhance: bool = True,
    enhance_method: str = 'hist_eq',
    verbose: bool = True,
    advanced: bool = False
) -> dict:
    """
    Process a single SDF microcirculation video.
    
    Pipeline:
        1. Video Stabilization
        2. PBR Calculation
        3. Blood Flow Velocity
        4. De Backer Score & TVD
    
    Args:
        input_path: Path to input video
        output_dir: Output directory for results
        pixel_to_um_ratio: Calibration factor (μm/pixel)
        grid_rows: Grid rows for flow analysis
        grid_cols: Grid columns for flow analysis
        enhance: Apply contrast enhancement
        enhance_method: Enhancement method ('clahe', 'hist_eq', 'gamma')
        verbose: Print detailed progress
        
    Returns:
        dict: Processing results
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video file not found: {input_path}")
    
    video_name = Path(input_path).stem
    results = {'video_name': video_name}
    
    # Create output directories
    video_output_dir = os.path.join(output_dir, 'videos', video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Initialize analyzers
    stabilizer = VideoStabilizer()
    pbr_calc = PBRCalculator()
    flow_analyzer = FlowSpeedAnalyzer(
        pixel_to_um_ratio=pixel_to_um_ratio,
        grid_rows=grid_rows,
        grid_cols=grid_cols
    )
    debacker_analyzer = DeBackerAnalyzer()
    
    total_start = time.time()
    
    # =========================================================================
    # Step 1: Video Stabilization
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("SDF Microcirculation Video Analysis")
        print("="*60)
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")
        print(f"Pixel Ratio: {pixel_to_um_ratio} μm/pixel")
        print(f"Grid: {grid_rows} × {grid_cols}")
        print("="*60)
        print("\n[Step 1/4] Video Stabilization...")
    
    stable_path = os.path.join(video_output_dir, f'{video_name}_stable.mp4')
    compare_path = os.path.join(video_output_dir, f'{video_name}_compare.mp4')
    start_time = time.time()
    stabilizer.stabilize_video(input_path, stable_path, compare=True, compare_output_path=compare_path, advanced=advanced)
    results['stabilize_time'] = time.time() - start_time
    results['stable_path'] = stable_path
    results['compare_path'] = compare_path
    
    if verbose:
        print(f"           Done in {results['stabilize_time']:.1f}s")
    
    # =========================================================================
    # Step 2: PBR Calculation
    # =========================================================================
    if verbose:
        print("\n[Step 2/4] Calculating PBR (Perfused Boundary Region)...")
    
    pbr_output_dir = os.path.join(output_dir, 'pbr')
    os.makedirs(pbr_output_dir, exist_ok=True)
    
    start_time = time.time()
    avg_pbr, std_pbr, _ = pbr_calc.process_video(
        stable_path, pbr_output_dir, pixel_to_um_ratio, video_name
    )
    results['pbr'] = {'mean': avg_pbr, 'std': std_pbr}
    results['pbr_time'] = time.time() - start_time
    
    if verbose:
        print(f"           Average PBR: {avg_pbr:.2f} ± {std_pbr:.2f} μm")
        print(f"           Done in {results['pbr_time']:.1f}s")
    
    # =========================================================================
    # Step 3: Blood Flow Velocity
    # =========================================================================
    if verbose:
        print("\n[Step 3/4] Calculating Blood Flow Velocity...")
        print(f"           Enhancement: {enhance}, Method: {enhance_method}")
    
    output_video = os.path.join(video_output_dir, f'{video_name}_flow.mp4')
    mask_video = os.path.join(video_output_dir, f'{video_name}_mask.mp4')
    speed_csv_path = os.path.join(video_output_dir, f'{video_name}_speeds.csv')
    
    start_time = time.time()
    region_speeds = flow_analyzer.process_video(
        stable_path, output_video, mask_video,
        enhance=enhance,
        method=enhance_method,
        file_name=speed_csv_path.replace('.csv', '')
    )
    results['flow'] = {
        'mean': float(np.mean(region_speeds)),
        'std': float(np.std(region_speeds)),
        'per_region': region_speeds.tolist()
    }
    results['flow_time'] = time.time() - start_time
    
    if verbose:
        print(f"           Average Flow: {results['flow']['mean']:.2f} μm/s")
        print(f"           Done in {results['flow_time']:.1f}s")
    
    # =========================================================================
    # Step 4: De Backer Score & TVD
    # =========================================================================
    if verbose:
        print("\n[Step 4/4] Calculating De Backer Score & TVD...")
    
    debacker_output_dir = os.path.join(output_dir, 'debacker')
    os.makedirs(debacker_output_dir, exist_ok=True)
    
    start_time = time.time()
    avg_tvd, avg_debacker = debacker_analyzer.analyze_video(
        mask_video, pixel_to_um_ratio, debacker_output_dir
    )
    results['tvd'] = avg_tvd
    results['debacker'] = avg_debacker
    results['debacker_time'] = time.time() - start_time
    
    if verbose:
        print(f"           TVD: {avg_tvd:.2f} μm")
        print(f"           De Backer Score: {avg_debacker:.2f}")
        print(f"           Done in {results['debacker_time']:.1f}s")
    
    # =========================================================================
    # Summary
    # =========================================================================
    results['total_time'] = time.time() - total_start
    
    if verbose:
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total Time: {results['total_time']:.1f}s")
        print("\nResults Summary:")
        print(f"  PBR:           {avg_pbr:.2f} ± {std_pbr:.2f} μm")
        print(f"  TVD:           {avg_tvd:.2f} μm")
        print(f"  De Backer:     {avg_debacker:.2f}")
        print(f"  Flow Velocity: {results['flow']['mean']:.2f} μm/s")
        print("\nOutput Files:")
        print(f"  {pbr_output_dir}/{video_name}_pbr_results.csv")
        print(f"  {debacker_output_dir}/{video_name}_debacker_analysis_results.csv")
        print(f"  {speed_csv_path}")
        print(f"  {video_output_dir}/{video_name}_stable.mp4")
        print(f"  {video_output_dir}/{video_name}_mask.mp4")
        print("="*60)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process single SDF microcirculation video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/single_process.py --input video.mp4 --output ./results
    
    # With enhancement
    python scripts/single_process.py --input video.mp4 --output ./results --enhance
    
    # Custom parameters
    python scripts/single_process.py --input video.mp4 --output ./results \\
        --pixel-ratio 1.1851 --grid-rows 6 --grid-cols 6 --enhance --method clahe
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input video file path'
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
        help=f'Number of grid rows (default: {GRID_ROWS})'
    )
    
    parser.add_argument(
        '--grid-cols',
        type=int,
        default=GRID_COLS,
        help=f'Number of grid columns (default: {GRID_COLS})'
    )
    
    parser.add_argument(
        '--enhance', '-e',
        action='store_true',
        help='Apply contrast enhancement to blood flow'
    )
    
    parser.add_argument(
        '--method', '-m',
        choices=['clahe', 'hist_eq', 'gamma'],
        default='hist_eq',
        help='Contrast enhancement method (default: hist_eq)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )

    parser.add_argument(
        '--advanced', '-a',
        action='store_true',
        help='Enable advanced split-restabilize mode (3-pass stabilization)'
    )

    args = parser.parse_args()
    
    # Process video
    try:
        results = process_video(
            input_path=args.input,
            output_dir=args.output,
            pixel_to_um_ratio=args.pixel_ratio,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            enhance=args.enhance,
            enhance_method=args.method,
            verbose=not args.quiet,
            advanced=args.advanced
        )
        
        print("\n[OK] Processing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
