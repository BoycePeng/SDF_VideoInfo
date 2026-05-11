"""
Unit tests for Flow Speed Analyzer module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.flow_speed import FlowSpeedAnalyzer


class TestFlowSpeedAnalyzer(unittest.TestCase):
    """Test cases for Flow Speed Analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FlowSpeedAnalyzer(grid_rows=3, grid_cols=3)
        # Create synthetic frames
        self.frame1 = np.zeros((100, 100), dtype=np.uint8)
        self.frame2 = np.zeros((100, 100), dtype=np.uint8)
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.grid_rows, 3)
        self.assertEqual(self.analyzer.grid_cols, 3)
    
    def test_extract_vessel_mask(self):
        """Test vessel mask extraction."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mask = self.analyzer.extract_vessel_mask(gray)
        
        # Check output is binary
        unique_vals = np.unique(mask)
        self.assertTrue(all(v in [0, 255] for v in unique_vals))
    
    def test_enhance_blood_flow_no_enhancement(self):
        """Test enhancement with None method."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        result = self.analyzer.enhance_blood_flow(image, mask, method=None)
        
        # Should return original image
        self.assertEqual(result.shape, image.shape)
    
    def test_draw_flow_visualization(self):
        """Test flow visualization."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        
        # Add some flow
        flow[50:60, 50:60, 0] = 5
        flow[50:60, 50:60, 1] = 5
        
        vis = self.analyzer.draw_flow_visualization(gray, flow)
        
        # Check output is BGR
        self.assertEqual(vis.shape[2], 3)
    
    def test_calculate_region_velocity(self):
        """Test velocity calculation."""
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        mask = np.ones((100, 100), dtype=np.uint8)
        
        # Add uniform flow
        flow[:, :, 0] = 5
        flow[:, :, 1] = 0
        
        velocity = self.analyzer.calculate_region_velocity(flow, mask)
        
        # Velocity should be positive
        self.assertGreater(velocity, 0)


if __name__ == '__main__':
    unittest.main()
