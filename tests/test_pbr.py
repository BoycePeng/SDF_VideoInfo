"""
Unit tests for PBR Calculator module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.pbr import PBRCalculator


class TestPBRCalculator(unittest.TestCase):
    """Test cases for PBR Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PBRCalculator()
        # Create synthetic vessel mask (simple rectangle)
        self.simple_mask = np.zeros((100, 100), dtype=np.uint8)
        self.simple_mask[20:80, 20:80] = 255
        
        # Create synthetic RBC mask (smaller rectangle inside)
        self.rbc_mask = np.zeros((100, 100), dtype=np.uint8)
        self.rbc_mask[30:70, 30:70] = 255
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create test grayscale image
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        filled, masked = self.calculator.preprocess_image(gray)
        
        # Check output shapes
        self.assertEqual(filled.shape, gray.shape)
        self.assertEqual(masked.shape, gray.shape)
        
        # Check output is binary
        unique_vals = np.unique(filled)
        self.assertTrue(all(v in [0, 255] for v in unique_vals))
    
    def test_extract_rbc_mask(self):
        """Test RBC mask extraction."""
        enhanced = np.random.randint(0, 200, (100, 100), dtype=np.uint8)
        rbc_mask = self.calculator.extract_rbc_mask(enhanced)
        
        # Check output is binary
        unique_vals = np.unique(rbc_mask)
        self.assertTrue(all(v in [0, 255] for v in unique_vals))
    
    def test_calculate_pbr(self):
        """Test PBR calculation."""
        pbr = self.calculator.calculate_pbr(self.simple_mask, self.rbc_mask)
        
        # PBR should be positive (vessel larger than RBC)
        self.assertGreater(pbr, 0)
    
    def test_process_binary_mask(self):
        """Test skeletonization and distance transform."""
        skeleton, mean_dist, dt = self.calculator.process_binary_mask(self.simple_mask)
        
        # Check outputs
        self.assertEqual(skeleton.shape, self.simple_mask.shape)
        self.assertEqual(dt.shape, self.simple_mask.shape)
        self.assertGreater(mean_dist, 0)


if __name__ == '__main__':
    unittest.main()
