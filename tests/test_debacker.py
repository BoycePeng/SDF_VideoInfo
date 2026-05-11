"""
Unit tests for De Backer Analyzer module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.debacker import DeBackerAnalyzer


class TestDeBackerAnalyzer(unittest.TestCase):
    """Test cases for De Backer Analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = DeBackerAnalyzer()
        # Create synthetic vessel mask (grid pattern)
        self.grid_mask = np.zeros((100, 100), dtype=np.uint8)
        # Add vertical lines
        self.grid_mask[:, 25] = 255
        self.grid_mask[:, 50] = 255
        self.grid_mask[:, 75] = 255
        # Add horizontal lines
        self.grid_mask[25, :] = 255
        self.grid_mask[50, :] = 255
        self.grid_mask[75, :] = 255
    
    def test_calculate_tvd(self):
        """Test TVD calculation."""
        tvd = self.analyzer.calculate_tvd(self.grid_mask)
        
        # TVD should be between 0 and 1 (proportion of vessel area)
        self.assertGreaterEqual(tvd, 0)
        self.assertLess(tvd, 1)
    
    def test_calculate_de_backer_score(self):
        """Test De Backer score calculation."""
        score = self.analyzer.calculate_de_backer_score(self.grid_mask)
        
        # Score should be positive
        self.assertGreater(score, 0)
    
    def test_calculate_metrics(self):
        """Test combined metrics calculation."""
        tvd, score = self.analyzer.calculate_metrics(self.grid_mask)
        
        # Both should be positive
        self.assertGreater(tvd, 0)
        self.assertGreater(score, 0)
    
    def test_empty_mask(self):
        """Test with empty mask."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        tvd = self.analyzer.calculate_tvd(empty_mask)
        self.assertEqual(tvd, 0)
    
    def test_full_mask(self):
        """Test with full vessel mask."""
        full_mask = np.ones((100, 100), dtype=np.uint8) * 255
        tvd = self.analyzer.calculate_tvd(full_mask)
        self.assertEqual(tvd, 1)


if __name__ == '__main__':
    unittest.main()
