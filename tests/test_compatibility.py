"""
Tests the compatibility with the v0.1.1.
"""

import pytest
import numpy as np
from pygidsim.experiment import ExpParameters


class TestExpParameters:
    """Test class for Experimental parameters validation."""

    def test_q_range(self):
        """Test that intensity values are non-negative."""
        exp_par_v011 = ExpParameters(
            q_xy_max=2.7,
            q_z_max=3.1,
        )

        exp_par_v012 = ExpParameters(
            q_xy_range=(0, 2.7),
            q_z_range=(0, 3.1),
        )

        assert exp_par_v011.q_xy_range == exp_par_v012.q_xy_range, "q_xy_range mismatch"
        assert exp_par_v011.q_z_range == exp_par_v012.q_z_range, "q_z_range mismatch"
