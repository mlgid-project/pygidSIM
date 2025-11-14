"""
Tests for CIF-based GIWAXS simulations.
"""

import pytest
import numpy as np
from pygidsim.giwaxs_sim import GIWAXSFromCif


class TestGIWAXSFromCif:
    """Test class for GIWAXSFromCif functionality."""

    def test_giwaxs_sim_random_orientation_with_mi(self, cif_file, exp_parameters):
        """Test GIWAXS simulation with random orientation and MI return."""
        el = GIWAXSFromCif(str(cif_file), exp_parameters)
        q_2d, intensity_2d, mi_2d = el.giwaxs.giwaxs_sim(
            orientation='random',
            return_mi=True,
            move_fromMW=True,
        )

        assert q_2d.shape[1] == len(intensity_2d) == len(mi_2d), \
            f"Shape mismatch in random orientation test with MI"
        assert len(intensity_2d) > 0, "No intensity data returned"
        assert len(mi_2d) > 0, "No MI data returned"

    def test_giwaxs_sim_random_orientation_with_mi_restriction(self, cif_file, exp_parameters, mi_restriction):
        """Test GIWAXS simulation with MI restriction."""
        el = GIWAXSFromCif(str(cif_file), exp_parameters)
        q_2d, intensity_2d, mi_2d = el.giwaxs.giwaxs_sim(
            orientation='random',
            max_mi=mi_restriction,
            return_mi=True,
            move_fromMW=True,
        )

        assert q_2d.shape[1] == len(intensity_2d) == len(mi_2d), \
            f"Shape mismatch in random orientation test with MI"
        assert abs(np.vstack(mi_2d)).max() <= abs(mi_restriction), \
            f"MI restriction failed"
        assert len(intensity_2d) > 0, "No intensity data returned"
        assert len(mi_2d) > 0, "No MI data returned"

    def test_giwaxs_sim_specific_orientation_no_mi(self, cif_file, exp_parameters, random_orientation):
        """Test GIWAXS simulation with specific orientation without MI return."""
        el = GIWAXSFromCif(str(cif_file), exp_parameters)
        q_2d, intensity_2d = el.giwaxs.giwaxs_sim(
            orientation=random_orientation,
            return_mi=False,
        )

        assert q_2d.shape[1] == len(intensity_2d), \
            f"Shape mismatch in specific orientation test without MI"
        assert len(intensity_2d) > 0, "No intensity data returned"

    def test_giwaxs_sim_powder_with_mi(self, cif_file, exp_parameters):
        """Test GIWAXS simulation with powder pattern and MI return."""
        el = GIWAXSFromCif(str(cif_file), exp_parameters)
        q_1d, intensity_1d, mi_1d = el.giwaxs.giwaxs_sim(
            orientation=None,
            return_mi=True,
        )

        assert len(q_1d) == len(intensity_1d) == len(mi_1d), \
            f"Shape mismatch in powder pattern test with MI"
        assert len(intensity_1d) > 0, "No intensity data returned"
        assert len(mi_1d) > 0, "No MI data returned"

    def test_giwaxs_sim_powder_no_mi(self, cif_file, exp_parameters):
        """Test GIWAXS simulation with powder pattern without MI return."""
        el = GIWAXSFromCif(str(cif_file), exp_parameters)
        q_1d, intensity_1d = el.giwaxs.giwaxs_sim(orientation=None)

        assert len(q_1d) == len(intensity_1d), \
            f"Shape mismatch in powder pattern test without MI"
        assert len(intensity_1d) > 0, "No intensity data returned"

    def test_intensity_values_positive(self, cif_file, exp_parameters):
        """Test that intensity values are non-negative."""
        el = GIWAXSFromCif(str(cif_file), exp_parameters)
        q_1d, intensity_1d = el.giwaxs.giwaxs_sim(orientation=None)

        assert np.all(intensity_1d >= 0), "Negative intensity values found"

    def test_q_values_positive(self, cif_file, exp_parameters):
        """Test that q values are positive."""
        el = GIWAXSFromCif(str(cif_file), exp_parameters)
        q_1d, intensity_1d = el.giwaxs.giwaxs_sim(orientation=None)

        assert np.all(q_1d > 0), "Non-positive q values found"
