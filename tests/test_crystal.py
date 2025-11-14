"""
Tests for Crystal-based GIWAXS simulations.
"""
import pytest
import numpy as np
from pygidsim.giwaxs_sim import GIWAXS


class TestCrystalGIWAXS:
    """Test class for Crystal-based GIWAXS functionality."""

    def test_crystal_with_occupancy_random_orientation_with_mi(self, test_crystal, exp_parameters):
        """Test GIWAXS simulation with crystal including occupancy."""
        el = GIWAXS(test_crystal, exp_parameters)
        q_2d, intensity_2d, mi_2d = el.giwaxs_sim(
            orientation='random',
            return_mi=True,
            move_fromMW=True,
        )

        assert q_2d.shape[1] == len(intensity_2d) == len(mi_2d), \
            "Shape mismatch in crystal test with occupancy"
        assert len(intensity_2d) > 0, "No intensity data returned"
        assert len(mi_2d) > 0, "No MI data returned"

    def test_crystal_with_occupancy_random_orientation_with_mi_restriction(self, test_crystal, exp_parameters,
                                                                           mi_restriction):
        """Test GIWAXS simulation with MI restriction."""
        el = GIWAXS(test_crystal, exp_parameters)
        q_2d, intensity_2d, mi_2d = el.giwaxs_sim(
            orientation='random',
            max_mi=mi_restriction,
            return_mi=True,
            move_fromMW=True,
        )

        assert q_2d.shape[1] == len(intensity_2d) == len(mi_2d), \
            "Shape mismatch in crystal test with occupancy"
        assert abs(np.vstack(mi_2d)).max() <= abs(mi_restriction), \
            f"MI restriction failed"
        assert len(intensity_2d) > 0, "No intensity data returned"
        assert len(mi_2d) > 0, "No MI data returned"

    def test_crystal_with_occupancy_specific_orientation_no_mi(self, test_crystal, exp_parameters, random_orientation):
        """Test GIWAXS simulation with specific orientation without MI return."""
        el = GIWAXS(test_crystal, exp_parameters)
        q_2d, intensity_2d = el.giwaxs_sim(
            orientation=random_orientation,
            return_mi=False,
        )

        assert q_2d.shape[1] == len(intensity_2d), \
            "Shape mismatch in specific orientation test"
        assert len(intensity_2d) > 0, "No intensity data returned"

    def test_crystal_with_occupancy_powder_with_mi(self, test_crystal, exp_parameters):
        """Test GIWAXS simulation with powder pattern and MI return."""
        el = GIWAXS(test_crystal, exp_parameters)
        q_1d, intensity_1d, mi_1d = el.giwaxs_sim(
            orientation=None,
            return_mi=True,
        )

        assert len(q_1d) == len(intensity_1d) == len(mi_1d), \
            "Shape mismatch in powder pattern test with MI"
        assert len(intensity_1d) > 0, "No intensity data returned"
        assert len(mi_1d) > 0, "No MI data returned"

    def test_crystal_with_occupancy_powder_no_mi(self, test_crystal, exp_parameters):
        """Test GIWAXS simulation with powder pattern without MI return."""
        el = GIWAXS(test_crystal, exp_parameters)
        q_1d, intensity_1d = el.giwaxs_sim(orientation=None)

        assert len(q_1d) == len(intensity_1d), \
            "Shape mismatch in powder pattern test without MI"
        assert len(intensity_1d) > 0, "No intensity data returned"


class TestCrystalWithoutOccupancy:
    """Test class for Crystal without occupancy GIWAXS functionality."""

    def test_crystal_no_occupancy_random_orientation_with_mi(self, test_crystal_no_occupancy, exp_parameters):
        """Test GIWAXS simulation with crystal without occupancy."""
        el = GIWAXS(test_crystal_no_occupancy, exp_parameters)
        q_2d, intensity_2d, mi_2d = el.giwaxs_sim(
            orientation='random',
            return_mi=True,
            move_fromMW=True,
        )

        assert q_2d.shape[1] == len(intensity_2d) == len(mi_2d), \
            "Shape mismatch in crystal test without occupancy"
        assert len(intensity_2d) > 0, "No intensity data returned"
        assert len(mi_2d) > 0, "No MI data returned"

    def test_crystal_no_occupancy_specific_orientation_no_mi(self, test_crystal_no_occupancy, exp_parameters,
                                                             random_orientation):
        """Test GIWAXS simulation with specific orientation without MI return."""
        el = GIWAXS(test_crystal_no_occupancy, exp_parameters)
        q_2d, intensity_2d = el.giwaxs_sim(
            orientation=random_orientation,
            return_mi=False,
        )

        assert q_2d.shape[1] == len(intensity_2d), \
            "Shape mismatch in specific orientation test"
        assert len(intensity_2d) > 0, "No intensity data returned"

    def test_crystal_no_occupancy_powder_with_mi(self, test_crystal_no_occupancy, exp_parameters):
        """Test GIWAXS simulation with powder pattern and MI return."""
        el = GIWAXS(test_crystal_no_occupancy, exp_parameters)
        q_1d, intensity_1d, mi_1d = el.giwaxs_sim(
            orientation=None,
            return_mi=True,
        )

        assert len(q_1d) == len(intensity_1d) == len(mi_1d), \
            "Shape mismatch in powder pattern test with MI"
        assert len(intensity_1d) > 0, "No intensity data returned"
        assert len(mi_1d) > 0, "No MI data returned"

    def test_crystal_no_occupancy_powder_no_mi(self, test_crystal_no_occupancy, exp_parameters):
        """Test GIWAXS simulation with powder pattern without MI return."""
        el = GIWAXS(test_crystal_no_occupancy, exp_parameters)
        q_1d, intensity_1d = el.giwaxs_sim(orientation=None)

        assert len(q_1d) == len(intensity_1d), \
            "Shape mismatch in powder pattern test without MI"
        assert len(intensity_1d) > 0, "No intensity data returned"


class TestCrystalWithoutAtoms:
    """Test class for Crystal without atoms GIWAXS functionality."""

    def test_crystal_no_atoms_random_orientation_with_mi(self, test_crystal_no_atoms, exp_parameters):
        """Test GIWAXS simulation with crystal without atoms."""
        el = GIWAXS(test_crystal_no_atoms, exp_parameters)
        q_2d, intensity_2d, mi_2d = el.giwaxs_sim(
            orientation='random',
            return_mi=True,
            move_fromMW=True,
        )

        assert q_2d.shape[1] == len(intensity_2d) == len(mi_2d), \
            "Shape mismatch in crystal test without atoms"
        assert len(intensity_2d) > 0, "No intensity data returned"
        assert len(mi_2d) > 0, "No MI data returned"

    def test_crystal_no_atoms_specific_orientation_no_mi(self, test_crystal_no_atoms, exp_parameters,
                                                         random_orientation):
        """Test GIWAXS simulation with specific orientation without MI return."""
        el = GIWAXS(test_crystal_no_atoms, exp_parameters)
        q_2d, intensity_2d = el.giwaxs_sim(
            orientation=random_orientation,
            return_mi=False,
        )

        assert q_2d.shape[1] == len(intensity_2d), \
            "Shape mismatch in specific orientation test"
        assert len(intensity_2d) > 0, "No intensity data returned"

    def test_crystal_no_atoms_powder_with_mi(self, test_crystal_no_atoms, exp_parameters):
        """Test GIWAXS simulation with powder pattern and MI return."""
        el = GIWAXS(test_crystal_no_atoms, exp_parameters)
        q_1d, intensity_1d, mi_1d = el.giwaxs_sim(
            orientation=None,
            return_mi=True,
        )

        assert len(q_1d) == len(intensity_1d) == len(mi_1d), \
            "Shape mismatch in powder pattern test with MI"
        assert len(intensity_1d) > 0, "No intensity data returned"
        assert len(mi_1d) > 0, "No MI data returned"

    def test_crystal_no_atoms_powder_no_mi(self, test_crystal_no_atoms, exp_parameters):
        """Test GIWAXS simulation with powder pattern without MI return."""
        el = GIWAXS(test_crystal_no_atoms, exp_parameters)
        q_1d, intensity_1d = el.giwaxs_sim(orientation=None)

        assert len(q_1d) == len(intensity_1d), \
            "Shape mismatch in powder pattern test without MI"
        assert len(intensity_1d) > 0, "No intensity data returned"


class TestCrystalValidation:
    """Test class for Crystal data validation."""

    def test_intensity_values_positive(self, test_crystal, exp_parameters):
        """Test that intensity values are non-negative."""
        el = GIWAXS(test_crystal, exp_parameters)
        q_1d, intensity_1d = el.giwaxs_sim(orientation=None)

        assert np.all(intensity_1d >= 0), "Negative intensity values found"

    def test_q_values_positive(self, test_crystal, exp_parameters):
        """Test that q values are positive."""
        el = GIWAXS(test_crystal, exp_parameters)
        q_1d, intensity_1d = el.giwaxs_sim(orientation=None)

        assert np.all(q_1d > 0), "Non-positive q values found"
