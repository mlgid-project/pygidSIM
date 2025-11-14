"""
Pytest configuration and shared fixtures for pygidsim tests.
"""
import pytest
import numpy as np
from pathlib import Path
from pygidsim.experiment import ExpParameters
from pygidsim.giwaxs_sim import Crystal


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent


@pytest.fixture
def cif_file(test_data_dir):
    """Return path to the test CIF file."""
    cif_path = test_data_dir / 'test_data' / 'MAPbI3.cif'
    if not cif_path.exists():
        pytest.skip(f"CIF file not found: {cif_path}")
    return cif_path


@pytest.fixture
def exp_parameters():
    """Return a standard ExpParameters instance for testing."""
    return ExpParameters(
        q_xy_max=5.0,
        q_z_max=5.0,
        ai=0.3,
        en=18_000,
    )


@pytest.fixture
def test_crystal():
    """Return a Crystal instance for testing."""
    spgr = 221
    lat_par = np.array([6.3026, 6.3026, 6.3026, 90., 90., 90.], dtype=np.float32)
    atoms = np.array(['Pb', 'I', 'I', 'I', 'N'])
    atoms_positions = np.array(
        [
            [0., 0., 0.],
            [0.5, 0., 0.],
            [0., 0.5, 0.],
            [0., 0., 0.5],
            [0.5, 0.5, 0.5]
        ], dtype=np.float32,
    )
    occupancy = np.array([1., 1., 1., 1., 1.], dtype=np.float32)

    return Crystal(lat_par, spgr, atoms, atoms_positions, occupancy)


@pytest.fixture
def test_crystal_no_occupancy():
    """Return a Crystal instance without occupancy for testing."""
    spgr = 221
    lat_par = np.array([6.3026, 6.3026, 6.3026, 90., 90., 90.], dtype=np.float32)
    atoms = np.array(['Pb', 'I', 'I', 'I', 'N'])
    atoms_positions = np.array(
        [
            [0., 0., 0.],
            [0.5, 0., 0.],
            [0., 0.5, 0.],
            [0., 0., 0.5],
            [0.5, 0.5, 0.5]
        ], dtype=np.float32,
    )

    return Crystal(lat_par, spgr, atoms, atoms_positions)


@pytest.fixture
def test_crystal_no_atoms():
    """Return a Crystal instance without atoms for testing."""
    return Crystal(
        spgr=221,
        lat_par=np.array([6.3026, 6.3026, 6.3026, 90., 90., 90.], dtype=np.float32),
    )


@pytest.fixture
def random_orientation():
    """Return a random orientation vector for testing."""
    return np.array([5., 7., 1.])


@pytest.fixture
def mi_restriction():
    """Return a maximum value for Miller indices."""
    return 3
