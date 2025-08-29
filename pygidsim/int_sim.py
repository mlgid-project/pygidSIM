import numpy as np
import psutil


class Intensity:
    """
    A class to calculate the GIWAXS intensities
    ...
    Attributes
    ----------
    atoms : np.ndarray
        list of elements in the structure
    atom_positions : np.ndarray
        relative atom coordinates (related with the atoms attribute)
    occ : np.ndarray
            atom occupancies (related with the atoms attribute)
    q_3d : np.ndarray
        peak vectors in 3d reciprocal space
    q_1d : np.ndarray
        absolute peak positions in reciprocal space
    mi : np.ndarray
        miller indices
    wavelength : float
        beam wavelength, Å
    ai : float
        incidence angle, deg
    database : Database
        database with form-factors

    Methods
    -------
    get_intensities():
        return intensities for each peak position
    """

    def __init__(self,
                 atoms: np.ndarray,  # list of elements in the structure,
                 atom_positions: np.ndarray,  # relative atom coordinates
                 occ: np.ndarray,  # occupancies, dtype=np.float32
                 q_3d: np.ndarray,  # 3d vectors - (E, 3) - [[qx, qy, qz]]
                 mi: np.ndarray,  # 3d vectors - (E, 3) - [[h, k, l]]
                 wavelength: float,  # angstrom
                 ai: float,  # Incidence angle, deg.
                 database,
                 ):
        self.atoms = atoms
        self.atom_positions = atom_positions
        self.occ = occ
        assert len(atoms) == len(atom_positions) == len(occ), "number of atoms, positions and occupancies do not match"
        self.q_3d = q_3d
        self.q_1d = np.linalg.norm(q_3d, axis=-1)  # 1d vectors - len(q), shape = N, N-number of peaks in 1D
        self.mi = mi

        self.wavelength = wavelength
        self.ai = ai
        self.database = database

    def get_intensities(self):
        """Return intensities for each peak position"""
        intensity = self._get_intensities_from_mi()
        return intensity

    def _get_intensities_from_mi(self) -> np.ndarray:
        """Return intensities for each peak position"""
        sum_sf = self._get_sf(self.database.full_atom_list, self.database.full_ff_matrix)  # structure factor
        intensity = np.abs(sum_sf) ** 2
        return intensity

    def _adaptive_block_size(self, max_ram_gb: float = 8.0) -> int:
        """Return max block_size"""
        mem = psutil.virtual_memory()
        max_ram_gb = min(mem.available / 1024 ** 3 / 2, max_ram_gb)
        bytes_per_complex64 = 8
        n_arrays = 3  # ffs_block, phase_factor, weighted_ffs
        max_elements = (max_ram_gb * (1024 ** 3)) / (len(self.atom_positions) * bytes_per_complex64 * n_arrays)
        return int(max_elements)

    def _get_sf(self,
                atom_list: np.ndarray,  # list of all possible elements: len = 213
                full_ff_matrix: np.ndarray,
                # shape = (213, q_max*1000), 213 - number of elements, q_max*1000 - calculated ff in range(0, q_max, 0.001))
                ) -> np.ndarray:
        """Return structure factor as a sum of scattering amplitudes for all atoms"""

        unique_atoms, inverse_indices = np.unique(self.atoms, return_inverse=True)
        unknown = set(unique_atoms) - set(atom_list)
        if unknown:
            raise KeyError(f"Unknown elements in atoms list: {unknown}")
        atom2idx = {el: i for i, el in enumerate(atom_list)}
        smbl_num_unique = [atom2idx[el] for el in unique_atoms]

        q_list_num = (self.q_1d * 1000).astype(np.int32)
        ff_matrix_unique = full_ff_matrix[smbl_num_unique][:, q_list_num]

        num_q = self.mi.shape[0]
        sum_sf = np.zeros(num_q, dtype=np.complex64)

        block_size = self._adaptive_block_size(max_ram_gb=8.0)
        for start in range(0, num_q, block_size):
            end = min(start + block_size, num_q)
            mi_block = self.mi[start:end]  # (block_size, 3)

            # (N_atoms, block_size): phase factor
            phase = np.dot(self.atom_positions, mi_block.T)
            phase_factor = np.exp(2j * np.pi * phase)

            ffs_block = ff_matrix_unique[:, start:end]
            ffs = ffs_block[inverse_indices]  # (N_atoms, block_size)

            weighted_ffs = ffs * self.occ[:, np.newaxis]
            sf_block = np.sum(weighted_ffs * phase_factor, axis=0)

            sum_sf[start:end] = sf_block
        return sum_sf
