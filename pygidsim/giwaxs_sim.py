import numpy as np
from typing import Union, Tuple, List, Optional

from xrayutilities.materials.cif import CIFFile
from xrayutilities.materials.spacegrouplattice import SGLattice

from pygidsim.experiment import ExpParameters
from pygidsim.q_sim import QPos
from pygidsim.int_sim import Intensity

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class Crystal:
    """
    A class to represent the crystal structure
    ...
    Attributes
    ----------
        lat_par : np.ndarray, dtype=np.float32
            lattice parameters np.array([a, b, c, α, β, γ])
        spgr : str or int
            space group number (1-230)
        cr_group : int
            crystal group number (1-7)
        atoms : np.ndarray
            list of elements in the structure
        atom_positions : np.ndarray
            relative atom coordinates (related with the atoms attribute)
        occ : np.ndarray, dtype=np.float32
            atom occupancies (related with the atoms attribute)
        scale : List[float, float, float]
            scale values for the lattice parameters
    """

    def __init__(self,
                 lat_par: np.ndarray,
                 spgr: Union[str, int] = 1,
                 atoms: Optional[np.ndarray] = None,
                 atom_positions: Optional[np.ndarray] = None,
                 occ: Optional[np.ndarray] = None,
                 scale: Optional[Tuple[float, float, float]] = None
                 ):
        assert lat_par.shape == (6,), 'lat_par should be a numpy array with 6 elements'
        if (atom_positions is not None) and (occ is None):
            occ = np.ones_like(atoms, np.float32)
        if scale is not None:
            assert len(scale) == 3, 'scale should have 3 elements — one for each length'
            lat_par[:3] = lat_par[:3] * scale
            if len(set(scale)) != 1:
                spgr = 1
        self.lat_par = lat_par
        self.spgr = spgr
        self.cr_group = self._cr_group()
        self._param_SGL = self._parameters_for_SGLattice()
        self.atoms = atoms
        self.atom_positions = atom_positions
        self.occ = occ
        self.scale = scale

    def _cr_group(self) -> Union[str, int]:
        """Return the crystal group from the space group"""
        cr_list = [(1, 2), (3, 15), (16, 74), (75, 142), (143, 167), (168, 194), (195, 230)]
        spgr_list = str(self.spgr).split(':')
        spgr_num = int(spgr_list[0])
        for idx, (start, end) in enumerate(cr_list):
            if spgr_num <= end:
                if (end == 167) and len(spgr_list) > 1 and spgr_list[1] == 'R':  # trigonal RHOMB
                    return '5:R'
                return idx + 1
        raise AttributeError('space group should be in range 1-231')

    def _parameters_for_SGLattice(self) -> np.ndarray:
        """Return the unique lattice parameters that are needed to receive miller indices"""
        parameters_for_hkl = {1: [0, 1, 2, 3, 4, 5],  # return [a, b, c, alpha, beta, gamma] - triclinic
                              2: [0, 1, 2, 4],  # return [a, b, c, beta] - monoclinic
                              3: [0, 1, 2],  # return [a, b, c] - orthorhombic
                              4: [0, 2],  # return [a, c] - tetragonal
                              5: [0, 2],  # return [a, c] - trigonal - HEX
                              '5:R': [0, 3],  # return [a, alpha] - trigonal - RHOMB
                              6: [0, 2],  # return [a, c] - hexagonal
                              7: [0]}  # return [a] - cubic
        return self.lat_par[parameters_for_hkl[self.cr_group]]


class GIWAXS:
    """
    A class to calculate the GIWAXS pattern from the crystal structure and experimental parameters
    ...
    Attributes
    ----------
        crystal : Crystal
            crystal structure representation
        exp : ExpParameters
            experiment parameters representation

    Methods
    -------
        giwaxs_sim(orientation):
            calculates the GIWAXS pattern
        get_mi(q_max):
            return allowed miller indices
    """

    def __init__(self,
                 crystal: Crystal,
                 exp_par: ExpParameters):
        self.crystal = crystal
        self.exp = exp_par

        self._mi = self._get_mi(self.exp.q_max)
        self._q_sim = QPos(self.crystal.lat_par, self._mi)
        self.q_3d = self._q_sim.q_3d
        self.q_abs = np.linalg.norm(self.q_3d, axis=-1)
        self._correct_init()

    def _correct_init(self):
        """Correct the initial attrivutes (by some reasons the initial q_abs could be greater than q_max)"""
        if self.q_abs.max() > self.exp.q_max:
            self._mi = self._mi[self.q_abs <= self.exp.q_max]
            self._q_sim = QPos(self.crystal.lat_par, self._mi)
            self.q_3d = self._q_sim.q_3d
            self.q_abs = np.linalg.norm(self.q_3d, axis=-1)

    def _get_mi(self, q_max) -> np.ndarray:
        """Return allowed miller indices"""
        mi = SGLattice(self.crystal.spgr, *self.crystal._param_SGL).get_allowed_hkl(qmax=q_max)
        return np.array(list(mi), dtype=np.float32)

    @property
    def mi(self) -> np.ndarray:
        """Return Miller indices"""
        return self._mi

    @property
    def rec(self) -> np.ndarray:
        """Return reciprocal vectors"""
        return self._q_sim.rec

    def giwaxs_sim(self,
                   orientation: Union[str, np.ndarray, None] = np.array([0., 0., 1.]),
                   max_mi: Union[None, int] = None,
                   return_mi=False,
                   move_fromMW=False,
                   ):
        """
        Calculates peak positions and their intensities in the GIWAXS pattern
        ...
        Parameters
        ----------
            orientation : Union[str, np.ndarray], optional
                orientation of the crystal growth (e.g. 'random' or np.array([0., 1., 0.]),
                default = np.array([0., 0., 1.])
                if None - no orientation (powder diffraction)
            max_mi: Union[None, int], optional
                restrict the maximum Miller Index
            return_mi : bool, optional
                return miller indices if True, default = False
            move_fromMW : bool, optional
                True if move peaks from missing wedge to visible area, default = False

        Return
        -------
            q_2d : np.ndarray
                peak positions - shape (2, peaks number)
            int_sum : np.ndarray
                peak intensities - shape (peaks number,)
        """
        if max_mi is not None:
            mi_idx = np.max(np.abs(self._mi), axis=1) <= abs(max_mi)
        else:
            mi_idx = np.ones(self._mi.shape[0], dtype=bool)
        if (self.crystal.atoms is None) or (self.crystal.atom_positions is None):
            intensity = None
        else:
            intensity = Intensity(
                self.crystal.atoms,
                self.crystal.atom_positions,
                self.crystal.occ,
                self.q_3d[mi_idx],
                self._mi[mi_idx],
                self.exp.wavelength,
                self.exp.ai,
                self.exp.database,
            ).get_intensities()
            mi_idx[mi_idx] &= intensity > 1e-6
            intensity = intensity[intensity > 1e-6]

        if orientation is None:
            if not return_mi:
                q_1d, int_sum, mi_sum = self.giwaxs_1d(
                    self.q_abs[mi_idx],
                    intensity,
                    None,
                    self.exp.wavelength,
                )
                return q_1d, int_sum
            else:
                q_1d, int_sum, mi_sum = self.giwaxs_1d(
                    self.q_abs[mi_idx],
                    intensity,
                    self._mi[mi_idx],
                    self.exp.wavelength,
                )
                return q_1d, int_sum, mi_sum
        else:
            q_3d_rot = self._q_sim.rotate_vect(orientation)[mi_idx]
            if not return_mi:
                q_2d, int_sum, _ = self.giwaxs_2d(
                    q_3d_rot,
                    intensity,
                    None,
                    (self.exp.q_xy_max, self.exp.q_z_max),
                    self.exp.wavelength,
                    move_fromMW,
                )
                return q_2d, int_sum
            else:
                q_2d, int_sum, mi_sum = self.giwaxs_2d(
                    q_3d_rot,
                    intensity,
                    self._mi[mi_idx],
                    (self.exp.q_xy_max, self.exp.q_z_max),
                    self.exp.wavelength,
                    move_fromMW,
                )
                return q_2d, int_sum, mi_sum

    @staticmethod
    def giwaxs_1d(q_1d: np.ndarray,  # shape - (peaks_num,)
                  intensity: Union[np.ndarray, None],  # shape - (peaks_num,)
                  mi: Union[np.ndarray, None],  # shape - (peaks_num, 3)
                  wavelength: float,  # beam wavelength, Å
                  ):
        """Calculate powder diffraction pattern for GIWAXS"""
        clusters = GIWAXS.cluster_mask(q_1d, r=1e-2)

        counts_per_cluster = np.bincount(clusters)
        q_1d_fin = np.bincount(clusters, weights=q_1d) / counts_per_cluster

        if intensity is None:
            int_fin = np.ones_like(q_1d_fin)
        else:
            intensity_corrected = GIWAXS._lorentz_correction_1d(q_1d, intensity, wavelength)
            int_fin = np.bincount(clusters, weights=intensity_corrected)

        # remove low intensity peaks
        int_mask = int_fin > int_fin.max() * 1e-7
        int_fin = int_fin[int_mask]
        q_1d_fin = q_1d_fin[int_mask]
        clusters[~int_mask[clusters]] = -1

        if mi is not None:
            # concatenate "the same mi" together
            unique_clusters = np.unique(clusters)
            mi_new = []
            for uc in unique_clusters:
                if uc == -1:
                    continue
                mi_new.append(mi[clusters == uc])
            return q_1d_fin, int_fin, mi_new

        return q_1d_fin, int_fin, mi

    @staticmethod
    def giwaxs_2d(q_3d: np.ndarray,  # shape - (peaks_num, 3)
                  intensity: Union[np.ndarray, None],  # shape - (peaks_num,)
                  mi: Union[np.ndarray, None],  # shape - (peaks_num, 3)
                  q_range: Tuple[float, float],
                  wavelength: float,  # beam wavelength, Å
                  move_fromMW=False,  # True if move peaks from missing wedge to visible area
                  ):
        """Convert 3d pattern to 2d space with summation the intensities"""
        q_xy = np.sqrt(q_3d[..., 0] ** 2 + q_3d[..., 1] ** 2)  # shape (peaks_num,)
        q_z = q_3d[..., 2]
        q_2d = np.stack((q_xy, q_z))  # shape (2, peaks_num)

        # take only peaks inside q_range
        q_2d[np.abs(q_2d) < 1e-4] = 0
        q_mask = ((q_2d[1] >= 0) &
                  (q_2d[1] <= q_range[1]) &
                  (q_2d[0] <= q_range[0]))
        q_2d = q_2d[:, q_mask]

        if intensity is not None:
            intensity = intensity[q_mask]
            intensity_corrected = GIWAXS._lorentz_correction_2d(q_2d, intensity, wavelength)
        if move_fromMW:
            q_2d = GIWAXS._move_fromMW(q_2d, wavelength)

        clusters = GIWAXS.cluster_mask(q_2d, r=2e-2)

        counts_per_cluster = np.bincount(clusters)
        sum_x = np.bincount(clusters, weights=q_2d[0])
        sum_y = np.bincount(clusters, weights=q_2d[1])
        q_2d_fin = np.vstack((sum_x, sum_y)) / counts_per_cluster  # shape (2, peaks_num)

        if intensity is None:
            int_fin = np.ones(q_2d_fin.shape[1])
        else:
            int_fin = np.bincount(clusters, weights=intensity_corrected)

        # remove low intensity peaks
        int_mask = int_fin > int_fin.max() * 1e-6
        int_fin = int_fin[int_mask]
        q_2d_fin = q_2d_fin[:, int_mask]
        clusters[~int_mask[clusters]] = -1

        if mi is not None:
            # concatenate "the same mi" together
            mi = mi[q_mask]
            unique_clusters = np.unique(clusters)
            mi_new = []
            for uc in unique_clusters:
                if uc == -1:
                    continue
                mi_new.append(mi[clusters == uc])
            return q_2d_fin, int_fin, mi_new

        return q_2d_fin, int_fin, mi

    @staticmethod
    def _lorentz_correction_1d(q_1d: np.ndarray,  # (peaks_num,)
                               intensities: np.ndarray,  # (peaks_num,)
                               wavelength: float = 12398 / 18000):  # wavelength, Angstrom
        k = 2 * np.pi / wavelength
        L = 1 / (q_1d ** 2 * np.sqrt(1 - (q_1d / (2 * k)) ** 2))

        return L * intensities

    @staticmethod
    def _lorentz_correction_2d(q_2d: np.ndarray,  # (2, peaks_num)
                               intensities: np.ndarray,  # (peaks_num,)
                               wavelength: float = 12398 / 18000):  # wavelength, Angstrom
        k = 2 * np.pi / wavelength

        condition_inMW = (k - abs(q_2d[0])) ** 2 > (k ** 2 - q_2d[1] ** 2)  # condition if peaks are in Missing Wedge
        L = np.empty_like(intensities)
        L[condition_inMW] = 2 * k / np.linalg.norm(q_2d, axis=0)[condition_inMW] ** 2
        L[~condition_inMW] = 1 / q_2d[0][~condition_inMW]

        return L * intensities

    @staticmethod
    def _move_fromMW(q_2d: np.ndarray,  # (2, peaks_num)
                     wavelength: float = 12398 / 18000,  # wavelength, Angstrom
                     ):
        """Move peaks from Missimg Wedge to the visible area"""
        k = 2 * np.pi / wavelength
        condition_inMW = (k - abs(q_2d[0])) ** 2 > (k ** 2 - q_2d[1] ** 2)  # condition if peaks are in Missing Wedge
        q_abs_mod = np.linalg.norm(q_2d[:, condition_inMW], axis=0)
        q_2d[0, condition_inMW] = q_abs_mod ** 2 / (2 * k)
        q_2d[1, condition_inMW] = q_abs_mod * np.sqrt(4 * (k ** 2) - q_abs_mod ** 2) / (2 * k)

        return q_2d

    @staticmethod
    def cluster_mask(q_sim: np.ndarray,  # shape (2, N) or (N,)
                     r: float):
        ndim = q_sim.ndim
        if ndim == 1:
            coords = q_sim.reshape(-1, 1)  # shape (N, 1)
        elif ndim == 2:
            coords = q_sim.T  # shape (N, 2)
        else:
            raise ValueError(f"Wrong q_sim shape: {q_sim.shape}")

        N = coords.shape[0]
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=r, p=np.inf)

        if pairs:
            i, j = zip(*pairs)
            i = np.array(i, dtype=int)
            j = np.array(j, dtype=int)

            row = np.concatenate([i, j])
            col = np.concatenate([j, i])
            data = np.ones(len(row), dtype=bool)

            adj = coo_matrix((data, (row, col)), shape=(N, N))
            n_clusters, labels = connected_components(adj, directed=False, return_labels=True)
        else:
            labels = np.arange(N)
        return labels


def _create_crystal_from_base(lat_par, spgr, base, scale=None):
    atoms, atom_positions, occ = zip(
        *[
            (atom[0].basename, atom[1], atom[2]) for atom in base
        ],
    )
    if np.isnan(atom_positions).any():
        raise ValueError("Some atoms have NaN positions")
    return Crystal(
        lat_par=np.array(lat_par, dtype=np.float32),
        spgr=spgr,
        atoms=np.array(atoms),
        atom_positions=np.array(atom_positions, dtype=np.float32),
        occ=np.array(occ, dtype=np.float32),
        scale=scale,
    )


class GIWAXSFromCif:
    """
    A class to calculate GIWAXS pattern from the CIF
    ...
    Attributes
    ----------
        crystal : Crystal
            crystal structure representation
        giwaxs : GIWAXS
    """

    def __init__(self,
                 path: str,
                 exp_par: ExpParameters,
                 scale: Optional[Tuple[float, float, float]] = None,
                 ):
        """
        Parameters
        ----------
            path : str
                path to the CIF file
            exp_par : ExpParameters
                experiment parameters representation
            scale : List[float, float, float]
                scale values for the lattice parameters
        """
        el = CIFFile(path)
        name = el.default_dataset
        if name is None:
            name = list(el.data.keys())[0]

        lat_par = (np.append(el.data[name].lattice_const, el.data[name].lattice_angles)).astype(np.float32)
        if hasattr(el.data[name], 'sgrp'):
            spgr = el.data[name].sgrp
        else:
            spgr = 1

        base = el.SGLattice().base()

        self.crystal = _create_crystal_from_base(lat_par, spgr, base, scale)
        self.giwaxs = GIWAXS(
            crystal=self.crystal,
            exp_par=exp_par,
        )


class GIWAXSFromSGLattice:
    """
    A class to calculate GIWAXS pattern from the SGLattice class in xrayutilities:
    https://xrayutilities.sourceforge.io/_modules/xrayutilities/materials/spacegrouplattice.html#SGLattice
    ...
    Attributes
    ----------
        crystal : Crystal
            crystal structure representation
        giwaxs : GIWAXS
    """

    def __init__(self,
                 lattice: SGLattice,
                 exp_par: ExpParameters,
                 scale: Optional[Tuple[float, float, float]] = None,
                 ):
        """
        Parameters
        ----------
            lattice : xrayutilities.materials.spacegrouplattice.SGLattice
            exp_par : ExpParameters
                experiment parameters representation
            scale : List[float, float, float]
                scale values for the lattice parameters
        """
        lat_par = np.array(list(lattice._parameters.values()), dtype=np.float32)
        spgr = lattice.space_group
        base = lattice.base()

        self.crystal = _create_crystal_from_base(lat_par, spgr, base, scale)
        self.giwaxs = GIWAXS(
            crystal=self.crystal,
            exp_par=exp_par,
        )


class GIWAXSFromCell:
    """
    A class to calculate GIWAXS pattern from the Cell class in celltools package:
     https://github.com/HammerSeb/celltools/
    ...
    Attributes
    ----------
        crystal : Crystal
            crystal structure representation
        giwaxs : GIWAXS
    """

    def __init__(self,
                 cell,
                 exp_par: ExpParameters,
                 scale: Optional[Tuple[float, float, float]] = None,
                 ):
        """
        Parameters
        ----------
            cell : celltools.cell.Cell
            exp_par : ExpParameters
                experiment parameters representation
            scale : List[float, float, float]
                scale values for the lattice parameters
        """
        from itertools import chain
        cell_atoms = list(chain.from_iterable(mol.atoms for mol in cell.molecules)) + cell.atoms
        atoms, atom_positions = zip(*[(atom.element, atom._v.vector) for atom in cell_atoms])
        atoms = np.array(atoms)
        atom_positions = np.array(atom_positions, dtype=np.float32)

        lengths = np.linalg.norm(cell.lattice._basis, axis=1)
        cos_angles = np.array(
            [
                np.dot(cell.lattice._basis[1], cell.lattice._basis[2]) / (lengths[1] * lengths[2]),  # α
                np.dot(cell.lattice._basis[0], cell.lattice._basis[2]) / (lengths[0] * lengths[2]),  # β
                np.dot(cell.lattice._basis[0], cell.lattice._basis[1]) / (lengths[0] * lengths[1])  # γ
            ],
        )
        angles = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
        lat_par = np.concatenate((lengths, angles))
        spgr = 1

        self.crystal = Crystal(
            lat_par=lat_par,
            spgr=spgr,
            atoms=atoms,
            atom_positions=atom_positions,
            scale=scale,
        )
        self.giwaxs = GIWAXS(
            crystal=self.crystal,
            exp_par=exp_par,
        )
