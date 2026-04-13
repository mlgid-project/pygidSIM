import math
from typing import List, Optional, Tuple
import numpy as np
from pygidsim.export_database import calculateFF


class Database:
    """
    A class to represent the form-factors matrix for all possible atoms.

    Attributes
    ----------
    full_atom_list : List[str]
        List of all possible atoms (len = 213).
    full_ff_matrix : np.ndarray
        Form-factors (ff) matrix for en=en in q_range = (0, q_max, 0.001).
        Shape (213, q_max*1000), where 213 is the number of elements and
        q_max*1000 is the number of points (ff) calculated in the range (0, q_max, 0.001).
    """

    def __init__(self,
                 en: float,  # energy in eV
                 q_max: float,  # max(q_abs) in Å^{-1}
                 ):
        self.full_ff_matrix, self.full_atom_list = calculateFF(en=en, q_max=q_max)


class ExpParameters:
    """
    A class to represent the experiment parameters.

    Attributes
    ----------
    q_xy_range : Tuple[float, float], optional
        Range for the q in xy direction, Å^{-1}.
    q_z_range : Tuple[float, float], optional
        Range for the q in z direction, Å^{-1}.
    q_xy_max : float, optional
        Upper limit for q in xy direction, Å^{-1}. Used if 'q_xy_range' is not provided.
    q_z_max : float, optional
        Upper limit for q in z direction, Å^{-1}. Used if 'q_z_range' is not provided.
    ai : float
        Incidence angle, degrees.
    wavelength : float
        Beam wavelength, Å.
    database : Optional[Database]
        Database with form-factors. Creates only if 'create_FF' is True; otherwise None.
    """

    def __init__(self,
                 q_xy_range: Optional[Tuple[float, float]] = None,
                 q_z_range: Optional[Tuple[float, float]] = None,
                 q_xy_max: Optional[float] = None,
                 q_z_max: Optional[float] = None,
                 ai: float = 0.3,  # Incidence angle, deg
                 en: float = 18000,  # Energy, eV
                 create_FF: bool = True,  # If True create database with form-factors.
                 ):
        if q_xy_range is None:
            if q_xy_max is not None:
                q_xy_range = (0.0, q_xy_max)
            else:
                raise ValueError('q_xy_range or q_xy_max must be provided.')
        if q_z_range is None:
            if q_z_max is not None:
                q_z_range = (0.0, q_z_max)
            else:
                raise ValueError('q_z_range or q_z_max must be provided.')

        assert len(q_xy_range) == len(q_z_range) == 2, "q_xy_range and q_z_range should be tuples of length 2."
        assert q_xy_range[1] > q_xy_range[0], "q_xy_range should be in the form (min, max) with max > min."
        assert q_z_range[1] > q_z_range[0], "q_z_range should be in the form (min, max) with max > min."

        self.q_xy_range = q_xy_range
        self.q_z_range = q_z_range
        q_xy_max = max(abs(self.q_xy_range[0]), abs(self.q_xy_range[1]))
        q_z_max = max(abs(self.q_z_range[0]), abs(self.q_z_range[1]))
        self.q_max = math.sqrt(q_xy_max ** 2 + q_z_max ** 2)
        self.ai = ai
        self.wavelength = 12398 / en

        self.database = Database(en=int(en), q_max=self.q_max) if create_FF else None
