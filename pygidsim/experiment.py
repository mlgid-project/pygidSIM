import math
import numpy as np
from pygidsim.export_database import calculateFF


class Database:
    """
    A class to represent the form-factors matrix for all possible atoms
    ...
    Attributes
    ----------
    full_atom_list : list
        list of all possible atoms - len = 213
    full_ff_matrix : np.ndarray
        form-factors (ff) matrix for en=en in q_range = (0, q_max, 0.001)
        shape = (213, q_max*1000), 213 - number of elements, q_max*1000 - calculated ff in range(0, q_max, 0.001)
    """

    def __init__(self,
                 en: float,  # energy in eV
                 q_max: float,  # max(q_abs) in Å^{-1}
                 ):
        self.full_ff_matrix, self.full_atom_list = calculateFF(en=en, q_max=q_max)


class ExpParameters:
    """
    A class to represent the experiment parameters
    ...
    Attributes
    ----------
    q_xy_max : float
        maximum value for the q in xy direction, Å^{-1}
    q_z_max : float
        maximum value for the q in z direction, Å^{-1}
    ai : float
        incidence angle, deg
    wavelength : float
        beam wavelength, Å
    database: Database
        database with form-factors
    """

    def __init__(self,
                 q_xy_max: float = 2.7,  # Å^{-1}
                 q_z_max: float = 2.7,  # Å^{-1}
                 ai: float = 0.3,  # Incidence angle, deg
                 en: float = 18000,  # Energy, eV
                 ):
        self.q_xy_max = q_xy_max
        self.q_z_max = q_z_max
        self.q_max = math.sqrt(q_xy_max ** 2 + q_z_max ** 2)
        self.ai = ai
        self.wavelength = 12398 / en

        self.database = Database(en=int(en), q_max=self.q_max)
