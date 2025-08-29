import numpy as np
import xrayutilities as xu
from pygidsim.data.atoms import atoms_list


def calculateFF(en,  # energy in eV
                q_max,  # max(q_abs) in Å^{-1}
                ):
    """Calculate and save form-factors for each possible atom in atom_list"""
    q_range = np.arange(0, q_max, 0.001)
    result_shape = (len(atoms_list), len(q_range))
    full_ff_matrix = np.empty(result_shape, dtype='complex128')

    for idx, atom in enumerate(atoms_list):
        ff = xu.materials.atom.Atom(atom, 1).f(q_range, en=en)  # form factor
        full_ff_matrix[idx] = ff

    return full_ff_matrix, np.array(atoms_list)
