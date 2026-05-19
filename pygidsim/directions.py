import itertools
import numpy as np


def get_unique_directions(max_index: int = 10) -> np.ndarray:
    """
    Generate unique orientations within the given range.

    Removes:
    - inversion symmetry:
        (h,k,l) == (-h,-k,-l)
    - collinear duplicates:
        (1,1,1) == (2,2,2)

    Parameters
    ----------
    max_index : int
        The maximum index.

    Returns
    -------
    np.ndarray
        The orientations of shape (num_orientations, 3).
    """
    hkl = get_mi(max_index)
    hkl = _reduce_directions(hkl)
    return hkl


def get_mi(max_index: int) -> np.ndarray:
    """
    Generate all the Laue indices within the given range excluding (0, 0, 0).
    No symmetry reduction is applied.

    Parameters
    ----------
    max_index : int
        The maximum index.

    Returns
    -------
    np.ndarray
        The Laue indices of shape (num_reflections, 3).
    """
    r = np.arange(-max_index, max_index + 1, dtype=np.int64)
    hkl = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)

    # remove (0,0,0)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    return hkl


def _reduce_directions(hkl: np.ndarray) -> np.ndarray:
    """
    Remove:
    - collinear multiples
    - inversion symmetry

    Examples
    --------
    (2,2,2) -> (1,1,1)

    (-1,-2,-3) -> (1,2,3)
    """
    hkl = hkl.copy()

    # gcd reduction fully vectorized
    g = np.gcd(
        np.gcd(np.abs(hkl[:, 0]), np.abs(hkl[:, 1])),
        np.abs(hkl[:, 2])
    )

    hkl = hkl // g[:, None]

    # canonical sign (vectorized)
    sign = np.where(
        hkl[:, 0] != 0,
        np.sign(hkl[:, 0]),
        np.where(
            hkl[:, 1] != 0,
            np.sign(hkl[:, 1]),
            np.sign(hkl[:, 2]),
        )
    )

    hkl[sign < 0] *= -1

    return np.unique(hkl, axis=0)
