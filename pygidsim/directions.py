import itertools
import numpy as np


def get_unique_directions(max_index: int) -> np.ndarray:
    """ Generate possible crystallographic orientations """
    direct = np.array(_get_all_directions(max_index), dtype=np.float32)
    direct = direct[np.argsort(np.linalg.norm(direct, axis=1))]
    direct_norm = direct / np.linalg.norm(direct, axis=1)[..., np.newaxis]
    dir_unique = np.unique(direct_norm, axis=0)
    return dir_unique


def _get_all_directions(max_index: int):
    rng = list(range(-max_index, max_index + 1))[::-1]
    conv_hkl_list = [miller for miller in itertools.product(rng, rng, rng) if any(i != 0 for i in miller)]
    return conv_hkl_list
