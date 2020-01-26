import numpy as np


def euclidean_distance(vector0: np.ndarray, vector1: np.ndarray) -> float:
    return np.sqrt(np.sum((vector1-vector0)**2))


class MapHashVecLabel:
    def __init__(self, dictionary: dict = None):
        if isinstance(dictionary, MapHashVecLabel):
            dictionary = dictionary._container
        self._container: dict = dict() if dictionary is None else dictionary

    def __copy__(self):
        return MapHashVecLabel(self._container)

    def __deepcopy__(self, memodict={}):
        return MapHashVecLabel(dict(self._container))

    def deepcopy(self):
        return self.__deepcopy__()

    def __setitem__(self, key, value):
        self._container[str(key)] = value

    def __getitem__(self, item):
        return self._container[str(item)]

    def __bool__(self):
        return bool(self._container)

    def popitem(self):
        key, value = self._container.popitem()
        return self._strvecTonpvec(key), value

    def asLists(self):
        keys = list(self._container.keys())
        return [self._strvecTonpvec(key) for key in keys], [self[key] for key in keys]

    def _strvecTonpvec(self, strvec):
        strvec = strvec.replace('[', '').replace(']', '')
        return np.fromstring(strvec, sep=' ')
