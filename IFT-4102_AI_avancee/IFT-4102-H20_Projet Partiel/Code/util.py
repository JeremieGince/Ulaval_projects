import numpy as np
from math import pi, exp

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

class GaussianDistribution:
    def __init__(self, average, variance):
        self.variance = variance
        self.average = average

    def evaluate(self, value):
        alpha = 1.0/(2*pi*self.variance)**(1/2.0)
        second_term = exp(((-1.0)/(2*self.variance))*((value - self.average))**2)
        return alpha*second_term

