import numpy as np
from math import pi, exp


def euclidean_distance(vector0: np.ndarray, vector1: np.ndarray) -> float:
    return np.sqrt(np.sum((vector1-vector0)**2))


def weighted_euclidean_distance(vector0: np.ndarray, vector1: np.ndarray, **kwargs) -> float:
    weights: np.ndarray = kwargs.get("weights", np.ones(np.shape(vector0)))
    return np.sqrt(np.sum(weights*(vector1 - vector0) ** 2))


def manhattan_distance(vector0: np.ndarray, vector1: np.ndarray) -> float:
    return sum(np.abs(vector1-vector0))


def chebyshev_distance(vector0: np.ndarray, vector1: np.ndarray) -> float:
    return np.max(np.abs(vector1-vector0))


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

    def __str__(self):
        return f"Gaussian distribution with average {self.average:.3f} and variance {self.variance:.3f}"


def computeTprFpr(ConfusionMatrix):
    Tpr = np.array([vector[idx] / np.sum(vector)
                    for idx, vector in enumerate(ConfusionMatrix.transpose())])
    cmSum = np.sum(ConfusionMatrix)
    Fpr = np.array([(np.sum(vector) - vector[idx]) / (cmSum - np.sum(ConfusionMatrix.transpose()[idx]))
                    for idx, vector in enumerate(ConfusionMatrix)])
    return np.array(Tpr), np.array(Fpr)


def computeTprFprList(ConfusionMatrixList):
    Tpr = [[vector[idx] / np.sum(vector)
           for idx, vector in enumerate(cm.transpose())]
           for cm in ConfusionMatrixList]
    cmSumList = [np.sum(cm) for cm in ConfusionMatrixList]
    Fpr = [[(np.sum(vector) - vector[idx]) / (cmSumList[jdx] - np.sum(cm.transpose()[idx]))
           for idx, vector in enumerate(cm)]
           for jdx, cm in enumerate(ConfusionMatrixList)]

    flatten = lambda l: [item for sublist in l for item in sublist]
    return np.array(flatten(Tpr)), np.array(flatten(Fpr))


def plotROCcurves(Tpr: np.ndarray, Fpr: np.ndarray, hmCurve: int = 1, title: str = "ROC curve", **kwargs):
    import os
    import matplotlib.pyplot as plt

    if hmCurve == 1 and len(Tpr.shape) == 1:
        Tpr = Tpr[np.newaxis, :]
        Fpr = Fpr[np.newaxis, :]

    print(Tpr.shape, Fpr.shape, hmCurve)
    assert Tpr.shape[0] == Fpr.shape[0] == hmCurve
    labels = kwargs.get("labels", range(hmCurve))

    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], label="random")
    for i in range(hmCurve):
        plt.plot(Fpr[i, :], Tpr[i, :], 'o', label=labels[i], lw=(hmCurve-i), alpha=0.7)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid()
    plt.legend()
    plt.savefig(f"{os.getcwd()}/{title}_plot.png", dpi=300)
    plt.show()
