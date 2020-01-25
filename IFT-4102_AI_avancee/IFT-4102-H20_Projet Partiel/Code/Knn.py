from classifieur import *
import numpy as np


class Knn(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vector_to_label: dict = dict()

    def train(self, train, train_labels):
        assert len(train) == len(train_labels)
        self.vector_to_label = {train[i]: train_labels[i] for i in range(len(train))}

    def predict(self, exemple, label):
        # https://fr.wikipedia.org/wiki/Recherche_des_plus_proches_voisins
        pass

    def test(self, test, test_labels):
        pass


def euclidean_distance(vector0: np.ndarray, vector1: np.ndarray) -> float:
    return np.sqrt(np.sum((vector1-vector0)**2))
