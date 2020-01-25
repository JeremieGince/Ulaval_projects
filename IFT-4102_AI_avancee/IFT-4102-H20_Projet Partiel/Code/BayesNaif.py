from classifieur import *
import numpy as np


class Nbc(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.number_of_classes: int = 0


    def train(self, train, train_labels):
        assert len(train) == len(train_labels)
        uniques: np.ndarray = np.unique(train_labels)
        self.number_of_classes = len(uniques)
        print(self.number_of_classes)

    def predict(self, exemple, label):
        # https://fr.wikipedia.org/wiki/Recherche_des_plus_proches_voisins
        pass

    def test(self, test, test_labels):
        pass
