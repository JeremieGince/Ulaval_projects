from classifieur import *
import numpy as np


class Nbc(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.number_of_classes: int = 0
        self.probability_of_each_class: list = list()
        self.probability_of_each_feature: list = list()

    def train(self, train, train_labels):
        assert len(train) == len(train_labels)
        uniques, counts = np.unique(train_labels, return_counts=True)

        self.number_of_classes = len(uniques)
        for ids in uniques:
            self.probability_of_each_class.insert(ids, counts[ids]/len(train_labels))

        for i in range(len(uniques)):
            self.probability_of_each_feature.append([])

        label_pairing = []
        for ids in uniques:
            label_pairing.insert(ids, [i for i in range(len(train_labels)) if train_labels[i] == ids])

        for ids in uniques:
            data = train[label_pairing[ids], :]
            for i in range(data.shape[1]):
                column = data[:, i]
                unique_features, count = np.unique(column, return_counts=True)
                for j in range(len(unique_features)):
                    self.probability_of_each_feature[ids].insert(unique_features[j], [count[k]/len(column) for k in range(len(count)) ])

        print(self.probability_of_each_feature)

    def predict(self, exemple, label):
        print(self.probability_of_each_feature)
        probs = [[], []]
        print(exemple)
        for feature in exemple:
            for i in range(len(exemple)):
                for j in range(2):
                    probs[j].append(self.probability_of_each_feature[j][i][feature])
        print(probs)



    def test(self, test, test_labels):
        for i in range (len(test)):
            self.predict(test[i], test_labels[i])
