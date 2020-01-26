from classifieur import *
import numpy as np


class Nbc(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.number_of_classes: int = 0
        self.probability_of_each_class: dict = dict()
        self.probability_of_each_feature: list = list()

    def train(self, train, train_labels):
        assert len(train) == len(train_labels)
        uniques, counts = np.unique(train_labels, return_counts=True)

        self.number_of_classes = len(uniques)
        for ids in uniques:
            self.probability_of_each_class[str(ids)] = counts[ids]/len(train_labels)

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
                self.probability_of_each_feature[ids].append({})
                for j in range(len(unique_features)):
                    self.probability_of_each_feature[ids][i][str(unique_features[j])] = count[j]/len(column)

    def predict(self, exemple, label):
        probs = [[], []]

        i = 0
        for feature in exemple:
            for j in range(2):
                try:
                    probs[j].append(self.probability_of_each_feature[j][i][str(feature)])
                except:
                    probs[j].append(0)
            i +=1

        prob1 = self.probability_of_each_class['0']*np.prod(probs[0])
        prob2 = self.probability_of_each_class['1']*np.prod(probs[1])

        isZero = prob1 > prob2

        if isZero and label == 0:
            return True
        elif not isZero and label == 1:
            return True
        else:
            return False


    def test(self, test, test_labels):
        count = 0
        for i in range (len(test)):
            if self.predict(test[i], test_labels[i]):
                count += 1
        print(count/len(test_labels))
