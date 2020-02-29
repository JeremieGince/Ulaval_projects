from classifieur import *
import numpy as np
from util import GaussianDistribution


def ReturnDictionnaryAsProbabilities(d, insertIn, index=-1):
    string = ""
    for k, v in d.items():
            string += insertIn % (k, v)
    return string


def DisplayTrainResultGaussian(l):
    count = 0
    for item in l:
        print("Probabilities knowing " + str(count))
        subcount = 0
        for subitem in item:
            print(f"P({subcount})",
                  "= {}".format(f"{subitem:.3f}"
                                if isinstance(subitem, (float, int))
                                else str(subitem)))
            subcount += 1
        count += 1


def displayTrainingResults(toDisplay):
    for internal in toDisplay:
        if type(internal) is dict:
            print("Probabilities for feature number " + str(toDisplay.index(internal)))
            # for k, v in internal.items():
            #     print(f"P({k}) = {v:.3f}")
            print(ReturnDictionnaryAsProbabilities(internal, "P(%s) = %s\n"))
        if type(internal) is list:
            print("Probabilities knowing " + str(toDisplay.index(internal)))
        if type(internal) is not str and type(internal) is not GaussianDistribution:
            displayTrainingResults(internal)






class Nbc(Classifier):
    """
    Naive Bayes classifier implementation

    reference: https://fr.wikipedia.org/wiki/Classification_naïve_bayésienne
    """
    def __init__(self, **kwargs):
        """
        Constructor of Nbc.

        Attributes
        ----------
        :attr probability_of_each_class: Maps each class with its frequency of apparition in the training data {class -> frequency} (dict)
        :attr probability_of_each_feature: For each class, it maps the frequency of apparition of the feature in the given class
        (class -> feature -> frequency)(list)
        """
        super().__init__(**kwargs)
        self.number_of_classes: int = 0
        self.probability_of_each_class: dict = dict()
        self.probability_of_each_feature: list = list()

    def train(self, train_set, train_labels):
        """
        This method will set the parameters of the Nbc
        :param train_set: The vectorized training
        :param train_labels: the labels of the training set. The ith label is the label of the ith vector.
        :return: None
        """
        assert len(train_set) == len(train_labels)
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
            data = train_set[label_pairing[ids], :]
            for i in range(data.shape[1]):
                column = data[:, i]
                unique_features, count = np.unique(column, return_counts=True)
                self.probability_of_each_feature[ids].append({})
                for j in range(len(unique_features)):
                    self.probability_of_each_feature[ids][i][str(unique_features[j])] = count[j]/len(column)
        print("Probability of each class")
        print(ReturnDictionnaryAsProbabilities(self.probability_of_each_class, "P(%s) = %s\n"))
        print("Probability of each feature")
        displayTrainingResults(self.probability_of_each_feature)

    def predict(self, example, label):
        """
        This method will classify the example, and then return true if the predicted label == label,
        else it returns false
        :param example: The vectorized sample
        :param label: the actual label of the vector
        :return: a tuple. (classified label, result)
        """
        probs = []
        for i in range(self.number_of_classes):
            probs.append([])
        i = 0
        for feature in example:
            for j in range(self.number_of_classes):
                try:
                    probs[j].append(self.probability_of_each_feature[j][i][str(feature)])
                except:
                    probs[j].append(0)
            i +=1

        probabilite_final = []
        for i in range(self.number_of_classes):
            probabilite_final.append(self.probability_of_each_class[str(i)]*np.prod(probs[i]))
        res = probabilite_final.index(max(probabilite_final))
        return res, res == label

    def test(self, test_set, test_labels):
        return Classifier.test(self, test_set, test_labels)

class NbcGaussian(Nbc):
    """
    Naive Bayes classifier implementation with gaussian probability distribution of features.

    reference: https://fr.wikipedia.org/wiki/Classification_naïve_bayésienne
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, train_set, train_labels):
        """
        This method will set the parameters of the Nbc
        :param train: The vectorized training sample. The ith element is the ith vector
        :param train_labels: the labels of the training set. The ith label is the label of the ith vector.
        :return: None
        """
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
            data = train_set[label_pairing[ids], :]
            for i in range(data.shape[1]):
                column = data[:, i]
                var = np.var(column)
                average = np.average(column)
                self.probability_of_each_feature[ids].append(GaussianDistribution(average, var))
        DisplayTrainResultGaussian(self.probability_of_each_feature)

    def predict(self, example, label):
        probs = []
        for i in range(self.number_of_classes):
            probs.append([])
        i = 0
        for feature in example:
            for j in range(self.number_of_classes):
                probs[j].append(self.probability_of_each_feature[j][i].evaluate(feature))
            i += 1

        probabilite_final = []
        for i in range(self.number_of_classes):
            probabilite_final.append(self.probability_of_each_class[str(i)]*np.prod(probs[i]))
        prediction = probabilite_final.index(max(probabilite_final))
        return prediction, prediction == label

    def test(self, test_set, test_labels):
        return Classifier.test(self, test_set, test_labels)

if __name__ == '__main__':
    import load_datasets
    import time

    train_ratio: float = 0.4

    print(f"Train ratio: {train_ratio}")
    print("\n")

    print('-' * 175)
    print(f"Iris dataset classification: \n")
    startTime = time.time()

    iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(train_ratio)
    iris_knn = NbcGaussian()

    iris_knn.train(iris_train, iris_train_labels)
    iris_knn.test(iris_test, iris_test_labels)

    print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

    print('-'*175)
    print(f"Congressional dataset classification: \n")
    startTime = time.time()

    cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(train_ratio)
    cong_knn = Nbc()
    cong_knn.train(cong_train, cong_train_labels)
    cong_knn.test(cong_test, cong_test_labels)

    print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

    print('-' * 175)
    for i in range(3):
        print(f"Monks({i+1}) dataset classification: \n")
        startTime = time.time()

        monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i+1)
        monks_knn = Nbc()
        monks_knn.train(monks_train, monks_train_labels)
        monks_knn.test(monks_test, monks_test_labels)

        print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

        print('-' * 175)
