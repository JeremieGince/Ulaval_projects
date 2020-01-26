from classifieur import *
import numpy as np
import util


class Knn(Classifier):

    defaultK: int = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_vector_to_label: util.MapHashVecLabel = util.MapHashVecLabel()
        self.test_vector_to_label: util.MapHashVecLabel = util.MapHashVecLabel()
        self._Kmin: int = 1
        self.K: int = Knn.defaultK
        self.Kmax: int = 25

    def setData(self, train, train_labels, test, test_labels):
        self.setTrainData(train, train_labels)
        self.setTestData(test, test_labels)

    def setTrainData(self, train, train_labels):
        assert len(train) == len(train_labels)
        self.train_vector_to_label = util.MapHashVecLabel({str(train[i]): train_labels[i] for i in range(len(train))})

    def setTestData(self, test, test_labels):
        assert len(test) == len(test_labels)
        self.test_vector_to_label = util.MapHashVecLabel({str(test[i]): test_labels[i] for i in range(len(test))})

    def train(self, train, train_labels, verbose: bool = True, findBestKWithCrossValidation: bool = False):
        self.setTrainData(train, train_labels)

        if findBestKWithCrossValidation:
            kToacc: dict = {k: self.crossValidation(train, train_labels, cv=5, k=k)
                            for k in range(self._Kmin, self.Kmax)}
            self.K = max(kToacc, key=kToacc.get)

        if verbose:
            print(f"\n Train results: \n"
                  f"Train set size: {len(train)} \n"
                  f"Chosen K: {self.K} \n")
        return self.test(train, train_labels, verbose, False)

    def predict(self, exemple, label):
        # https://fr.wikipedia.org/wiki/Recherche_des_plus_proches_voisins
        return self.naivePrediction(exemple, label)

    def naivePrediction(self, exemple, label, distanceFunc=util.euclidean_distance):
        train_data: util.MapHashVecLabel = self.train_vector_to_label.deepcopy()
        neighbors: list = list()

        while train_data:
            train_vector, cls = train_data.popitem()
            distance: float = distanceFunc(exemple, train_vector)
            neighbor: tuple = (train_vector, distance, cls)
            neighbors.append(neighbor)

        # nearest_neighbors: list = [vec for vec, dist, cls in sorted(neighbors, key=lambda nei: nei[1])[:self.K]]
        nearest_classes: list = [cls for vec, dist, cls in sorted(neighbors, key=lambda nei: nei[1])[:self.K]]
        nearest_classes_count: dict = {cls: nearest_classes.count(cls) for cls in set(nearest_classes)}
        prediction_cls: int = max(nearest_classes_count, key=nearest_classes_count.get)
        return prediction_cls, prediction_cls == label

    def test(self, test, test_labels, verbose: bool = True, testMessage: bool = True) -> tuple:
        self.setTestData(test, test_labels)
        confusionMatrix: np.ndarray = self.getConfusionMatrix(test, test_labels)
        accuracy: float = self.getAccuracy(test, test_labels)
        precision = self.getPrecision(test, test_labels)
        recall = self.getRecall(test, test_labels)

        if verbose:
            if testMessage:
                print(f"\n Test results: \n"
                      f"Test set size: {len(test)} \n")
            print(f"Confusion Matrix: \n {confusionMatrix}",
                  f"Accuracy: {accuracy:.2f} %",
                  f"Precision: {precision:.5f}",
                  f"Recall: {recall:.5f}",
                  sep='\n')

        return confusionMatrix, accuracy, precision, recall

    def getAccuracy(self, test=None, test_labels=None):
        if test is None or test_labels is None:
            test, test_labels = self.test_vector_to_label.asLists()
        return Classifier.getAccuracy(self, test, test_labels)

    def crossValidation(self, train, train_labels, cv: int = 5, k: int = defaultK):
        di: int = int(len(train)/cv)

        accuracy_list: list = list()
        for i in range(cv):
            train_i = list(train)
            train_labels_i = list(train_labels)
            if i < cv - 1:
                crossSet = list(train_i[i * di:(i + 1) * di])
                del train_i[i * di:(i + 1) * di]

                crossSet_labels = list(train_labels_i[i * di:(i + 1) * di])
                del train_labels_i[i * di:(i + 1) * di]
            else:
                crossSet = list(train_i[i * di:])
                del train_i[i * di:]

                crossSet_labels = list(train_labels_i[i * di:])
                del train_labels_i[i * di:]

            knn = Knn()
            knn.K = k
            knn.setTrainData(train_i, train_labels_i)
            acc = knn.getAccuracy(crossSet, crossSet_labels)
            accuracy_list.append(acc)

        return np.array(accuracy_list).mean()


if __name__ == '__main__':
    import load_datasets
    import time

    startTime = time.time()
    train_ratio: float = 0.9
    findBestKWithCrossValidation: bool = True

    print(f"Train ratio: {train_ratio} \n")

    print('-' * 175)
    print(f"Iris dataset classification: \n")
    train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)
    knn = Knn()
    knn.train(train, train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
    knn.test(test, test_labels)

    print('-'*175)
    print(f"Congressional dataset classification: \n")
    train, train_labels, test, test_labels = load_datasets.load_congressional_dataset(train_ratio)
    knn = Knn()
    knn.train(train, train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
    knn.test(test, test_labels)

    print('-' * 175)
    for i in range(3):
        print(f"Monks({i+1}) dataset classification: \n")
        train, train_labels, test, test_labels = load_datasets.load_monks_dataset(i+1)
        knn = Knn()
        knn.train(train, train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
        knn.test(test, test_labels)

        print('-' * 175)

    print(f"\n --- Elapse time: {time.time()-startTime:.2f} s --- \n")
