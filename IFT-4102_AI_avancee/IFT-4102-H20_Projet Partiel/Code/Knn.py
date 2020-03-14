from classifieur import *
import numpy as np
import util


class Knn(Classifier):
    """
    Knn is a classifier used to classify numeric data by classes.

    reference: https://fr.wikipedia.org/wiki/Recherche_des_plus_proches_voisins
    """

    defaultK: int = 5  # Default value of K, K is the number of nearest neighbors used to classify data.

    def __init__(self, **kwargs):
        """
        Constructor of Knn.
        :param kwargs: {
            Kmin: initialisation of the attribute _Kmin, must be higher than 0. (int),
            Kmax: initialisation of the attribute _Kmax, must be higher than 0. (int),
            K: initialisation of the attribute K, must be higher than 0. (int)
            distance_func: Distance function for the naive prediction. (func)
        }

        Attributes
        ----------
        :attr train_vector_to_label: Map vectors train_set data to it's label {data -> label} (MapHashVecLabel)
        :attr test_vector_to_label: Map vectors test_set data to it's label {data -> label} (MapHashVecLabel)
        :attr _Kmin: Value of the minimum K used in train_set method to fit the best K value. (int)
        :attr _Kmax: Value of the maximum K used in train_set method to fit the best K value. (int)
        :attr K: Value of the parameter K representing the number of nearest neighbors used to classify data. (int)
        :attr distance_func: Distance function for the naive prediction. (func)

        """
        super().__init__(**kwargs)
        self.train_vector_to_label: util.MapHashVecLabel = util.MapHashVecLabel()
        self.test_vector_to_label: util.MapHashVecLabel = util.MapHashVecLabel()
        self._Kmin: int = kwargs.get("Kmin", 1)
        self.K: int = kwargs.get("K", Knn.defaultK)
        self._Kmax: int = kwargs.get("Kmax", 25)
        self.distance_func = kwargs.get("distance_func", util.euclidean_distance)

    def setData(self, train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, test_labels: np.ndarray):
        """
        Put the data in memory by setting the attribute train_vector_to_label and test_vector_to_label.
        :param train: Array of training data (np.ndarray)
        :param train_labels: Array of the labels associated with the training data (np.ndarray)
        :param test: Array of testing data (np.ndarray)
        :param test_labels: Array of the labels associated with the testing data (np.ndarray)
        :return: None
        """
        self.setTrainData(train, train_labels)
        self.setTestData(test, test_labels)

    def setTrainData(self, train: np.ndarray, train_labels: np.ndarray):
        """
        Put the data in memory by setting the attribute train_vector_to_label.
        :param train: Array of training data (np.ndarray)
        :param train_labels: Array of the labels associated with the training data (np.ndarray)
        :return: None
        """
        assert len(train) == len(train_labels)
        self.train_vector_to_label = util.MapHashVecLabel({str(train[i]): train_labels[i] for i in range(len(train))})

    def setTestData(self, test: np.ndarray, test_labels: np.ndarray):
        """
        Put the data in memory by setting the attribute test_vector_to_label.
        :param test: Array of testing data (np.ndarray)
        :param test_labels: Array of the labels associated with the testing data (np.ndarray)
        :return: None
        """
        assert len(test) == len(test_labels)
        self.test_vector_to_label = util.MapHashVecLabel({str(test[i]): test_labels[i] for i in range(len(test))})

    def setKmin(self, new_Kmin: int):
        """
        Setter of Kmin attribute.
        :param new_Kmin: The new value of Kmin. (int)
        :return: None
        """
        assert new_Kmin >= 1
        self._Kmin = new_Kmin

    def getKmin(self) -> int:
        """
        Getter of Kmin attribute.
        :return: The value of Kmin. :rtype int
        """
        return self._Kmin

    def setKmax(self, new_Kmax: int):
        """
        Setter of Kmax attribute.
        :param new_Kmax: The new value of Kmax. (int)
        :return: None
        """
        assert new_Kmax >= 1
        self._Kmax = new_Kmax

    def getKmax(self) -> int:
        """
        Getter of Kmax attribute.
        :return: The value of Kmax. :rtype int
        """
        return self._Kmax

    def train(self, train_set: np.ndarray, train_labels: np.ndarray,
              verbose: bool = True, findBestKWithCrossValidation: bool = False) -> tuple:
        """
        TODO: La prédiction pourrait grandement être augmenté si nous utilisons une métric de distance
              pondéré comme la "util.weighted_euclidean_distance" et que nous utilisons la méthode train
              afin de trouver les poids idéaux selon l'article suivant:
              https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2006-Paredes-IEEETPAMI.pdf

        :param train_set: Ensemble de données d'entraînement (np.ndarray)
        :param train_labels: Ensembles des étiquettes des données d'entraînement (np.ndarray)
        :param verbose: Vrai si nous voulons afficher certaines statistiques, Faux sinon (boo;)
        :param findBestKWithCrossValidation: Vrai si on veux utiliser la cross validation pour trouver le K optimal,
        Faux sinon (bool)
        :return: confusionMatrix, accuracy, precision, recall (tuple)
        """
        self.setTrainData(train_set, train_labels)

        if findBestKWithCrossValidation:
            kToacc: dict = {k: self.crossValidation(train_set, train_labels, cv=5, k=k)
                            for k in range(self._Kmin, self._Kmax)}
            self.K = max(kToacc, key=kToacc.get)

        displayArgs = {"dataSize": len(train_set), "title": "Train results", "preMessage": f"Chosen K: {self.K} \n"}

        return self.test(train_set, train_labels, verbose, displayArgs)

    def predict(self, example, label) -> (int, bool):
        """
        Perform the classification of the current current example.

        Référence: https://fr.wikipedia.org/wiki/Recherche_des_plus_proches_voisins

        :param example: data sample (np.ndarray)
        :param label: the class of this example. (int)
        :return: predicted class of the sample, predicted class == label
        """
        return self.naivePrediction(example, label)

    def naivePrediction(self, example, label):
        """
        Perform the naive classification of the current current example.
        :param example: data sample (np.ndarray)
        :param label: the class of this example. (int)
        :return: predicted class of the sample, predicted class == label
        """
        train_data: util.MapHashVecLabel = self.train_vector_to_label.deepcopy()
        neighbors: list = list()

        while train_data:
            train_vector, cls = train_data.popitem()
            distance: float = self.distance_func(example, train_vector)
            neighbor: tuple = (train_vector, distance, cls)
            neighbors.append(neighbor)

        # nearest_neighbors: list = [vec for vec, dist, cls in sorted(neighbors, key=lambda nei: nei[1])[:self.K]]
        nearest_classes: list = [cls for vec, dist, cls in sorted(neighbors, key=lambda nei: nei[1])[:self.K]]
        nearest_classes_count: dict = {cls: nearest_classes.count(cls) for cls in set(nearest_classes)}
        prediction_cls: int = max(nearest_classes_count, key=nearest_classes_count.get)
        return prediction_cls, prediction_cls == label

    def test(self, test_set, test_labels, verbose: bool = True, displayArgs: dict = None) \
            -> (np.ndarray, float, float, float):
        self.setTestData(test_set, test_labels)
        return Classifier.test(self, test_set, test_labels, verbose, displayArgs)

    def getAccuracy(self, test_set=None, test_labels=None):
        if test_set is None or test_labels is None:
            test_set, test_labels = self.test_vector_to_label.asLists()
        return Classifier.getAccuracy(self, test_set, test_labels)

    def crossValidation(self, train_set, train_labels, cv: int = 5, k: int = defaultK) -> float:
        """
        Perform the cross validation for the given train_set data
        :param train_set: Training data (np.ndarray)
        :param train_labels: Training labels (np.ndarray)
        :param cv: Cross validation factor (int)
        :param k: The number of nearest neighbors used to classify data. (int)
        :return: cross validation :rtype float
        """
        di: int = int(len(train_set) / cv)

        accuracy_list: list = list()
        for i in range(cv):
            train_i = list(train_set)
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
            knn.setTrainData(np.array(train_i), np.array(train_labels_i))
            acc = knn.getAccuracy(np.array(crossSet), np.array(crossSet_labels))
            accuracy_list.append(acc)

        return np.array(accuracy_list).mean()


if __name__ == '__main__':
    import load_datasets
    import time

    train_ratio: float = 0.90
    findBestKWithCrossValidation: bool = True
    distanceFunc = util.chebyshev_distance

    print(f"Train ratio: {train_ratio}")
    print(f"findBestKWithCrossValidation: {findBestKWithCrossValidation}")
    print("\n")

    print('-' * 175)
    print(f"Iris dataset classification: \n")
    startTime = time.time()

    iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(train_ratio)
    iris_knn = Knn(distance_func=distanceFunc)
    iris_knn.train(iris_train, iris_train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
    iris_knn.test(iris_test, iris_test_labels)

    print(f"\n --- Elapse time: {1_000*(time.time() - startTime):.2f} ms --- \n")

    print('-'*175)
    print(f"Congressional dataset classification: \n")
    startTime = time.time()

    cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(train_ratio)
    cong_knn = Knn(distance_func=distanceFunc)
    cong_knn.train(cong_train, cong_train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
    cong_knn.test(cong_test, cong_test_labels)

    print(f"\n --- Elapse time: {1_000*(time.time() - startTime):.2f} ms --- \n")

    print('-' * 175)
    for i in range(3):
        print(f"Monks({i+1}) dataset classification: \n")
        startTime = time.time()

        monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i+1)
        monks_knn = Knn(distance_func=distanceFunc)
        monks_knn.train(monks_train, monks_train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
        monks_knn.test(monks_test, monks_test_labels)

        print(f"\n --- Elapse time: {1_000*(time.time() - startTime):.2f} ms --- \n")

        print('-' * 175)

