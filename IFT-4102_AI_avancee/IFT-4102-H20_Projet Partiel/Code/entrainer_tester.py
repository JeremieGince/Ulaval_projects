import time

import load_datasets
import util
from BayesNaif import Nbc, NbcGaussian
from Knn import Knn

if __name__ == '__main__':
    ###################################################################################################################
    #  Partie 1 - Knn
    ###################################################################################################################

    knn_train_ratio: float = 0.9  # Ici, on initialise train ratio pour le Knn
    findBestKWithCrossValidation: bool = True  # Ici, on veut faire de la cross validation afin d'optimiser nos K
    distanceFunc = util.euclidean_distance  # On choisi notre métric de distance comme étant la distance euclidienne

    print(f"Knn Train ratio: {knn_train_ratio}")
    print(f"findBestKWithCrossValidation: {findBestKWithCrossValidation}")
    print("\n")

    print('-' * 175)
    print(f"Iris dataset classification: \n")
    startTime = time.time()

    #  Entrainement sur l'ensemble de données Iris

    iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(knn_train_ratio)
    iris_knn = Knn(distance_func=distanceFunc)
    iris_knn.train(iris_train, iris_train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
    iris_knn.test(iris_test, iris_test_labels)

    print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

    print('-' * 175)
    print(f"Congressional dataset classification: \n")
    startTime = time.time()

    #  Entrainement sur l'ensemble de données Congressional

    cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(
        knn_train_ratio)
    cong_knn = Knn(distance_func=distanceFunc)
    cong_knn.train(cong_train, cong_train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
    cong_knn.test(cong_test, cong_test_labels)

    print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

    #  Entrainement sur les ensembles de données Monks

    print('-' * 175)
    for i in range(3):
        print(f"Monks({i + 1}) dataset classification: \n")
        startTime = time.time()

        monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i + 1)
        monks_knn = Knn(distance_func=distanceFunc)
        monks_knn.train(monks_train, monks_train_labels, findBestKWithCrossValidation=findBestKWithCrossValidation)
        monks_knn.test(monks_test, monks_test_labels)

        print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

        print('-' * 175)

    ###################################################################################################################
    #  Partie 2 - Bayes Naif
    ###################################################################################################################

    nbc_train_ratio: float = 0.4

    print(f"Bayes Naif Train ratio: {nbc_train_ratio}")
    print("\n")

    print('-' * 175)
    print(f"Iris dataset classification: \n")
    startTime = time.time()

    iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(nbc_train_ratio)
    iris_knn = NbcGaussian()

    #  Entrainement sur l'ensemble de données Iris

    iris_knn.train(iris_train, iris_train_labels)
    iris_knn.test(iris_test, iris_test_labels)

    print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

    print('-' * 175)
    print(f"Congressional dataset classification: \n")
    startTime = time.time()

    #  Entrainement sur l'ensemble de données Congressional

    cong_train, cong_train_labels, cong_test, cong_test_labels = load_datasets.load_congressional_dataset(
        nbc_train_ratio)
    cong_knn = Nbc()
    cong_knn.train(cong_train, cong_train_labels)
    cong_knn.test(cong_test, cong_test_labels)

    print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

    #  Entrainement sur les ensembles de données Monks

    print('-' * 175)
    for i in range(3):
        print(f"Monks({i + 1}) dataset classification: \n")
        startTime = time.time()

        monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(i + 1)
        monks_knn = Nbc()
        monks_knn.train(monks_train, monks_train_labels)
        monks_knn.test(monks_test, monks_test_labels)

        print(f"\n --- Elapse time: {time.time() - startTime:.2f} s --- \n")

        print('-' * 175)
