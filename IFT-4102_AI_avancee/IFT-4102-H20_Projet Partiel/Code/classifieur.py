import numpy as np


class Classifier:
    """
	Parent class for every classifier. Abstract class.
	"""

    def __init__(self, **kwargs):
        """
		c'est un Initializer.
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
        pass

    def train(self, train: np.ndarray, train_labels: np.ndarray,
			  verbose: bool = True, **kwargs):
        """
		Used to train_set the current model.

		:param train: une matrice de type Numpy et de taille nxm, avec
		n : le nombre d'example d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)

		:param train_labels : est une matrice numpy de taille nx1
		:param verbose: True To print results else False. (bool)
		:param kwargs: Parameters to pass to child classes.
		"""

        raise NotImplementedError()

    def predict(self, example, label) -> (int, bool):
        """
		Prédire la classe d'un example donné en entrée
		:param example: Matrice de taille 1xm
		:param label: The label of the example.

		:return (prediction, prediction == label). :rtype (int, bool)

		"""
        raise NotImplementedError()

    def test(self, test_set, test_labels, verbose: bool = True, displayArgs: dict = None) \
            -> (np.ndarray, float, float, float):
        """
		c'est la méthode qui va tester votre modèle sur les données de test_set
		l'argument test_set est une matrice de type Numpy et de taille nxm, avec
		n : le nombre d'example de test_set dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)

		test_labels : est une matrice numpy de taille nx1

		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire

		Faites le test_set sur les données de test_set, et afficher :
		- la matrice de confision (confusion matrix)
		- l'accuracy
		- la précision (precision)
		- le rappel (recall)

		Bien entendu ces tests doivent etre faits sur les données de test_set seulement

		"""
        confusionMatrix: np.ndarray = self.getConfusionMatrix(test_set, test_labels)
        accuracy: float = self.getAccuracy(test_set, test_labels)
        precision = self.getPrecision(test_set, test_labels)
        recall = self.getRecall(test_set, test_labels)

        if verbose:
            if displayArgs is None:
                displayArgs = {"dataSize": len(test_set), "title": "Test results", "preMessage": ""}
            self.displayStats(confusionMatrix, accuracy, precision, recall, dataSize=displayArgs["dataSize"],
                              title=displayArgs["title"], preMessage=displayArgs["preMessage"])

        return confusionMatrix, accuracy, precision, recall

    def getAccuracy(self, test_set: np.ndarray, test_labels: np.ndarray) -> float:
        accuracy: float = 0
        for idx, exemple in enumerate(test_set):
            prediction, check = self.predict(exemple, test_labels[idx])
            accuracy += int(check)
        return 100 * (accuracy / len(test_set))

    def getConfusionMatrix(self, test_set: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
        # https://en.wikipedia.org/wiki/Confusion_matrix
        labels: list = sorted(list(set(test_labels)))
        labelsToCountclassification: dict = {lbl: [0 for _ in labels] for lbl in labels}
        for idx, example in enumerate(test_set):
            prediction, check = self.predict(example, test_labels[idx])
            labelsToCountclassification[test_labels[idx]][labels.index(prediction)] += 1

        confusionMatrix: np.ndarray = np.array([labelsToCountclassification[lbl] for lbl in labels]).transpose()
        return confusionMatrix

    def getPrecision(self, test_set: np.ndarray, test_labels: np.ndarray) -> np.float:
        # https://fr.wikipedia.org/wiki/Pr%C3%A9cision_et_rappel
        confusionMatrix: np.ndarray = self.getConfusionMatrix(test_set, test_labels)
        precisionVector: np.ndarray = np.array([vector[idx] / np.sum(vector)
                                                for idx, vector in enumerate(confusionMatrix)])
        return precisionVector.mean()

    def getRecall(self, test_set: np.ndarray, test_labels: np.ndarray) -> np.float:
        # https://fr.wikipedia.org/wiki/Pr%C3%A9cision_et_rappel
        confusionMatrix_T: np.ndarray = self.getConfusionMatrix(test_set, test_labels).transpose()
        recallVector: np.ndarray = np.array([vector[idx] / np.sum(vector)
                                             for idx, vector in enumerate(confusionMatrix_T)])
        return recallVector.mean()

    def displayStats(self, confusionMatrix: np.ndarray = None, accuracy: float = None, precision: float = None,
                     recall: float = None, dataSize: int = None, title: str = "", preMessage: str = ""):
        print((f"\n {title}:" if title else ""),
              f"Data set size: {dataSize}",
              (f"{preMessage}" if preMessage else ""),
              f"Confusion Matrix: \n {confusionMatrix}",
              f"Accuracy: {accuracy:.2f} %",
              f"Precision: {precision:.5f}",
              f"Recall: {recall:.5f}",
              sep='\n')
