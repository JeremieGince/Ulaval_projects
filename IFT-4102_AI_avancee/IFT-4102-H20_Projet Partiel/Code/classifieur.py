"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement
	* predict 	: pour prédire la classe d'un exemple donné
	* test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes test, predict et test de votre code.
"""

import numpy as np


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Classifier:

	def __init__(self, **kwargs):
		"""
		c'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		pass
		
	def train(self, train, train_labels):
		"""
		c'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		
		
		------------
		Après avoir fait l'entrainement, faites maintenant le test sur 
		les données d'entrainement
		IMPORTANT : 
		Vous devez afficher ici avec la commande print() de python,
		- la matrice de confision (confusion matrix)
		- l'accuracy
		- la précision (precision)
		- le rappel (recall)
		
		Bien entendu ces tests doivent etre faits sur les données d'entrainement
		nous allons faire d'autres tests sur les données de test dans la méthode test()
		"""
		raise NotImplementedError()

	def predict(self, exemple, label):
		"""
		Prédire la classe d'un exemple donné en entrée
		exemple est de taille 1xm
		
		si la valeur retournée est la meme que la veleur dans label
		alors l'exemple est bien classifié, si non c'est une missclassification

		"""
		raise NotImplementedError()

	def test(self, test, test_labels):
		"""
		c'est la méthode qui va tester votre modèle sur les données de test
		l'argument test est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		test_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		Faites le test sur les données de test, et afficher :
		- la matrice de confision (confusion matrix)
		- l'accuracy
		- la précision (precision)
		- le rappel (recall)
		
		Bien entendu ces tests doivent etre faits sur les données de test seulement
		
		"""
		raise NotImplementedError()

	def getAccuracy(self, test, test_labels) -> float:
		accuracy: float = 0
		for idx, exemple in enumerate(test):
			prediction, check = self.predict(exemple, test_labels[idx])
			accuracy += int(check)
		return 100 * (accuracy / len(test))

	def getConfusionMatrix(self, test, test_labels) -> np.ndarray:
		labels: list = sorted(list(set(test_labels)))
		labelsToCountclassification: dict = {lbl: [0 for _ in labels] for lbl in labels}
		for idx, example in enumerate(test):
			prediction, check = self.predict(example, test_labels[idx])
			labelsToCountclassification[test_labels[idx]][labels.index(prediction)] += 1

		confusionMatrix: np.ndarray = np.array([labelsToCountclassification[lbl] for lbl in labels]).transpose()
		return confusionMatrix

	def getPrecision(self):
		raise NotImplementedError()

	def getRecall(self):
		raise NotImplementedError()
