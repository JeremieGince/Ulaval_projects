À ouvrir avec Notepad afin que le rendu de ce fichier soit conforme avec les exigence.

Description des classes

Fichier	     		Classe				Description
load_datasets.py	load_iris_dataset		Permet  de charger le dataset Iris en mémoire.
			load_congressional_dataset	Permet  de charger le dataset Congressional en mémoire.
			load_monks_dataset		Permet de charger le dataset Monks en mémoire.

util.py			MapHashVecLabel			Est un dictionnaire permettant l'utilisation de liste en guise de clef.
			GaussianDistribution		Représentation d'une distribution gaussienne avec une variance et une moyenne donnée.

		
classifieur.py		Classifier			Classe de base pour les méthodes de classification. Regroupe les fonctionnalités communes de ces méthodes.
			

BayesNaif.py		Nbc				Naive Bayesian Classifier : permet d'entrainer et de tester un modèle Bayésien naif. Celui-ci est de nature discrète.
			NbcGaussian			Permet d'entrainer un modèle Bayésien naif avec des features de type continu et dont la distribution de probabilité suit une courbe gaussienne.

Knn.py			Knn				K Nearest Neighbors: permet d'entrainer et de tester un modèle des k plus proches voisins.


Répartition des tâches
Programmation du KNN : Jérémie
Programmation du NBC : Richard
Rédaction du README: Richard
Définition de la matrix de confusion, du rappel et de la précision : Jérémie
Écriture du rapport: Richard/Jérémie


Difficultées rencontrées
-Pour le NBC, nous avons eu de la difficulté a établir les structures de données qui contiennent 
les résultats de l'entrainement. Nous avons opté pour des structures de bases en python (des listes et des dictionnaires).
Il y a cependant place à l'amélioration. Celles-ci pourraient être caché sous des classes qui exposeraient des fonctionnalitées.
Le code serait ainsi plus simple et la facilité de lecture du code serait amélioré. 

- Pour ce qui est du Knn, nous avons eu beaucoup de difficulté à trouver un moyen afin de préentraîner le modèle dans l'optique 
d'optimiser la prédiction du Knn. Malheureusement, après beaucoup de recherche sur internet, aucune méthode satifesant nos exigences
a été trouvé dans un délais raisonnable. Une telle méthode nous aurait permit d'effectuer un préentraînement sur les données et ainsi
accélérer la prédiction de notre Knn qui est plutôt lent à ce niveau. 