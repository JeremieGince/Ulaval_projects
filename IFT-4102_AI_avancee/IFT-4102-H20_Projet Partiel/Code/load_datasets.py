import random
import numpy as np


def load_iris_dataset(train_ratio: float) -> tuple:
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont etre attribués à l'entrainement,
        le rest des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisé
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    conversion_labels: dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    # TODO : le code ici pour lire le dataset

    data: list = list()
    data_labels: list = list()
    with open('datasets/bezdekIris.data', 'r') as file:
        lines: list = file.readlines()
        random.shuffle(lines)
        for line in lines:
            line_vectorized: list = line.strip().split(',')
            if line_vectorized and line_vectorized[-1] in conversion_labels:
                flower_class: int = conversion_labels[line_vectorized[-1]]
                train_data: list = [float(i) for i in line_vectorized[:-1]]
                data.append(train_data)
                data_labels.append(flower_class)

    split_idx: int = int(len(data)*train_ratio)

    train: np.ndarray = np.array(data[:split_idx])
    train_labels: np.ndarray = np.array(data_labels[:split_idx])

    test: np.ndarray = np.array(data[split_idx:])
    test_labels: np.ndarray = np.array(data_labels[split_idx:])

    return train, train_labels, test, test_labels


def load_congressional_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Vous pouvez utiliser un dictionnaire pour convertir les attributs en numériques 
    # Notez bien qu'on a traduit le symbole "?" pour une valeur numérique
    # Vous pouvez biensur utiliser d'autres valeurs pour ces attributs
    conversion_labels = {'republican': 0, 'democrat': 1,
                         'n': 0, 'y': 1, '?': 2}

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/house-votes-84.data', 'r')

    # TODO : le code ici pour lire le dataset

    # La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)


def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks
    
    Notez bien que ce dataset est différent des autres d'un point de vue
    exemples entrainement et exemples de tests.
    Pour ce dataset, nous avons 3 différents sous problèmes, et pour chacun
    nous disposons d'un fichier contenant les exemples d'entrainement et 
    d'un fichier contenant les fichiers de tests. Donc nous avons besoin 
    seulement du numéro du sous problème pour charger le dataset.

    Args:
        numero_dataset: lequel des sous problèmes nous voulons charger (1, 2 ou 3 ?)
		par exemple, si numero_dataset=2, vous devez lire :
			le fichier monks-2.train contenant les exemples pour l'entrainement
			et le fichier monks-2.test contenant les exemples pour le test
        les fichiers sont tous dans le dossier datasets
    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """

    # TODO : votre code ici, vous devez lire les fichiers .train et .test selon l'argument numero_dataset

    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)


if __name__ == '__main__':
    load_iris_dataset(0.7)
