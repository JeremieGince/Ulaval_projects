from load_datasets import *
from BayesNaif import *



nbc = NbcGaussian()

train, train_labels, test, test_labels = load_iris_dataset(0.70)
nbc.train(train, train_labels)
nbc.test(test, test_labels)