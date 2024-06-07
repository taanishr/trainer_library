import numpy as np
import numpy.typing as npt
import sklearn as sk
from sklearn import linear_model as linear
from sklearn import metrics as metrics
import torch
from torch.utils.data import Dataset
from torchvision import datasets as datasetsta
from torch.utils.data import DataLoader
import manifoldlearning

# SPECIFICATIONS:
# Perform regression on OOSE points
# Use MSE

class RegressionTests:
    def __init__(self, samples, labels,  m: int, n: int) -> None:
        self.samples = samples
        self.labels = labels
        self.m = m
        self.n = n
        self.label_ind = np.random.choice(n, m, replace=False)
        self.unlabel_ind = np.ones(samples.shape[0], dtype=bool)
        self.unlabel_ind[self.label_ind] = False
        self.regressor = linear.Ridge()
        self.trained = False

    def __generate_graphs(self):
        self.W, self.G, self.L = manifoldlearning.manifoldlearning.generate_knn(self.samples)
        self.G_l, self.L_l = manifoldlearning.manifoldlearning.partition_knn(self.G, self.label_ind)

    def __train_helper(self) -> None:
        (ev, eigs) = manifoldlearning.manifoldlearning.embed(G=self.G_l)
        self.regressor.fit(X=eigs, y=self.labels[self.label_ind])

    def train(self) -> None:
        self.__generate_graphs()
        self.__train_helper()
        self.trained = True

    def ground_accuracy(self):
        if not self.trained:
            return Exception("Call train() first")
        (ev, eigs) = manifoldlearning.manifoldlearning.embed(G=self.G_l)
        return self.regressor.score(X=eigs, y=self.labels[self.label_ind])

    def OOSE_accuracy(self) -> float:
        if not self.trained:
            return Exception("Call train() first")
        o = manifoldlearning.OOSE(self.samples, self.label_ind)
        (OOSE_embeddings, *d_tests) = o.embed(W=self.W, G=self.G, L=self.L, G_l=self.G_l, L_l=self.L_l)
        score = self.regressor.score(X=OOSE_embeddings, y=self.labels[self.unlabel_ind])
        predict = self.regressor.predict(X=OOSE_embeddings)
        return score, predict
        
        # label_embeddings = OOSE.embed(self.samples[self.unlabel_ind])
        # OOSE_embeddings = OOSE(self.samples[self.unlabel_ind])
        # return self.classifier.score(OOSE_embeddings, label_embeddings)


