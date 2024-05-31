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
    def __init__(self, samples: npt.NDArray[Any], m: int, n: int, ) -> None:
        self.samples = samples
        self.m = m
        self.n = n
        self.unlabel_ind = np.random.choice(n, m, replace=False)
        self.label_ind = np.ones(samples.shape[0], dtype=bool)
        self.label_ind[self.unlabel_ind] = False
        self.regressionmodel = linear.Ridge()

    def __train(self) -> None:
        embeddings = manifoldlearning.embed(self.samples[self.unlabel_ind])
        # self.classifier.fit(X=self.samples, Y=embeddings)

    def mse(self) -> float:
        self.__train()
        # label_embeddings = OOSE.embed(self.samples[self.unlabel_ind])
        # OOSE_embeddings = OOSE(self.samples[self.unlabel_ind])
        # return self.classifier.score(OOSE_embeddings, label_embeddings)


