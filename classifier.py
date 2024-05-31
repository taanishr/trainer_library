import numpy as np
import sklearn as sk
from sklearn import linear_model as linear
from sklearn import metrics as metrics
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import manifoldlearning

class ClassificationTests:
    '''
    Current architecture stinks, here's the TODO:
    1. Generate G, W, L, G_L and W_L per class
    2. Feed G_L and W_L into manifoldlearning.embed and into OOSE
    3. This will ensure we don't have different embeddings (possible reason for 5% performance)
    4. Sort of optional: Cayley transform implementation for OOSE (faster?)
    '''
    def __init__(self, samples, labels,  m: int, n: int, classifier: str = "hinge") -> None:
        self.samples = samples
        self.labels = labels
        self.m = m
        self.n = n
        self.label_ind = np.random.choice(n, m, replace=False)
        self.unlabel_ind = np.ones(samples.shape[0], dtype=bool)
        self.unlabel_ind[self.label_ind] = False
        self.classifier = linear.SGDClassifier(loss=classifier)

    def __train(self) -> None:
        (evs, eigs) = manifoldlearning.manifoldlearning.embed(self.samples[self.label_ind])
        self.classifier.fit(X=eigs,y=self.labels[self.label_ind])

    def mse(self) -> float:
        self.__train()
        #label_embeddings = manifoldlearning.manifoldlearning.embed(self.samples[self.unlabel_ind])
        o = manifoldlearning.OOSE(self.samples, self.label_ind)
        (OOSE_embeddings, *d_tests) = o.embed()
        #return self.classifier.score(OOSE_embeddings, label_embeddings)
        #OOSE_embeddings = np.array(OOSE_embeddings[:,1])
        #print(OOSE_embeddings)
        #return self.classifier.score(self.samples[self.unlabel_ind], OOSE_embeddings);
        return self.classifier.score(X=OOSE_embeddings, y=self.labels[self.unlabel_ind])


    
