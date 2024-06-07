import numpy as np
import sklearn as sk
from sklearn import linear_model as linear
from sklearn import svm as svm
from sklearn import metrics as metrics
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import manifoldlearning
import matplotlib.pyplot as plt

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
        #self.classifier = linear.SGDClassifier(loss=classifier)
        self.classifier = svm.LinearSVC(C=15)
        self.trained = False

    def __generate_graphs(self):
        self.W, self.G, self.L = manifoldlearning.manifoldlearning.generate_knn(self.samples)
        self.G_l, self.L_l = manifoldlearning.manifoldlearning.partition_knn(self.G, self.label_ind)
    
    def __train_helper(self) -> None:
        #(evs, eigs) = manifoldlearning.manifoldlearning.embed(self.samples[self.label_ind])
        (ev, self.eigs) = manifoldlearning.manifoldlearning.embed(G=self.G_l) # save training points
        self.classifier.fit(X=self.eigs,y=self.labels[self.label_ind])

    def train(self):
        self.__generate_graphs()
        self.__train_helper()
        self.trained = True

    def ground_accuracy(self):
        #self.W, self.G, self.L = manifoldlearning.manifoldlearning.generate_knn(self.samples)
        #self.G_l, self.L_l = manifoldlearning.manifoldlearning.partition_knn(self.G, self.label_ind)
        #self.__train()
        if not self.trained:
            return Exception("Call train() first")
        (ev, eigs) = manifoldlearning.manifoldlearning.embed(G=self.G_l)
        return self.classifier.score(X=eigs, y=self.labels[self.label_ind])

    def OOSE_accuracy(self) -> float:
        #self.W, self.G, self.L = manifoldlearning.manifoldlearning.generate_knn(self.samples)
        #self.G_l, self.L_l = manifoldlearning.manifoldlearning.partition_knn(self.G, self.label_ind)
        #self.__train()
        #label_embeddings = manifoldlearning.manifoldlearning.embed(self.samples[self.unlabel_ind])
        if not self.trained:
            return Exception("Call train() first")
        o = manifoldlearning.OOSE(self.samples, self.label_ind)
        (self.X0, self.OOSE_embeddings, d_tests) = o.embed(W=self.W, G=self.G, L=self.L, G_l=self.G_l, L_l=self.L_l)
        print(self.X0)
        print(self.OOSE_embeddings)
        #return self.classifier.score(OOSE_embeddings, label_embeddings)
        #OOSE_embeddings = np.array(OOSE_embeddings[:,1])
        #print(OOSE_embeddings)
        #return self.classifier.score(self.samples[self.unlabel_ind], OOSE_embeddings);
        score = self.classifier.score(X=self.OOSE_embeddings, y=self.labels[self.unlabel_ind])
        self.predict = self.classifier.predict(X=self.OOSE_embeddings)
        return score, self.predict

   
    # plots a row
    def visualize_preds_helper(self,x, y, z, color, size=(30,30), theta=120):
        num_subplots = (int)(360 / theta) + 1
        fig = plt.figure(figsize=size)
        for i in range(1,num_subplots):
            ax = fig.add_subplot(1, num_subplots, i, projection='3d');
            ax.view_init(azim=theta*i)
            ax.scatter(x,y,z, c=color)

    def visualize_preds(self,subplots):
        sorted_pred_ind = self.predict.argsort()
        y_pred_sorted = self.predict[sorted_pred_ind]
        Xk_sorted = self.OOSE_embeddings[sorted_pred_ind]
        for subplot in subplots:
            self.visualize_preds_helper(Xk_sorted[:,subplot[0]], Xk_sorted[:,subplot[1]], Xk_sorted[:,subplot[2]], color=y_pred_sorted)

    def visualize_pts(self):
        #plt.scatter(self.OOSE_embeddings[:, 0], self.OOSE_embeddings[:, 1],  c='black')
        plt.scatter(self.eigs[:, 0], self.eigs[:,1],  c='red')
        plt.scatter(self.X0[:, 0], self.X0[:,1], c='blue')
