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

'''
TODO:
Return intermdiates, not scores
seperation of functions
randomzie samples, not e.v.s
Add comments

Need to fix classification experiments too
Also need to look into fixing knn construction. Doesnt seem to generate knns that are fully connected all the time
This is a big problem
'''

class Classifier:
    # Every time you do train call, I want to generate a new set of label and unlabelled indicies
    def __init__(self, samples, labels) -> None:
        self.samples = samples
        self.labels = labels
        self.classifier = svm.LinearSVC(C=15)
        self.trained = False

    def __generate_graphs(self):
        # generate knn over all training samples
        self.W, self.G, self.L = manifoldlearning.manifoldlearning.generate_knn(self.train_samples, n_neighbors=self.nn)

        # partition based on labeled indices
        # self.G_l, self.L_l = manifoldlearning.manifoldlearning.partition_knn(self.G, self.label_ind)
        self.G_l, self.L_l = manifoldlearning.manifoldlearning.create_subknn(self.train_samples[self.label_ind], n_neighbors=self.nn)
    
    def __train_helper(self) -> None:
        # train classifier based on embeddings (eigenvectors and eigenvalues of samples)
    
        (self.ev, self.eigs) = manifoldlearning.manifoldlearning.embed(r=self.r,G=self.G_l) 
        self.classifier.fit(X=self.eigs[:,1:self.r],y=self.train_labels[self.label_ind])

    def train(self, n, m, nn: int = 100, r: int = 10, reset_indices: bool = False, debug: bool = True):
        # save important properties in class variables that all members can access
        self.n = n      # total number of elements
        self.m = m      # number of labeled elements
        self.r = r      # number of eigenvectors 
        self.nn = nn    # number of nearest neighbors
        
        # print parameters
        if debug:
            debug_msg = f"""train() is being ran with the following parameters:
            Number of samples: {self.n}
            Number of labeled samples: {self.m}
            Nearest neighbors: {self.nn}
            Eigenvectors: {self.r}
            """
            print(debug_msg)

        '''
            Pick training indices from samples
            If not trained or reset_indices is set to true, indices will be reset
        '''
        if not self.trained or reset_indices:
            self.indices = np.random.choice(len(self.samples), self.n, replace=False)
            self.train_samples = self.samples[self.indices]                
            self.train_labels = self.labels[self.indices]                                 

            self.label_ind = np.random.choice(self.n, self.m, replace=False)    # labeled indices 
            self.unlabel_ind = np.ones(self.n, dtype=bool)                      # unlabeled indices
            self.unlabel_ind[self.label_ind] = False                            

        self.__generate_graphs()
        self.__train_helper()

        # Set trained to true
        self.trained = True

    def ground_score(self):
        # return ground score of classifier (training accuracy)
        if not self.trained:
            self.train()
        return self.classifier.score(X=self.eigs[:,1:self.r], y=self.train_labels[self.label_ind])

    def OOSE_score(self) -> float:
        # return OOSE score of classifier
        if not self.trained:
            self.train()
        o = manifoldlearning.OOSE(self.train_samples, self.label_ind)
        (self.X0, self.OOSE_embeddings, self.d_tests) = o.embed(r=self.r, W=self.W, L=self.L, vecs=self.eigs, vals=self.ev)
        self.score = self.classifier.score(X=self.OOSE_embeddings, y=self.train_labels[self.unlabel_ind])
        self.predict = self.classifier.predict(X=self.OOSE_embeddings)
        return self.score, self.predict

   
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
        # Plotting first and second eigenvectors in 2D. We ignore the 0 eigenvector here.
        plt.scatter(self.eigs[:, 1], self.eigs[:,2],  c='red')
        plt.scatter(self.X0[:, 0], self.X0[:,1], c='blue')

class ClassificationExperiments:
    @staticmethod
    def run(tester: Classifier, n: int, m:int, nn: int = 100, r: int = 10, trials: int = 5):
        if (tester is None):
            error("Tester not provided")
        ground_scores = []
        OOSE_scores = []
        for i in range(trials):
            tester.train(n, m, nn=nn, r=r, reset_indices=True)
            gs = tester.ground_score()
            os, op = tester.OOSE_score()
            ground_scores.append(gs)
            OOSE_scores.append(os)
            print("Completed trial {i}".format(i=i))

        return ground_scores, OOSE_scores

    @staticmethod
    def vary_eigenvectors(tester: Classifier, n: int, m: int, nn: int = 100, steps = None, trials: int = 5):
        ground_scores = []
        OOSE_scores = []
        for step in steps:
            gs, os = ClassificationExperiments.run(tester, n, m, nn=nn, r=step, trials=trials)
            ground_scores.append(gs)
            OOSE_scores.append(os)
            print("Completed step {i}".format(i=step))
        return ground_scores, OOSE_scores

    @staticmethod
    def vary_neighbors(tester: Classifier, n: int, m: int, steps = None, r: int = 10, trials: int = 5):
        ground_scores = []
        OOSE_scores = []
        for step in steps:
            gs, os = ClassificationExperiments.run(tester, n, m, nn=step, r=r, trials=trials)
            ground_scores.append(gs)
            OOSE_scores.append(os)
            print("Completed step {i}".format(i=step))

        return ground_scores, OOSE_scores