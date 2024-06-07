# Perform KNN on both original data and on OOSE points
# Calculate overlap between the neighborhoods

# functions
# initialize based on datasets
# generate the knns
# calculate overlap

import sklearn as sk
import torch
import manifoldlearning
from sklearn.neighbors import kneighbors_graph
import numpy as np
import manifoldlearning

class KNNTests:
    def __init__(self, data, n: int, m: int):
        self.data = data
        self.label_ind = np.random.choice(n, m, replace=False)
        self.n = n
        self.m = m
        self.label_ind = np.random.choice(n, m, replace=False)
        self.unlabel_ind = np.ones(n, dtype=bool)
        self.unlabel_ind[self.label_ind] = False
        self.unlabel_data = data[self.unlabel_ind]

    def generate_knns(self, k=100):
        self.orig_knn = kneighbors_graph(self.unlabel_data, n_neighbors=k) 
        
        o = manifoldlearning.OOSE(self.data, self.label_ind)
        (OOSE_embeddings, *dtests) = o.embed()
        self.oose_knn = kneighbors_graph(OOSE_embeddings, n_neighbors=k)
    
    @staticmethod
    def nn_overlap(arr1, arr2, k=100):
        if arr1.shape != arr2.shape:
            raise Exception("nn_overlap: Arrays are not the same size")

        if arr1.ndim != 1:
            raise Exception("nn_overlap: Dimension is larger than 1")

        n = arr1.shape[0]
        score = 0

        for i, (x,y) in enumerate(zip(arr1,arr2)):
            if x == 1 and y == 1:
                #print(i)
                score = score + 1

        score = score / k

        return score

    def knn_overlap(self):
        orig_knn_i = self.orig_knn.toarray()
        oose_knn_i = self.oose_knn.toarray()
        
        n = oose_knn_i.shape[0]
        mean_score = 0

        n1 = np.linalg.norm(orig_knn_i-oose_knn_i)
        print(n1)

        for i, (x,y) in enumerate(zip(orig_knn_i, oose_knn_i)):
            '''
            Was playing around with randomly changing items to see if 100% thing was a bug. It does not seem to a bug. This algorithm is goated
            rand = np.random.choice(8000, 4000, replace=False)
            x[rand] = 0
            rand = np.random.choice(8000, 4000, replace=False)
            y[rand] = 0
            '''
            mean_score = mean_score + KNNTests.nn_overlap(x,y)
            #print("-------")

        mean_score = mean_score / n 

        return mean_score

  
 # TODO: vary eigenvectors
