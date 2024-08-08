# Perform KNN on both original data and on OOSE points
# Calculate overlap between the neighborhoods

# functions
# initialize based on datasets
# generate the knns
# calculate overlap


# visualize 
# neighborhood distribution based on class (embedding and data space)
# neighborhood size
import sklearn as sk
import torch
import manifoldlearning
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import numpy as np
import manifoldlearning

class KNN:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.initialized = False

    def generate_knns(self, n: int, m: int, nn: int = 100, r: int = 10, reset_knns: bool = False, debug: bool = True):
        # look over eigenvectors of entire dataset
        # save important properties in class variables that all members can access
        self.n = n      # total number of elements
        self.m = m      # number of labeled elements
        self.r = r      # number of eigenvectors 
        self.nn = nn    # number of nearest neighbors

        # print parameters
        if debug:
            debug_msg = f"""generate_knns() is being ran with the following parameters:
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
        if not self.initialized or reset_knns:
            self.indices = np.random.choice(len(self.samples), self.n, replace=False)
            self.train_samples = self.samples[self.indices]                
            self.train_labels = self.labels[self.indices]                                 

            self.label_ind = np.random.choice(self.n, self.m, replace=False)    # labeled indices 
            self.unlabel_ind = np.ones(self.n, dtype=bool)                      # unlabeled indices
            self.unlabel_ind[self.label_ind] = False       

        '''
        Generate knns
        '''

        self.orig_knn = NearestNeighbors(n_neighbors=nn)
        self.orig_knn.fit(self.train_samples[self.unlabel_ind])     # Orignal knn based on unlabeled points
        
        
        o = manifoldlearning.OOSE(self.train_samples, self.label_ind)
        (self.X0, self.OOSE_embeddings, self.dtests) = o.embed(r=self.r)               # OOSE embeddings (unlabeled points)
        self.oose_knn = NearestNeighbors(n_neighbors=nn)               
        self.oose_knn.fit(self.OOSE_embeddings)

        self.initialized = True
    
    @staticmethod
    def nn_overlap(nn_1, nn_2, k=100):
        if nn_1.shape != nn_2.shape:
            raise Exception("nn_overlap: Arrays are not the same size")

        if nn_1.ndim != 1:
            raise Exception("nn_overlap: Dimension is larger than 1")
    
        k = min(k, len(nn_1), len(nn_2))                # Ensure k <= total number of nearest neighbors

        overlap = np.intersect1d(nn_1[:k], nn_2[:k])    # returns array of shared indices

        # return overlap array and overlap score
        return (overlap, overlap.size/k)
    
    def knn_overlap(self, k):
        if not self.initialized:
            raise Exception("KNNs have not been generated")
        
        '''
        Get distances and indices of nearest neighbors (first self.nn nearest neighbors)
        '''
        [orig_dist, orig_nn] = self.orig_knn.kneighbors(self.train_samples[self.unlabel_ind])
        [oose_dist, oose_nn] = self.oose_knn.kneighbors(self.OOSE_embeddings)

        scores = []
        for orig_row, oose_row in zip(orig_nn, oose_nn):
            # Calculate the overlap using the first k neighbors
            _, score = KNN.nn_overlap(orig_row[:k], oose_row[:k], k)
            scores.append(score)

        return scores
    

def vary_knn(n: int, m: int, trials: int = 5, k: KNN = None, steps = None):
    if steps is None:
        raise ValueError("Steps must be provided.")
    if k is None:
        raise ValueError("KNN object must be provided.")
    
    '''
    For each trial, calculate knn overlap for each step
    '''
    scores = [[] for _ in range(trials)]
    for trial in range(trials):
        k.generate_knns(n=n, m=m, nn=max(steps), reset_knns=True)  # regenerate KNNs each trial (new indices)
        for step in steps:
            scores[trial].append(k.knn_overlap(step))

    return scores

def compare_distributions(k: KNN = None, nn: int = 100):
    if k is None:
        raise ValueError("KNN object must be provided.")
    if not k.initialized:
        raise ValueError("KNNs have not been generated yet")
    
    [orig_dist, orig_nn] = k.orig_knn.kneighbors(k.train_samples[k.unlabel_ind])
    [orig_dist, oose_nn] = k.oose_knn.kneighbors(k.OOSE_embeddings)

    '''
    Get training labels and labels for unlabeled points
    '''
    orig_labels = k.train_labels[k.unlabel_ind]
    '''
    Calculate number of labels that match the original label for each knn
    Parallelized for efficiency
    '''
    # print(train_labels[orig_nn[:, :nn]])
    # print(train_labels[oose_nn[:, :nn]])
    # print(orig_labels[:, np.newaxis])


    # orig labels or train labels? when indexing train_labels[orig_nn[:, :nn]]
    orig_score = np.sum(orig_labels[orig_nn[:, :nn]] == orig_labels[:, np.newaxis])
    oose_score = np.sum(orig_labels[oose_nn[:, :nn]] == orig_labels[:, np.newaxis])

    total_comparisons = (k.n-k.m) * nn
    return (orig_score / total_comparisons, oose_score / total_comparisons)

def print_distributions(index: int, k: KNN = None, nn: int = 100):
    if k is None:
        raise ValueError("KNN object must be provided.")
    if not k.initialized:
        raise ValueError("KNNs have not been generated yet")
    
    [orig_dist, orig_nn] = k.orig_knn.kneighbors(k.train_samples[k.unlabel_ind])
    [orig_dist, oose_nn] = k.oose_knn.kneighbors(k.OOSE_embeddings)

    '''
    Get training labels and labels for unlabeled points
    '''
    orig_labels = k.train_labels[k.unlabel_ind]

    '''
    Print parameters
    '''
    print("Parameters: ")
    print(f"Total nearest neighbors: {k.nn}")
    print(f"Number of nearest neighbors examined: {nn}\n")

    '''
    Print ground truth label and labels from each knn
    '''
    print(f"Original label:{orig_labels[index, np.newaxis]}")
    print(f"Original knn labels:{orig_labels[orig_nn[index, :nn]]}")
    print(f"Oose knn labels:{orig_labels[oose_nn[index, :nn]]}")

    '''
    Calculate number of labels that match the original label for each knn
    Parallelized for efficiency

    [3]
    '''
    orig_score = np.sum(orig_labels[orig_nn[index, :nn]] == orig_labels[index, np.newaxis])
    oose_score = np.sum(orig_labels[oose_nn[index, :nn]] == orig_labels[index, np.newaxis])

    total_comparisons = nn
    return (orig_score / total_comparisons, oose_score / total_comparisons)

