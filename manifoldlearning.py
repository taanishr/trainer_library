# Code for OOSE library
# Makes code more reuseable, easier to deal with

# TODO: cayley transform + cuda

'''
directives
'''
import graphlearning as gl

from jax import numpy as jnp

from jax.experimental import sparse as jsparse

import numpy as np

import numpy.typing as npt

from typing import Any

import scipy

import optim as optim

import pyamg

class nystrom:
    def __init__(self, data, label_ind):
        if (data.shape[0] != label_ind.shape[0]): # ensure labels is 1 dimensional
            raise RuntimeError("Mismatched data and label sizes")
        self.n = label_ind.shape[0]
        self.m1 = data.shape[0]
        self.m2 = self.n - self.m2
        self.data = data
        self.label_ind = label_ind

    def embed(self):
        '''
        unlabeled mask, useful for later calculation
        '''
        ul_ind = np.ones(self.n, type=bool)
        ul_ind[self.label_ind] = False

        '''
        Generate knn and labeled subgraph
        '''
        G, W, L = manifoldlearning.generate_knn(self.data)
        # G_l, L_l = manifoldlearning.partition_knn(G, self.label_ind)

        '''
        Labeled and unlabeled-labeled laplacians
        '''
        L_l = L[self.label_ind][:,self.label_ind].toArray()
        L_12 = L[self.label_ind][:,ul_ind].toArray()

        '''
        Normalized labeled and unlabeled-labeled laplacians
        '''
        L_lnorm = np.linalg.norm(L_l, axis=0)
        L_12norm = np.linalg.norm(L_12, axis=0)

        A = L_l / L_lnorm
        B = L_12 / L_12norm

        A_i = scipy.linalg.sqrtm(A)
        
        '''
        Calculate nystrom embeddings according to Fowlkes et al.
        '''
        Q = A + A_i @ B @ B.T @ A_i
        (U,S,T) = scipy.linalg.svd(Q)
        V=np.concatenate((A, B.T),0) @ A_i @ U @ scipy.linalg.pinv(np.sqrt([S]))

        return V
    

class OOSE:
    def __init__(self, data, label_ind):
        self.n = data.shape[0]
        self.m1 = label_ind.shape[0]
        self.m2 = self.n - self.m1
        self.data = data
        self.label_ind = label_ind

    def init(self, r: int = 10, c0: int = 2., W = None, L = None, vecs = None, vals = None):
        '''
        ones vectors, useful for later calculation
        '''
        om = np.ones((self.m1,1))
        om2 = np.ones((self.m2,1))

        '''
        unlabeled mask, useful for later calculation
        '''
        ul_ind = np.ones(self.n, dtype=bool)
        ul_ind[self.label_ind] = False

        '''
        Generate knn and labeled subgraph
        '''
        if W is None or L is None or vecs is None or vals is None:
            # W, L, vecs or vals not found. Generating new embeddings
            W, G, L = manifoldlearning.generate_knn(self.data)
            G_l, L_l = manifoldlearning.partition_knn(G, self.label_ind)
            vals, vecs = G_l.eigen_decomp(k=r)

        #if G_l == None or L_l == None:
        #    G_l, L_l = manifoldlearning.partition_knn(G, self.label_ind)

        '''
        Perform eigenvector decomposition over labeled subgraph 
        and sort eigenvectors/eigenvalues
        '''
        #vals, vecs = G_l.eigen_decomp(k=r)
        s_ind = np.argsort(vals)
        vals[s_ind]
        vecs = np.array(vecs[:,s_ind])
        X1 = vecs[:,1:r] # throw away first eigenvector, corresponding with eve 0
        '''
        Calculate c1 and P from OOSE paper
        '''
        c1 = -X1.T@om
        self.C = c0*np.eye(r-1) - X1.T@X1 - 1.0/self.m2 * (c1@c1.T)

        '''
        Store unlabeled laplacian (L_22 in block form)
        '''
        L_22_s = L[ul_ind][:,ul_ind]
        L_22 = jsparse.BCOO.from_scipy_sparse(L_22_s)

        '''
        Projection onto set of mean-zero vectors 
        '''
        # NOTE: JNP and NP are not exactly interchangable
        # v is the same as om2, but as a jax np array
        # this is to facilitate gradient descent in optim.py
        # optim.py unfortunately uses vmap, which efficiently batches
        # a function call over an array
        v_s = np.expand_dims(np.ones(self.m2),1)
        o_s = 1/np.sqrt(self.n*v_s)
        v = jnp.expand_dims(jnp.ones(self.m2),1)
        o = 1/jnp.sqrt(self.n)*v
        #v = np.expand_dims(np.ones(self.m2),1)
        #o = 1/np.sqrt(self.n)*om2
        P_s = lambda x : x - o_s@(o_s.T@x)
        self.A_s = lambda x : P_s(L_22_s@P_s(x))
        
        P = lambda x : x - o@(o.T@x)
        self.A = lambda x : P(L_22@P(x))


        self.B = P(1/self.m2 * L_22@om2@c1.T +  L[ul_ind][:,self.label_ind]@X1) # Second matrix is L_21 (labelled to unlabelled)

        '''
        Calculate eigenvectors of A 
        Randomly initializes approximation eigenvectors,
        then uses LOBPCG to solve generalized eigenproblem (which starts with approximation)
        '''
        
        A_ = scipy.sparse.linalg.LinearOperator(L_22.shape, self.A_s)
        X = np.random.randn(self.B.shape[0], r) # randomly initialized approximation for eigenvectors
        l, Vg = scipy.sparse.linalg.lobpcg(A_, X, M=None, tol=1e-8, largest=False,
                                        verbosityLevel=0,
                                        maxiter=1000) # solve generalized eigenproblem
        
        eigenVectors = Vg
        eigenValues = l

        s = eigenValues.argsort()
        vals = eigenValues[s]
        vecs = eigenVectors[:,s]

        '''
        Fine tune initialization
        '''
        W0 = W[ul_ind][:, self.label_ind]
        d0 = np.array(np.reciprocal(W0.sum(-1))).squeeze()
        D0 = scipy.sparse.spdiags(d0, 0, self.m2, self.m2)
        L0 = D0@W0

        self.X0 = L0@X1

    '''
    Perform grad descent to further fine tune initializations (analogous to SSM)
    '''
    def optim(self, beta: float = 0.9, start: int = 0, stop: int = 100, num: int = 40):
        step_sizes = 1*np.power(beta,np.linspace(start,stop,num=num))
        #step_sizes = np.append(step_sizes,0)
        step_sizes=np.array(step_sizes)
        Xk, FKs, FOCs, lmaxs = optim.optim.optim(self.X0, self.A, self.B, self.C, step_sizes=step_sizes)
        self.Xk = Xk
        return self.Xk, FKs, FOCs, lmaxs
    
    def embed(self, r: int = 10, c0: int = 2., beta: float = 0.9, start: int = 0, stop: int = 100, num: int = 40, W = None, L = None, vecs=None, vals=None):
        self.init(r, c0, W=W, L=L, vecs=vecs, vals=vals)
        (self.OOSE_embeddings, *dtests) = self.optim(beta, start, stop, num) 
        return self.X0, self.OOSE_embeddings, dtests 

class manifoldlearning:
    @staticmethod
    def embed(X: npt.NDArray[Any] = None, G: gl.graph = None, r: int=10):
        if X is None and G is None:
            raise Exception("No data or graph provided")
        if X is not None and G is not None:
            raise Exception("Both data and graph provided")
        if G is None:
            W, G, L = manifoldlearning.generate_knn(X)

        ev, eigs = G.eigen_decomp(k=r)
        #ev = ev[1:r]
        #eigs = eigs[:,1:r]
        return ev, eigs

    # lower nn's to like 30 and do two seperate knns
    # calculate degree between labeleld and unlabelled
    # calculate first half block seperately
    # top right and bottom left are the same blocks

    #def embed(G: gl.graph, r: int=10):
    #    ev, eigs = G.eigen_decomp(k=r)
    #    ev = ev[1:r]
    #    eigs = eigs[1:,1:r]
    #    return ev, eigs
    
    @staticmethod
    def generate_knn(data, n_neighbors: int = 100, bandwidth: int = 0.2, kernel: str='symgaussian', normalization: str='combinatorial'):
        W = gl.weightmatrix.knn(data=data,k=n_neighbors,kernel=kernel)
        G = gl.graph(W)
        L = G.laplacian(normalization=normalization)
        return W, G, L
    
    @staticmethod
    def connect_graph(G):
        '''
        Connect a graph by connecting each node not in the largest connected component
        to the closet node in largest connected component
        '''

        connections = []

        # get LCC and non_LCC indices
        (lcc, lcc_ind) = G.largest_connected_component()
        non_lcc_ind = np.setdiff1d(np.arange(G.weight_matrix.shape[0]), lcc_ind)

        # find shortest distances
        for node in non_lcc_ind:
            (shortest_dist, lcc_node) = G.dijkstra(lcc_ind, node, return_cp=True)
            best_connection = (node, lcc_node[0], shortest_dist[0]) 
            connections.append(best_connection)
            # shortest_dist = float('inf')
            # best_connection = None

            # for lcc_node in lcc_ind:
            #     dist = G.distance(node, lcc_node)
            #     if dist < shortest_dist:
            #         shortest_dist = dist
            #         best_connection = (node, lcc_node, shortest_dist)
                
            # if (best_connection):
            #     connections.append(best_connection)

        '''
            update weight matrix
            assume symgaussian weight calculation
        '''
        for connection in connections:
            G.weight_matrix[connection[0], connection[1]] = connection[2]
            



    
    @staticmethod
    def partition_knn(G, ind, normalization: str='combinatorial'):
        g_partition = G.subgraph(ind=ind)
        # if not g_partition.isconnected():
        #     raise Exception("Graph is not connected")
        while not g_partition.isconnected():
            manifoldlearning.connect_graph(g_partition)
        l_partition = g_partition.laplacian(normalization=normalization)
        return g_partition, l_partition
    
    @staticmethod
    def create_subknn(partition, n_neighbors, bandwidth: int = 0.2, kernel: str='symguassian', normalization: str='combinatorial'):
        _, g_partition, l_partition = manifoldlearning.generate_knn(data=partition, n_neighbors=n_neighbors, bandwidth=bandwidth, )
        # l_partition = g_partition.laplacian(normalization=normalization)
        return g_partition, l_partition


    # @staticmethod
    # def create_subknn(partition, n_neighbors, bandwidth: int = 0.2, kernel: str='symgaussian' normalization: str='combinatorial'):
    #     g_partition = generate_knn(data=partition, n_neighbors=n_neighbors, bandwidth=bandwidth, )
    #     l_partition = g_partition.laplacian(normalization=normalization)
    #     return g_partition, l_partition


    @staticmethod
    def OOSE(X):
        pass
