# In[3]:


import os


# In[4]:


os.environ['JAX_PLATFORMS'] = 'cpu'


# In[5]:


import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
import scipy
import scipy.misc
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn import svm
from sklearn import linear_model as linear
from sklearn import metrics as metrics

import graphlearning as gl
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


# In[6]:


import sys

sys.path.append('../')


# In[7]:


import knn


# In[8]:


fmnist = datasets.FashionMNIST("/home/taanish/FashionMNIST", train=True, download=False)


# In[9]:


X = fmnist.data.numpy()
X = np.reshape(X, (60000, 784))
labels = fmnist.targets.numpy()[:60000]


# In[10]:


import manifoldlearning


# In[11]:


import classifier


# In[12]:


cl = classifier.Classifier(X, labels)


# In[ ]:


cl.train(n=10000, m=500, nn=50)

