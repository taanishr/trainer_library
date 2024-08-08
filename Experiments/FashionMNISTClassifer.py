# TODO: Clear graph after each plt.savefig
# coding: utf-8

# In[1]:
import os

# In[2]:
os.environ['JAX_PLATFORMS'] = 'cpu'

# In[3]:


import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
import scipy
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn import svm
import scipy.misc

import graphlearning as gl

import matplotlib.pyplot as plt


# In[4]:


from sklearn import linear_model as linear
from sklearn import metrics as metrics


# In[5]:


from torchvision import datasets, transforms


# In[9]:


import sys


# In[10]:


sys.path.append('../')


# In[11]:


import classifier


# In[98]:


import pickle


# In[99]:


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# In[93]:


objs = []


# In[15]:


fmnist = datasets.FashionMNIST("/home/taanish/FashionMNIST", train=True, download=False)


# In[28]:


X = fmnist.data.numpy()


# In[32]:


X = np.reshape(X, (60000, 784))

# In[33]:


labels = fmnist.targets.numpy()[:60000]


# In[20]:


# 10000 labels, 10% labeled


# In[39]:


cl = classifier.Classifier(X, labels)


# In[22]:


# first experiment: average scores


# In[42]:


gs, os = classifier.ClassificationExperiments.run(tester=cl, n=10000, m=1000, trials=3)


# In[43]:


gs_101 = gs


# In[44]:


os_101 = os


# In[94]:


objs.append(gs_101)
objs.append(os_101)


# In[45]:


gs_101


# In[47]:


os_101


# In[59]:


# second experiment: average scores, varying eigenvectors


# In[49]:


steps = [10, 15, 20]


# In[52]:


gsve_101, osve_101 = classifier.ClassificationExperiments.vary_eigenvectors(tester=cl, n=10000, m=1000, trials=3, steps=steps)


# In[95]:


objs.append(gsve_101)
objs.append(osve_101)


# In[81]:


plt.scatter(steps, [np.mean(s) for s in osve_101])
plt.savefig("fm_osve_101.png")
plt.close()


# In[80]:


plt.scatter(steps, [np.mean(s) for s in gsve_101])
plt.savefig("fm_gsve_101.png")
plt.close()


# In[60]:


# third experiment: average scores, varying nearest neighbors


# In[55]:


nn_steps = [100, 150, 200]


# In[62]:


gsnn_101, osnn_101 = classifier.ClassificationExperiments.vary_neighbors(tester=cl, n=10000, m=1000, trials=3, steps=nn_steps)


# In[96]:


objs.append(gsnn_101)
objs.append(osnn_101)


# In[82]:


plt.scatter(nn_steps, [np.mean(s) for s in osnn_101])
plt.savefig("fm_osnn_101.png")
plt.close()


# In[83]:


plt.scatter(nn_steps, [np.mean(s) for s in gsnn_101])
plt.savefig("fm_gsnn_101.png")
plt.close()


# In[ ]:


# 10000 labels, 20% labeled


# In[ ]:


cl_102 = classifier.Classifier(X, labels)


# In[ ]:


# first experiment: average scores


# In[ ]:


gs_102, os_102 = classifier.ClassificationExperiments.run(tester=cl_102, n=10000, m=2000, trials=3)


# In[ ]:


objs.append(gs_102)
objs.append(os_102)


# In[ ]:


# second experiment: average scores, varying eigenvectors


# In[ ]:


gsve_102, osve_102 = classifier.ClassificationExperiments.vary_eigenvectors(tester=cl_102, n=10000, m=2000, trials=3, steps=steps)


# In[ ]:


objs.append(gsve_102)
objs.append(osve_102)


# In[ ]:


plt.scatter(steps, [np.mean(s) for s in osve_102])
plt.savefig("fm_osve_102.png")
plt.close()


# In[ ]:


plt.scatter(steps, [np.mean(s) for s in gsve_102])
plt.savefig("fm_osve_102.png")
plt.close()


# In[ ]:


# third experiment: average scores, varying nearest neighbors


# In[ ]:


gsnn_102, osnn_102 = classifier.ClassificationExperiments.vary_neighbors(tester=cl_102, n=10000, m=2000, trials=3, steps=nn_steps)


# In[ ]:


objs.append(gsnn_102)
objs.append(osnn_102)


# In[ ]:


plt.scatter(nn_steps, [np.mean(s) for s in osnn_102])
plt.savefig("fm_osnn_102.png")
plt.close()


# In[ ]:


plt.scatter(nn_steps, [np.mean(s) for s in gsnn_102])
plt.savefig("fm_gsnn_102.png")
plt.close()


# In[ ]:


# 20000 labels, 5% labeled


cl_205 = classifier.Classifier(X, labels)


# In[ ]:


# first experiment: average scores


# In[ ]:


gs_205, os_205 = classifier.ClassificationExperiments.run(tester=cl_205, n=20000, m=5000, trials=3)


# In[ ]:


objs.append(gs_205)
objs.append(os_205)


# In[ ]:


# second experiment: average scores, varying eigenvectors


# In[ ]:


gsve_205, osve_205 = classifier.ClassificationExperiments.vary_eigenvectors(tester=cl_205, n=20000, m=5000, trials=3, steps=steps)


# In[ ]:


objs.append(gsve_205)
objs.append(osve_205)


# In[ ]:


plt.scatter(steps, [np.mean(s) for s in osve_205])
plt.savefig("fm_osve_205.png")
plt.close()


# In[ ]:


plt.scatter(steps, [np.mean(s) for s in gsve_205])
plt.savefig("fm_osve_205.png")
plt.close()


# In[ ]:


# third experiment: average scores, varying nearest neighbors


# In[ ]:


gsnn_205, osnn_205 = classifier.ClassificationExperiments.vary_neighbors(tester=cl_205, n=20000, m=5000, trials=3, steps=nn_steps)


# In[ ]:


objs.append(gsnn_205)
objs.append(osnn_205)


# In[ ]:


plt.scatter(nn_steps, [np.mean(s) for s in osnn_205])
plt.savefig("fm_osnn_205.png")
plt.close()


# In[ ]:


plt.scatter(nn_steps, [np.mean(s) for s in gsnn_205])
plt.savefig("fm_gsnn_205.png")
plt.close()


# In[ ]:


# 50000 labels, 10% labeled



cl_510 = classifier.Classifier(X, labels)


# In[ ]:


# first experiment: average scores


# In[ ]:


gs_510, os_510 = classifier.ClassificationExperiments.run(tester=cl_510, n=50000, m=2000, trials=3)


# In[ ]:


objs.append(gs_510)
objs.append(os_510)


# In[ ]:


# second experiment: average scores, varying eigenvectors


# In[ ]:


gsve_510, osve_510 = classifier.ClassificationExperiments.vary_eigenvectors(tester=cl_510, n=50000, m=2000, trials=3, steps=steps)


# In[ ]:


objs.append(gsve_510)
objs.append(osve_510)


# In[ ]:


plt.scatter(steps, [np.mean(s) for s in osve_510])
plt.savefig("fm_osve_510.png")
plt.close()


# In[ ]:


plt.scatter(steps, [np.mean(s) for s in gsve_510])
plt.savefig("fm_gsve_510.png")
plt.close()


# In[ ]:


# third experiment: average scores, varying nearest neighbors


# In[ ]:


gsnn_510, osnn_510 = classifier.ClassificationExperiments.vary_neighbors(tester=cl_510, n=50000, m=2000, trials=3, steps=nn_steps)


# In[ ]:


objs.append(gsnn_510)
objs.append(osnn_510)


# In[ ]:


plt.scatter(nn_steps, [np.mean(s) for s in osnn_510])
plt.savefig("fm_osnn_510.png")
plt.close()


# In[ ]:


plt.scatter(nn_steps, [np.mean(s) for s in gsnn_510])
plt.savefig("fm_gsnn_510.png")
plt.close()


# In[ ]:


save_object(objs, "fm_objects")

