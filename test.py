#%%
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist


#%%
X = np.array([[9, 0], [1, 4], [2, 3], [8, 5]])
print(X)
print(pdist(X))
Z = linkage(X, method='ward')
print(Z)

#%%
c, coph_dists = cophenet(Z, pdist(X))
print(c)
print(coph_dists)
