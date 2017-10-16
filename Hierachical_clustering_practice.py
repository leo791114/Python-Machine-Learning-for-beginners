#%%
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

#%%
# Suppress scientific float notation
np.set_printoptions(precision=5, suppress=True)
# https://stackoverflow.com/questions/2891790/how-to-pretty-printing-a-numpy-array-without-scientific-notation-and-with-given

#%%
# Generating two clusters: a with 100 points, b with 50 points:
np.random.seed(4711)  # for repeatability
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[
                                  100, ])  # multivariate_normal ?
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50, ])
X = np.concatenate((a, b))

print(np.mean(a, axis=0))
print(np.mean(b, axis=0))
print(X.shape)
print(type(X))
print(np.mean(X, axis=0))
print(X)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

#%%
# Generate linkage matrix
Z = linkage(X, 'ward')
print(type(Z))
print(Z.shape)
print(Z.size)
print(X[52], X[53])
print(Z)
