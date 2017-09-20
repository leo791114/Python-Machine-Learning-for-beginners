
# coding: utf-8

# <h3 style='color: blue'>K-means clustering: NBA player</h3>
# <p> &emsp; K-means clustering is a type of unsupervised learning, which is used when we have unlabeled
# data. The goal of this algorithm is to find groups in the data, with the number of groups represented by the variabled K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.</p>
# 
# <h4 style='color: red'>Processes:</h4>
# <ol>
#     <li>Randomly pick k centroids(sensible initial partition) from the samples</li>
#     <li>Assign remaining individuals (samples) to the centroid (cluster) which they were <em>closest</em> to (by Euclidean distance)</li>
#     <li>Recalculate the centroid to the <em>mean value</em> of the values of all samples in the cluster.</li>
#     <li>Repeat process 2 and 3 until there are no more relocations, or reaches the tolerance or maximum of iterations that is pre-chosen by the user.</li>
# </ol>
# 
# <h4 style='color: red'>Results:</h4>
# <ol>
#     <li>The centroids of the K clusters, which can be used to label data</li>
#     <li>Labels for the training data (each data point is assigned to a single cluster)</li>
# </ol>
# 
# <h5>Reference:</h5>
# <ol>
#     <li><a href='http://mnemstudio.org/clustering-k-means-example-1.htm'>k-Means: Step-By-Step</li>
#     <li><a href='https://www.datascience.com/blog/introduction-to-k-means-clustering-algorithm-learn-data-science-tutorials'>Introduction to K-means Clusting</a></li>
# </ol>

# <h4 style='color: red'>sklearn.metrics:</h4>
# <p>The <em>sklearn.metrics</em> module includes score functions, performance metrics and pairwise
# metrics and distance computations</p>
# 
# <p style='color: blue'>sklearn.metrics.<b>silhouette_samples</b>(X, labels, metric='euclidean', **kwds)</p>
# <p>Compute the Silhouette Coefficient for each sample.</p>
#     
# <p>The Silhouette Coefficient is a measure of how well samples are clustered with the samples  that are similar to themselves.</p>
#     
# <p>Clustering models with high Silhouette Coefficient are said to be dense, where samples in the same cluster are similiar to each other, and well separated, where samples in different clusters are  not very similar to each other.</p>
# 
# <p>The Silhouette Coefficient is calculated using the mean intra-cluster distance (<mark>a</mark>) and the mean nearest-cluster distance (<mark>b</mark>) for each sample. The Silhouette Coefficient for a sample is <mark>(b-a)/max(a,b)</mark>.

# In[1]:

# import packages
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram


# In[10]:

df = pd.read_csv('player_traditional.csv')
# print(df.columns)
# print(df.index)
# print(df)
X = df.iloc[:,2:]
print(X)

