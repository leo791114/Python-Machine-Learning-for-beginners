
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
# <h4 style='color: red'>Confusing KMeans parameter:</h4>
# 
# <table align='left' style='margin-bottom: 10px'>
#     <tr style='border: 1px solid black'>
#         <th style='text-align: left; border-right: 1px solid black'>max_iter</th>
#         <td style='text-align: left'>
#            Maximum number of iterations of the K-means algorithm for a single run. (int, default: 300)
#         </td>
#     </tr>
#     <tr style='border: 1px solid black'>
#         <th style='text-align: left; border-right: 1px solid black'>n_init</th>
#         <td style='text-align: left'>
#            Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. (int, default: 10)
#         </td>
#     </tr>
# </table>
# 
# <h5>Example:</h5>
# <p>With <mark>max_inter=300</mark> and <mark>n_init=15</mark>, kmeans will choose initial centroids 15 times, and each run will use up to 300 iterations. The best out of those 10 runs will be the final result.</p>
# <p>The centroids are chosen by weighted probability where the probability is propotional to <mark>D(x)^2</mark>  (the distance between new dat a point which is the candidate of new centroid and the nearest centroid that has already been chosen, k-means++)</p>
#     
# <h5>Reference:</h5>
# <ol>
#     <li><a href='http://mnemstudio.org/clustering-k-means-example-1.htm'>k-Means: Step-By-Step</li>
#     <li><a href='https://www.datascience.com/blog/introduction-to-k-means-clustering-algorithm-learn-data-science-tutorials'>Introduction to K-means Clusting</a></li>
#     <li><a href='https://www.naftaliharris.com/blog/visualizing-k-means-clustering/'>Visualizing K-Means Clustering</a></li>
#     <li><a href='https://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work'>How exactly does k-means++ work?</a></li>
#     <li><a href='https://stats.stackexchange.com/questions/246061/what-are-the-advantages-of-the-pre-defined-initial-centroids-in-clustering'>What are the advantages of the pre-defined initial centroids in clustering?</a></li>
#     <li><a href='https://stackoverflow.com/questions/40895697/sklearn-kmeans-parameter-confusion'>Sklearn Kmeans paremeter confusion?</a></li>
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

# In[2]:

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


# In[3]:

# Load data file as pandas dataframe
df = pd.read_csv('player_traditional.csv')
X = df.iloc[:,2:]
print(X)


# In[4]:

# Standardize
sc = StandardScaler()
sc.fit(X)

# X's mean values of each feature
X_mean = X.mean(axis = 0)
print(sc.mean_)
print(X_mean)

# X's variances of each feature which are used to compute scale_
print(sc.var_)

# X's standard deviations of each feature
X_std = X.std(axis = 0)
print(sc.scale_)
print(X_std)

# Transform X 
X_train_std = sc.transform(X)
print(X_train_std)


# In[19]:

#Normal Kmeans method

km_norm = KMeans(n_clusters=3, init='random', max_iter=300, tol=1e-04, random_state=0)
y_km = km_norm.fit(X_train_std)
y_km.predict(X_train_std)
y_2_km = km_norm.fit_predict(X_train_std)
print(y_km.labels_)
print(y_2_km)


# In[25]:

# k-means++ method
km_pp = KMeans(n_clusters=3, init='k-means++',n_init=10, max_iter= 300, tol=1e-04)
y_km_pp = km_pp.fit(X_train_std)
y_km_pp.predict(X_train_std)
y_km_pp_2 = km_pp.fit_predict(X_train_std)

print(y_km_pp.labels_)
print(y_km_pp_2)


# In[ ]:



