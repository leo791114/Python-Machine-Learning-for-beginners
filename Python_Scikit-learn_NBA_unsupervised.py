
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

# In[ ]:



