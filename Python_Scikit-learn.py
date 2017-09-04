
# coding: utf-8

# <h3 style="color: blue">KNN Classifier - IRIS</h3>
# <h4 style="color: red">Processes:</h4>
# <ol>
#     <li>
#         <b>Import module:</b> 
#         <ul><li>datasets, train_test_split, KNeighborsClassifier</li></ul>
#     </li>
#     <li>
#         <b>Create Data:</b>
#         <ul>
#             <li>classes, labels</li>
#             <li>Data set --> training set/data, testing set/data</li>
#         </ul>
#     </li>
#     <li>
#         <b>Create Model:</b>
#         <ul>
#             <li>fit, predict</li>
#         </ul>
#     </li>
#     <li>
#         <b>Visualization</b>
#     </li>
# </ol>

# <h4 style="color: green">sklear.cross_validation:</b></h4>
# <h5><li>train_test_split(*array, **options):</li></h5>
# <p>To avoid <b>overfitting</b>, it is common practice when performing a (supervised) machine learning experiment to hold out
# part of the available data as a <b>test set</b> <em>X_test, y_test</em>.
# <table align='left'>
#     <tr style='border: 2px solid black'>
#         <th style='text-align: left; border-right: 2px solid black'>Parameters:</th>
#         <td>
#             <ul style='list-style:none; text-align: left; line-height: 2rem'>
#                 <li><b>*arrays:</b> Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes</li>
#                 <li><b>test_size:</b> float, int, None, optional</li>
#                 <li><b>train_size:</b> float, int, None, default None</li>
#                 <li>
#                    <b>random_state:</b> int, RandomState instance or None, optional(default=None)                     </li>
#                 <li>
#                     <b>shuffle:</b> boolean, optional(default=True)
#                     <p style='margin-top: 0; text-indent: 40px'>Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.</p>
#                 </li>
#                 <li><b>stratify:</b> array-like or None (default is None)</li>
#             </ul>
#         </td>
#     </tr>
#     <tr style='border: 2px solid black'>
#         <th style='text-align: left; border-right: 2px solid black'>Returns:</th>
#         <td>
#             <ul style="list-style: none; text-align: left; line-height: 2rem">
#                 <li>
#                     <b>splitting:</b> list, length=2 * len(arrays)
#                     <p style='margin-top: 0; text-indent: 40px'>List containing train-test split of inputs</p>
#                 </li>
#             </ul>
#         </td>
#     </tr>
# </table>

# <h4 style='color: green'>sklearn.neighbors:</h4>
# <h5>&emsp;It provides functionality for unsupervised and supervised neighbors-based learning methods.</h5>
# <p>
#     &emsp;(The principle behind nearest neighbor methods is to find a predefined number of training       samples closest in distance to the new point, and predict the label from these.)
# </p>
# <p>
#     &emsp;<b>Unsupervised nearest neighbors:</b> It's the foundation of many other learning methods,       notably manifold learning and spectral clustering.
# </p>
# <p>
#     &emsp;<b>Supervised neighbors-based learning:</b> It comes in two flavors: <em style='color:      red'>classification</em> for data with discrete labels, and <em style='color: red'>regression</em> for data with continuous labels.
# </p>

# <li>
#     <em>sklearn.neighbors.<b>KNeighborsClassifier</b></em>(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,, metric='minkowski', metric_params=None, n_jobs=1, **kwargs):
# </li>
# <p>Classifier implementing the k-nearest neighbors vote.</p>
# 
# <table align='left'>
#     <tr>
#         <th style='border: solid 2px black'>Parameters:</th>
#         <td style='text-align: left; border: solid 2px black'>
#             <ul style='list-style: none; line-height: 2.5rem'>
#                 <li>
#                     <b>n_neighbors:</b> int, optional (default=5)
#                     <br>
#                     &nbsp; Number of neighbors to use by default for <em>kneighbors</em> queries.
#                 </li>
#                 <li>
#                     <b>weights:</b> str or callable, optional (default = 'uniform')
#                     <br>
#                     &nbsp;
#                     weight function in prediciton.
#                     <ul>
#                         <li>'uniform'</li>
#                         <li>'distance'</li>
#                         <li>[callable]</li>
#                     </ul>
#                 </li>
#                 <li>
#                     <b>algorithm:</b> {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
#                     <br>
#                     &nbsp; 
#                     Algorithm used to compute the nearest neighbors:
#                     <ul>
#                         <li>'ball_tree' will use <em>BallTree</em></li>
#                         <li>'kd_tree' will use <em>KDTree</em></li>
#                         <li>'brute' will use a brute-force search</li>
#                         <li>'auto' will attempt to decide the most appropriate algorithm</li>
#                     </ul>
#                 </li>
#                 <li>
#                     <b>leaf_size:</b> int, optional (default = 30)
#                     <br>
#                     &nbsp;
#                     Leaf size passed to BallTree or KDTree. This can affect the speed of the 
#                     construction and query, as well as the memory required to store the tree.
#                 </li>
#                 <li>
#                     <b>p:</b> string or callable, default 'minkowski'
#                     &nbsp;
#                     Power parameter for the Minkowski metric
#                     <ul>
#                         <li>p = 1: manhattan_distance(l1)</li>
#                         <li>p = 2: euclidean_distance(l2)</li>
#                         <li>p = arbitrary: minkowski_distance(l_p)</li>
#                     </ul>
#                 </li>
#                 <li>
#                     <b>metric:</b> string or callable, default 'minkowski'
#                     &nbsp;
#                     The distance metric to use for the tree
#                 </li>
#                 <li>
#                     <b>metric_params:</b> dict, optional (default = None)
#                     &nbsp;
#                     Additional keyword arguments for the metric function
#                 </li>
#                 <li>
#                     <b>n_jobs:</b> int, optional (default = 1)
#                     &nbsp;
#                     The number of parallel jobs to run for neighbors search
#                 </li>
#             </ul>  
#         </td>
#     </tr>
# </table>

# <h4 style='color:green'>sklearn.prepocessing:</h4>
# <h5>
#     &emsp;It provides several common utility functions and transformer classes to change raw feature vectors
#     into a representation that is more suitable for the downstream estimators.
# </h5>
# <p>
#     &emsp;Learning algorithms benefit from standardization of the data set. (ex: dealing with outliers)
# </p>
# 

# <li>sklearn.preprocessing.<b>StandardScaler</b>(copy=True, with_mean=True, with_std=True)</li>
# <p>&emsp;Standardize features by removing the mean and scalling to unit variance. (normally distributed data, ex: <b style='color: red'>Gaussian with 0 mean and unit variance</b>,(X-X_mean)/X_std)</p>
# <p>&emsp;Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the transformer method.</p>
# 
# <table align='left'>
#     <tr style='border: solid 2px black'>
#         <th style='border: solid 2px black'>Parameters:</th>
#         <td>
#             <ul style='list-style:none; text-align: left; line-height: 2.5rem'>
#                 <li>
#                     <b>copy:</b> boolean, optional, default True
#                 </li>
#                 <li>
#                     <b>with_mean:</b> boolean, default True
#                     <br>
#                     &nbsp; 
#                     If True, center the data before scaling.
#                 </li>
#                 <li>
#                     <b>with_std:</b> boolean, default True
#                     <br>
#                     &nbsp; 
#                     If True, scale the data to unit variance (or equivalently, unit standard
#                     deviation)
#                 </li>
#             </ul>
#         </td>
#     </tr>
#     <tr style='border: solid 2px black'>
#         <th style='border: solid 2px black'>Attributes</th>
#         <td>
#             <ul style='list-style: none; text-align: left; line-height: 2.5rem'>
#                 <li>
#                     <b>scale<u style='white-space: pre'>  </u>:</b> ndarray, shape[n<u> </u>features]
#                     <br>
#                     &nbsp;
#                     Per feature relative scaling of the data
#                 </li>
#                 <li>
#                     <b>mean<u style='white-space: pre'>  </u>:</b> array of floats with shape [n<u> </u>features]
#                     <br>
#                     &nbsp;
#                     The mean value for each feature in the training set.
#                 </li>
#                 <li>
#                     <b>var<u style='white-space: pre'>  </u>:</b> array of floats with shape [n<u> </u>features]
#                     <br>
#                     &nbsp;
#                     The variance for each feature in the training set
#                 </li>
#                 <li>
#                     <b>n_samples_seen:</b> int
#                     <br>
#                     &nbsp;
#                     The number of samples processed by the estimator.
#                 </li>
#             </ul>
#         </td>
#     </tr>
# </table>

# In[1]:

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[7]:

iris = datasets.load_iris()
# print(iris)
print(iris.target_names)
print(iris.feature_names)
iris_X = iris.data
iris_y = iris.target
print(len(iris_X))
print(len(iris_y))


# In[16]:

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
print(len(X_train))
# print(X_train)
print(len(y_train))


# In[28]:

#Standardize
sc = StandardScaler()
sc.fit(X_train)

# X_train's mean values of each feature
X_train_mean = X_train.mean(axis=0)
print(X_train_mean)
print(sc.mean_)

# X_train's variance for each feature
print(sc.var_)

# X_train's standard deviation of each feature
X_train_std = X_train.std(axis=0)
print(X_train_std)
print(sc.scale_)

# transform X_train and X_test
X_train_standard = sc.transform(X_train)
X_test_standard = sc.transform(X_test)


# In[36]:

knn = KNeighborsClassifier()
# Fit the module using X as training data and y as target values
knn.fit(X_train_standard, y_train)
print(knn.fit(X_train_standard, y_train))

# Predict the class labels for the provided data
y_predict = knn.predict(X_test_standard)
print(len(y_predict))
print(y_predict)

#compare y_predict and y_test
print(y_test)
print(y_test!=y_predict)
print('Misclassified samples: %d'%(y_test!=y_predict).sum())


# In[38]:

# Visualization
plt.scatter(y_predict, y_test, alpha=0.2)
plt.show()


# In[67]:

index_diff = np.where(y_test!=y_predict)
array_diff = np.array([])
print(type(index_diff))
print(len(index_diff))
print(type(index_diff[0]))

for i in index_diff[0]:
    print(y_test[i], y_predict[i])
    np.append(array_diff, [[y_test[i], y_predict[i]]])
    print(array_diff)
    
# for i in np

