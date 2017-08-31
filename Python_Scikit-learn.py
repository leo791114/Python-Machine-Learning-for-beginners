
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
# <h5>train_test_split(*array, **options):</h5>
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
#                 <li><b>random_state:</b> int, RandomState instance or None, optional(default=None)</li>
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
# <h5>It provides functionality for unsupervised and supervised neighbors-based learning methods.</h5>
# <p>
#     <b>Unsupervised nearest neighbors:</b> It's the foundation of many other learning methods, notably        manifold learning and spectral clustering.
# </p>
# <p>
#     <b>Supervised neighbors-based learning:</b> It comes in two flavors: <em style='color: red'>classification</em> for data with discrete labels, and <em style='color: red'>regression</em> for data with continuous labels.
# </p>

# In[ ]:

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

