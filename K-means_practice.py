#%%
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import numpy as np
import pandas as pd

#%%

data_csv = pd.read_csv('data1024.csv')
data_col = data_csv.columns[0].split('\t')
data_val = data_csv.values.tolist()
df_val = []

#%%

# Split df_val element
for ele in data_val:
    idx = data_val.index(ele)
    df_val.append(ele[0].split('\t'))

print(df_val)

#%%
# Create DataFrame
df = pd.DataFrame(df_val, columns=data_col)

print(type(df))
print(df.columns)
print(df.index)

#%%
# Draw scatter plot of the dataframe of input

fig1 = plt.figure(1)  # create a figure instance
ax = fig1.gca()  # get the current axes
ax.grid(color='c', linestyle=':')  # draw the grid
ax.set_axisbelow(True)  # put the grid behind the graph
ax.set_axis_bgcolor('whitesmoke')
plt.scatter(df[df.columns[1]], df[df.columns[2]], c='black')
plt.xlabel(df.columns[1])
plt.ylabel(df.columns[2])
plt.show()

#%%
# Choose K and run the algorithm
f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

X = zip(f1, f2)
print(X)

#%%
# Converting iterator to list
# X_list = list(X)
# print(X_list)

#%%
# Converting iterator to set
# X_set = set(X)
# print(X_set)

#%%
#Converting iterator to matrix
X_matrix = np.matrix(X)
print(X_matrix)
print(type(X_matrix))

#%%
# Converting iterator to Numpy matrix
# X_matrix = np.matrix(X)
kmeans = KMeans(n_clusters=2).fit(X_list)
kmeans
kmeans.labels_
kmeans.cluster_centers_


#Reference
# https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
# http://blog.blackwhite.tw/2013/05/python-yield-generator.html
# https://stackoverflow.com/questions/31683959/the-zip-function-in-python-3
# https://stackoverflow.com/questions/14637154/performing-len-on-list-of-a-zip-object-clears-zip
