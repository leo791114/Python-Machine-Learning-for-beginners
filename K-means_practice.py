#%%
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.axes  as ax
import numpy as np
import pandas as pd

#%%

data_csv = pd.read_csv('data1024.csv')
data_col = data_csv.columns[0].split('\t')
data_val = data_csv.values.tolist()
df_val = []

#%%

#Split df_val element
for ele in data_val:
    idx = data_val.index(ele)
    df_val.append(ele[0].split('\t'))

print (df_val)

#%%
#Create DataFrame
df = pd.DataFrame(df_val, columns=data_col)

print(type(df))
print(df.columns)
print(df.index)

#%%
#Draw scatter plot of the dataframe of input 

fig1 = plt.figure(1)  #create a figure instance
ax = fig1.gca()  #get the current axes
ax.grid(color='c', linestyle=':')  #draw the grid 
ax.set_axisbelow(True)  #put the grid behind the graph
ax.set_axis_bgcolor('grey')
plt.scatter(df[df.columns[1]], df[df.columns[2]], c='black')
plt.xlabel(df.columns[1])
plt.ylabel(df.columns[2])
plt.show()


#%%
f = plt.figure(1)

#%%
f.axes

# print(df_val)
# print(type(data_csv.values.tolist()))
# print(df_col)








