#%%
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

#%%

data_csv = pd.read_csv('data1024.csv')
df_col = data_csv.columns[0].split('\t')

print(data_csv.values.tolist())
print(type(data_csv.values.tolist()))
print(df_col)

# print(data_csv)
# print(type(data_csv))








