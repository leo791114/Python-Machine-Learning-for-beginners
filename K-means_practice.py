#%%
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

#%%
df = pd.read_csv('data_1024.csv')
print(df)
