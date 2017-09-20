
# coding: utf-8

# Hui-Chun Hung, 2017

# # Python Machine Learning Essentials

# # Grouping objects by similarity using k-means

# In[1]:

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


# In[2]:

df = pd.read_csv('player_traditional.csv')
X = df.iloc[:,2:].values


# In[3]:

#標準化

sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X)


# In[4]:

X_train_std


# In[5]:

#使用一般的KMean
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X_train_std)
print(km.labels_)


# In[6]:

#使用k-means++

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0).fit(X_train_std)

# 印出分群結果
print("分群結果：")
print(km.labels_)
print("----------我是分隔線----------")



# In[7]:

new_df = pd.read_csv('player_traditional2.csv')


# In[9]:

#Performing hierarchical clustering on a distance matrix
row_dist = pd.DataFrame(squareform(pdist(new_df, metric='euclidean')))
row_dist


# In[10]:

# correct approach: Condensed distance matrix
row_clusters = linkage(pdist(new_df, metric='euclidean'), method='complete')
pd.DataFrame(row_clusters, 
             columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
             index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])


# In[11]:

row_clusters


# In[12]:

f = open('player.txt','r')
label = f.read().split('\n')


# In[15]:

row_dendr = dendrogram(row_clusters, 
                       labels=df.index)
plt.tight_layout()
plt.ylabel('Euclidean distance')
#plt.savefig('dendrogram.png', dpi=3000, 
#            bbox_inches='tight')
plt.show()


# In[ ]:

#Attaching dendrograms to a heat map
fig = plt.figure(figsize=(100,100))
axd = fig.add_axes([0.07,0.1,0.2,0.6]) #設定X Y 軸 位置 高度 寬度
row_dendr = dendrogram(row_clusters, orientation='left') #right 將樹狀圖以逆時針轉90度

#dendrogram樹狀圖物件本身是個字典的leaves鍵，藉此可以讀到集群標籤
df_rowclust = new_df.ix[row_dendr['leaves'][::-1]]

axd.set_xticks([]) 
axd.set_yticks([])
#移除掉軸的刻度以便修改熱度圖的顏色
# remove axes spines from dendrogram
for i in axd.spines.values():
        i.set_visible(False)
        
# plot heatmap

axm = fig.add_axes([0.1,0.1,0.27,0.6]) # x-pos, y-pos, width, height
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r') #將重新排列的dataframe把入熱度圖
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_xticks(np.linspace(0,22,23))
axm.set_yticklabels(label)
axm.set_yticks(np.linspace(0,485,486))
# plt.savefig('./figures/heatmap.png', dpi=300)
plt.xticks(rotation=90)
plt.show()
#plt.savefig('./figures/dendrogram.png', dpi=300, 
#            bbox_inches='tight')


# In[ ]:



