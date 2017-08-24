
# coding: utf-8

#  <h1 style="color: red">Numpy for Beginners</h1>

# In[1]:

import numpy as np  #Import numpy


# In[2]:

array = np.array([[1,2,3],[4,5,6]])  #Create a two-dimensional array


# In[3]:

print(array)
print('Dim', array.ndim)
print('Shape', array.shape)
print('Size', array.size)


# In[4]:

a = np.array([1,2,3,4,5])
b = np.arange(2,14,2).reshape((2,3))#Create an array from two to twelve, not included, with step of two
                                    #.reshape() reshape the array for 1*6 to 2*3
print(a)
print(b)
c = np.zeros(6) #Create an array with six floating 0.
print('\n')
print(c)
print(c.dtype)
d = np.ones(6, dtype = np.int) #Create an array with six integer 1.
print('\n')
print(d)
print(d.dtype)
e = np.random.random((2,3)) #Create an 2*3 array with elements that are randomly picked between 0 to 1.
print('\n')
print(e)
#f = b-a
print('\n')
# print(f)

print(a,b,c,d,e,sep='\n')
# print(a,b,c,d,e,f, sep='\n')
#print(f<-9)
#print(f==5)


# In[5]:

a = np.array([[3,6,9],[12,15,18]])
print(a)
print(np.sum(a))
print(np.sum(a, axis=1)) #add all elements on the basis of row
print(np.sum(a, axis=0)) #add all elements on the basis of column


# In[6]:

a = np.array([[2,4,6],[8,10,12]])
b = np.arange(6).reshape((2,3))
print(a,b,a-b,a*b, sep='\n') #* is normal multiplication
print('\n')
print(b.T) #transpose
print(np.dot(a,b.T)) # This is array multiplication


# In[7]:

#Merge
A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B))) #vertical merge
print(np.hstack((A,B))) #horizontal merge


# In[8]:

#Split
a = np.arange(12).reshape((3,4))
print(a)
print(np.vsplit(a,3)) #Vertically split the array into 3 new arrays
print(np.hsplit(a,2)) #Horizontally split the array into 2 new arrays


# <h1 style="color: red">Pandas for Beginners</h1>

# <h3 style="color:blue">Series</h3>

# In[26]:

import pandas as pd
s = pd.Series([1,'abc','6',np.nan,44,1])
a = np.array([1,'abc','6',np.nan,44,1])
print(t)
print(s)
print(a)


#  <h3 style="color: blue">DataFrame</h3> 

# In[10]:

#Method one: use array to create DataFrame
print(np.random.random()) # np.random.random returns random floats in the half-open interval [0 1)
print(np.random.randn()) # np.random.randn returns random floats sampled from a univariate nomal distribution


df = pd.DataFrame(np.random.randn(7,3))
print(df)


# In[11]:

#Method one: use array to create DataFrame
eat = np.random.randint(10, size=(7,3))*5+50
print(eat)
dates = pd.date_range('20170812', periods=7)
print(dates)
# print(np.dtype(dates))
df0 = pd.DataFrame(eat)
print(df0)
df1 = pd.DataFrame(eat, index=dates, columns=['Breakfast', 'Lunch', 'Dinner'])
print(df1)


# In[12]:

#Method two: use dictionary to create DataFrame
df2 = pd.DataFrame({'Decimal': pd.Series([1,3,6,4], index=list(range(4)), dtype='float32'),
                   'Integer': np.array([3]*4, dtype='int32'),
                   'Time': pd.Timestamp('20170812'),
                   'Category': pd.Categorical(['test','train','test','train'])})
x = pd.Series(1,index=list(range(4)), dtype='float32')
# print(x)
y = np.array([3]*4, dtype='int32')
#print(y)
z = pd.Timestamp('20170816')
#print(z)
k = pd.Categorical(['test','train','test','train'])
# print(k)
print(df2)
#print(df2.dtypes)
#print(df2.index)
#print(df2.columns)
#print(df2.values)
#print(df2.T)
print(df2.sort_index(axis=1,ascending=False)) #axis=1 takes columns as sorting index
print(df2.sort_values(by='Decimal'))


# In[13]:

# Pandas: Selecting data
print(df1)
print('\n')
print(df1['Lunch'])
print('\n')
print(df1[0:3])


# In[14]:

print(df1)
# Pandas: Selecting data by label (refer to the label and not the position)
print(df1.loc['20170812'])
print(df1.loc['20170813':'20170816',['Breakfast','Lunch']])
print('\n')
# Pandas: Selecting data by position
print(df1.iloc[3,1])
print(df1.iloc[3:5,1:3])
print('\n')
# Pandas: Selecting data by ix
print(df1.ix[:3,['Lunch','Dinner']])


# In[15]:

# Pandas: Conditional expression
print(df1[df1.Lunch > 80])


# In[16]:

# Pandas: Re-assign data
#df1.Dinner[df1.Lunch>80]=40  --> this method Pandas would select df1.Dinner first, and then returns
#a DataFrame that is singly-indexed. Then, another python operation df1_dinner[df1.Lunch>80] will
#happens. THis is because pandas sees these operations as separate events.
df1.loc[:,['Dinner']][df1.Lunch>80] = 40
print(df1[df1.Lunch>80])
df1.loc['20170814','Lunch'] = np.nan
print(df1)
print(df1.dtypes)


# In[17]:

# Pandas: Dealing with missing values
print(df1.isnull())
df1.isnull().sum()


# In[18]:

# Panadas: Dealing with missing values --> replace them with 0
df1.fillna(value=0)


# In[19]:

# Panadas: Dealing with missing values --> erase the missing values
print(df1)
df1.dropna(
    axis=0,    #0: operates under the base of row; 1: operates under the base of column
    how='any'  #'any':Drop the whole row or column if there is any NaN value.
               #'all':Drop the whole row or column if the values in the row or column are all NaN.
)


# In[20]:

# Pandas: Other usages
print(df1)
df1.dropna(thresh=3)  #Keep only the rows with at least 3 non-na values
df1.dropna(subset=['Breakfast']) #Only drops the rows or columns in the subset that have NaN value.


# In[21]:

# Pandas: Dealing with missing values --> interpolation
from sklearn.preprocessing import Imputer
# print(df1.dtypes)
print(df1.values)
print(df1.values.dtype)
# print(df1.values.dtypes)
imr = Imputer(missing_values='NaN', strategy='mean', axis=0) #axis = 0 for columns, axis = 1 for rows
imr_2 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df1)
imputed_data = imr.transform(df1.values)
imputed_data_2 = imr_2.fit_transform(df1) #fit + transform
print(imputed_data)
print(imputed_data_2)


# In[22]:

# Pandas: load file

#load csv file
data = pd.read_csv('nescsv2.csv')
print(data)
data = data.T
print(data)
data.to_csv('nescsv2_transpose.csv')


# In[23]:

# Pandas merge DataFrame

#concat

df1_merge = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2_merge = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3_merge = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
# print(df1_merge, df2_merge, df3_merge,sep='\n')
res = pd.concat([df1_merge, df2_merge, df3_merge], axis = 0, ignore_index = True)
# print('\n')
# print(res)
# print('\n')
print(df1_merge.columns)
print(df1_merge.index)
#join: {'inner', 'outer'}
df4_merge = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'],index=[1,2,3])
df5_merge = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'],index=[2,3,4])
res_outer = pd.concat([df4_merge, df5_merge], axis=0, join='outer', ignore_index=False)
res_inner = pd.concat([df4_merge, df5_merge], axis=0, join='inner', ignore_index=False)
print(res_outer)
res_reset = res_inner.reset_index(drop=True)
print(list(res_reset)) # Same as print(list(res_reset))
print(res_reset)
print(res_reset.groupby(['b','c']))
# print(res_outer, res_inner, sep='\n')
# print(diff_inner_outer)


# In[24]:

df1_test = pd.DataFrame({
    'Date': pd.Timestamp('20170812'),
    'Fruit': ['Banana', 'Orange', 'Apple', 'Celery'],
    'Num': [22.1,8.6,7.6,10.2],
    'Color': ['Yello', 'Orange', 'Green', 'Green']
})

df2_test = pd.DataFrame({
    'Fruit': ['Banana', 'Orange', 'Apple', 'Celery', 'Apple', 'Orange'],
    'Num': [22.1, 8.6, 7.6, 10.2, 22.1, 8.6],
    'Color': ['Yellow','Orange','Green','Green','Red','Orange']
})

dates_test = []
for i in range(1,7):
    if i <=4:
        dates_test.append(pd.Timestamp('20170812'))
    else:
        dates_test.append(pd.Timestamp('20170813')) 

df2_test.insert(0, 'Date', dates_test)
# print(dates_test)
print(df2_test)
print(df1_test)

df_merge_test = pd.concat([df1_test, df2_test])
print(df_merge_test)

df_merge_test = df_merge_test.reset_index(drop = True) #pd.concat([df1_test, df2_test], ignore_index=True) could get the same effect
print(df_merge_test)

print (list(df_merge_test)) #Same as list(df_merge_test.columns) 
df_gpby = df_merge_test.groupby(list(df_merge_test))
# print(df_gpby.groups)
print(df_gpby.groups.values())
# print(df_gpby.count())

# for x in df_gpby.groups.values():
#     print(x)
idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
print(idx)

df_merge_test.reindex(idx)


# 
