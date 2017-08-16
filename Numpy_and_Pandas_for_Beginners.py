
# coding: utf-8

#  <h1 style="color: red">Numpy for Beginners</h1>

# In[2]:

import numpy as np  #Import numpy


# In[3]:

array = np.array([[1,2,3],[4,5,6]])  #Create a two-dimensional array


# In[4]:

print(array)
print('Dim', array.ndim)
print('Shape', array.shape)
print('Size', array.size)


# In[5]:

a = np.array([1,2,3,4,5])
b = np.arange(2,14,2).reshape((2,3))#Create an array from two to twelve, not included, with step of two
                                    #.reshape() reshape the array for 1*6 to 2*3
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
f = b-a
print('\n')
print(f)

print(a,b,c,d,e,f, sep='\n')
print(f<-9)
print(f==5)


# In[ ]:

a = np.array([[3,6,9],[12,15,18]])
print(a)
print(np.sum(a))
print(np.sum(a, axis=1)) #add all elements on the basis of row
print(np.sum(a, axis=0)) #add all elements on the basis of column


# In[ ]:

a = np.array([[2,4,6],[8,10,12]])
b = np.arange(6).reshape((2,3))
print(a,b,a-b,a*b, sep='\n') #* is normal multiplication
print('\n')
print(b.T) #transpose
print(np.dot(a,b.T)) # This is array multiplication


# In[ ]:

#Merge
A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B))) #vertical merge
print(np.hstack((A,B))) #horizontal merge


# In[ ]:

#Split
a = np.arange(12).reshape((3,4))
print(a)
print(np.vsplit(a,3)) #Vertically split the array into 3 new arrays
print(np.hsplit(a,2)) #Horizontally split the array into 2 new arrays


# <h1 style="color: red">Pandas for Beginners</h1>

# <h3 style="color:blue">Series</h3>

# In[25]:

import pandas as pd
s = pd.Series([1,'abc','6',np.nan,44,1])
a = np.array([1,'abc','6',np.nan,44,1])

print(s)
print(a)


#  <h3 style="color: blue">DataFrame</h3> 

# In[26]:

#Method one: use array to create DataFrame
print(np.random.random()) # np.random.random returns random floats in the half-open interval [0 1)
print(np.random.randn()) # np.random.randn returns random floats sampled from a univariate nomal distribution


df = pd.DataFrame(np.random.randn(7,3))
print(df)


# In[43]:

#Method one: use array to create DataFrame
eat = np.random.randint(10, size=(7,3))*5+50
print(eat)
dates = pd.date_range('20170812', periods=7)
print(dates)
#print(np.dtype(dates))
df0 = pd.DataFrame(eat)
print(df0)
df1 = pd.DataFrame(eat, index=dates, columns=['Breakfast', 'Lunch', 'Dinner'])
print(df1)


# In[83]:

#Method two: use dictionary to create DataFrame
df2 = pd.DataFrame({'Decimal': pd.Series([1,3,6,4], index=list(range(4)), dtype='float32'),
                   'Integer': np.array([3]*4, dtype='int32'),
                   'Time': pd.Timestamp('20170812'),
                   'Category': pd.Categorical(['test','train','test','train'])})
x = pd.Series(1,index=list(range(4)), dtype='float32')
#print(x)
y = np.array([3]*4, dtype='int32')
#print(y)
z = pd.Timestamp('20170816')
#print(z)
k = pd.Categorical(['test','train','test','train'])
#print(k)
#print(df2)
#print(df2.dtypes)
#print(df2.index)
#print(df2.columns)
#print(df2.values)
#print(df2.T)
print(df2.sort_index(axis=1,ascending=False)) #axis=1 takes columns as sorting index
print(df2.sort_values(by='Decimal'))


# In[89]:

# Pandas: Selecting data
print(df1)
print('\n')
print(df1['Lunch'])
print('\n')
print(df1[0:3])


# In[132]:

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


# In[111]:

# Pandas: Conditional expression
print(df1[df1.Lunch > 80])


# In[140]:

# Pandas: Re-assign data
#df1.Dinner[df1.Lunch>80]=40  --> this method Pandas would select df1.Dinner first, and then returns
#a DataFrame that is singly-indexed. Then, another python operation df1_dinner[df1.Lunch>80] will
#happens. THis is because pandas sees these operations as separate events.
df1.loc[:,['Dinner']][df1.Lunch>80] = 40
print(df1[df1.Lunch>80])
df1.loc['20170814','Lunch'] = np.nan
print(df1)
print(df1.dtypes)


# In[130]:

# Pandas: Dealing with missing values
print(df1.isnull())
df1.isnull().sum()


# In[131]:

# Panadas: Dealing with missing values --> replace them with 0
df1.fillna(value=0)


# In[146]:

# Panadas: Dealing with missing values --> erase the missing values
print(df1)
df1.dropna(
    axis=0,    #0: operates under the base of row; 1: operates under the base of column
    how='any'  #'any':Drop the whole row or column if there is any NaN value.
               #'all':Drop the whole row or column if the values in the row or column are all NaN.
)


# In[155]:

# Pandas: Other usages
print(df1)
df1.dropna(thresh=3)  #Keep only the rows with at least 3 non-na values
df1.dropna(subset=['Breakfast']) #Only drops the rows or columns in the subset that have NaN value.


# In[ ]:


