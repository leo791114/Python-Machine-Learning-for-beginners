
# coding: utf-8

#  <h1 style="color: red">Numpy for Beginners</h1>

# In[4]:

import numpy as np  #Import numpy


# In[2]:

array = np.array([[1,2,3],[4,5,6]])  #Create a two-dimensional array


# In[3]:

print(array)
print('Dim', array.ndim)
print('Shape', array.shape)
print('Size', array.size)


# In[33]:

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
#f = b-a
print('\n')
print(f)

print(a,b,c,d,e,f, sep='\n')
print(f<-9)
print(f==5)


# In[36]:

a = np.array([[3,6,9],[12,15,18]])
print(a)
print(np.sum(a))
print(np.sum(a, axis=1)) #add all elements on the basis of row
print(np.sum(a, axis=0)) #add all elements on the basis of column


# In[44]:

a = np.array([[2,4,6],[8,10,12]])
b = np.arange(6).reshape((2,3))
print(a,b,a-b,a*b, sep='\n') #* is normal multiplication
print('\n')
print(b.T) #transpose
print(np.dot(a,b.T)) # This is array multiplication


# In[46]:

#Merge
A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B))) #vertical merge
print(np.hstack((A,B))) #horizontal merge


# In[47]:

#Split
a = np.arange(12).reshape((3,4))
print(a)
print(np.vsplit(a,3)) #Vertically split the array into 3 new arrays
print(np.hsplit(a,2)) #Horizontally split the array into 2 new arrays


# <h1 style="color: red">Pandas for Beginners</h1>

# <h3 style="color:blue">Series</h3>

# In[7]:

import pandas as pd
s = pd.Series([1,'abc','6',np.nan,44,1])
a = np.array([1,'abc','6',np.nan,44,1])

print(s)
print(a)


#  <h3 style="color: blue">DataFrame</h3> 

# In[ ]:

#Method one: use array to create DataFrame

df = pd.DataFrame(np.random)
print(df)

