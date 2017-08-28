
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
# np.random.random() v.s. np.random.rand()
print('\n')
print(e)
#f = b-a
print('\n')
# print(f)

print(a,b,c,d,e,sep='\n')
# print(a,b,c,d,e,f, sep='\n')
#print(f<-9)
#print(f==5)


# In[6]:

a = np.array([[3,6,9],[12,15,18]])
print(a)
print(np.sum(a))
print(np.sum(a, axis=1)) #add all elements on the basis of row
print(np.sum(a, axis=0)) #add all elements on the basis of column


# In[7]:

a = np.array([[2,4,6],[8,10,12]])
b = np.arange(6).reshape((2,3))
print(a,b,a-b,a*b, sep='\n') #* is normal multiplication
print('\n')
print(b.T) #transpose
print(np.dot(a,b.T)) # This is array multiplication


# In[8]:

#Merge
A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B))) #vertical merge
print(np.hstack((A,B))) #horizontal merge


# In[9]:

#Split
a = np.arange(12).reshape((3,4))
print(a)
print(np.vsplit(a,3)) #Vertically split the array into 3 new arrays
print(np.hsplit(a,2)) #Horizontally split the array into 2 new arrays


# <h1 style="color: red">Pandas for Beginners</h1>

# <h3 style="color:blue">Series</h3>

# In[10]:

import pandas as pd
s = pd.Series([1,'abc','6',np.nan,44,1])
a = np.array([1,'abc','6',np.nan,44,1])
# print(t)
print(s)
print(a)


#  <h3 style="color: blue">DataFrame</h3> 

# In[11]:

#Method one: use array to create DataFrame
print(np.random.random()) # np.random.random returns random floats in the half-open interval [0 1)
print(np.random.randn()) # np.random.randn returns random floats sampled from a univariate nomal distribution


df = pd.DataFrame(np.random.randn(7,3))
print(df)


# In[12]:

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


# In[13]:

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


# In[14]:

# Pandas: Selecting data
print(df1)
print('\n')
print(df1['Lunch'])
print('\n')
print(df1[0:3])


# In[15]:

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


# In[16]:

# Pandas: Conditional expression
print(df1[df1.Lunch > 80])


# In[17]:

# Pandas: Re-assign data
#df1.Dinner[df1.Lunch>80]=40  --> this method Pandas would select df1.Dinner first, and then returns
#a DataFrame that is singly-indexed. Then, another python operation df1_dinner[df1.Lunch>80] will
#happens. THis is because pandas sees these operations as separate events.
df1.loc[:,['Dinner']][df1.Lunch>80] = 40
print(df1[df1.Lunch>80])
df1.loc['20170814','Lunch'] = np.nan
print(df1)
print(df1.dtypes)


# In[18]:

# Pandas: Dealing with missing values
print(df1.isnull())
df1.isnull().sum()


# In[19]:

# Panadas: Dealing with missing values --> replace them with 0
df1.fillna(value=0)


# In[20]:

# Panadas: Dealing with missing values --> erase the missing values
print(df1)
df1.dropna(
    axis=0,    #0: operates under the base of row; 1: operates under the base of column
    how='any'  #'any':Drop the whole row or column if there is any NaN value.
               #'all':Drop the whole row or column if the values in the row or column are all NaN.
)


# In[21]:

# Pandas: Other usages
print(df1)
df1.dropna(thresh=3)  #Keep only the rows with at least 3 non-na values
df1.dropna(subset=['Breakfast']) #Only drops the rows or columns in the subset that have NaN value.


# In[22]:

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


# In[23]:

# Pandas: load file

#load csv file
data = pd.read_csv('nescsv2.csv')
print(data)
data = data.T
print(data)
data.to_csv('nescsv2_transpose.csv')


# In[24]:

# Pandas concat DataFrame

#concat

df1_concat = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2_concat = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3_concat = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
# print(df1_concat, df2_concat, df3_concat,sep='\n')
res = pd.concat([df1_concat, df2_concat, df3_concat], axis = 0, ignore_index = True)
# print('\n')
print(res)
# print('\n')
print(df1_concat.columns)
print(df1_concat.index)

#join: {'inner', 'outer'}
df4_concat = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'],index=[1,2,3])
df5_concat = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'],index=[2,3,4])
res_outer = pd.concat([df4_concat, df5_concat], axis=0, join='outer', ignore_index=False) #join default is outer
res_inner = pd.concat([df4_concat, df5_concat], axis=0, join='inner', ignore_index=False)

print(res_outer)

res_reset = res_inner.reset_index(drop=True)
print(list(res_reset.columns)) # Same as print(list(res_reset))
print(res_reset)
print(res_reset.groupby(['b','c']))
print(res_reset.groupby(['b','c']).groups)
print(res_reset.groupby(['b','c']).groups.values())
# print(res_outer, res_inner, sep='\n')
# print(diff_inner_outer)


# <h4 style='color: blue'>Pandas append DataFrame</h4>
# <span>DataFrame.<b>append</b>(other, ignore_index=False, verify_integrity=False)</span>
# <br>
# <p>Append rows of <i>other</i> to the end of this frame, returning a new object. Columns not in this frame are added as new columns</p>
# <br>
# <table align="left">
#     <tr style="border-bottom: 5px solid white">
#         <td style="border-right: 3px solid white"><b>Parameters:</b></td>
#         <td align='left'>
#             <ul style="list-style: none; text-align: left; line-height: 2rem">
#                 <li><b>Other:</b>The data to append</li>
#                 <li><b>ignore_index:</b>If True, do not use the index labels</li>
#                 <li><b>verify_integrity:</b>If True, raise ValueError on creating index with duplicates</li>
#             <ul>
#         </td>
#     </tr>
#     <tr>
#         <td style="border-right: 3px solid white"><b>Returns:</b></td>
#         <td align='left'>
#             <ul style="list-style: none; text-align: left; line-height: 2rem">
#                 <li><b>append:</b>DataFrame</li>
#             </ul>
#         </td>
#     </tr>
# </table>

# In[25]:

df1_append = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2_append = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

res_append = df1_append.append(df2_append, ignore_index=True)
print(res_append)


# <h4 style='color: blue'>Pandas merge DataFrame</h4>
# <span>DataFrame.<b>merge</b>(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x','_y'), copy=True, indicator=False)</span>
# <br>
# <p>Merge DataFrame objects by performing a database-style join operation by columns or indexes</p>
# <br>
# <table align='left'>
#     <tr style="border-bottom: 5px solid white">
#         <td style='border-right: 2px solid white'><b>Parameters:</b></td>
#         <td>
#             <ul style='list-style:none ; text-align: left; line-height: 2rem'>
#                 <li><b>right:</b> DataFrame</li>
#                 <li><b>how:</b> {'left','right','outer','inner'}, default 'inner'</li>
#                 <li><b>on:</b> label or list</li>
#                 <li><b>left_on:</b> label or list, or array-like</li>
#                 <li><b>right_on:</b> label or list, or array-like</li>
#                 <li><b>sort:</b> boolean, default False</li>
#                 <li><b>suffixes:</b> 2-length sequence</li>
#                 <li><b>copy:</b> boolean, default True</li>
#                 <li><b>indicator:</b> boolean or string, default False</li>
#             </ul>
#         </td>
#     </tr>
#     <tr>
#         <td><b>Returns:</b></td>
#         <td>
#             <ul style='list-style:none; text-align: left; line-height: 2rem'>
#                 <li><b>merged:</b> DataFrame</li>
#             </ul>
#         </td>
#     </tr>
# </table>

# In[26]:

left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                     'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']})
right = pd.DataFrame({'key':['K1','K2','K3','K4'],
                      'C':['C0','C1','C2','C3'],
                      'D':['D0','D1','D2','D3']})

print(left)
print(right)
print('\n')
res_merge_1 = left.merge(right, on='key')
res_merge_2 = left.merge(right,how='inner',left_index=True,right_index=True) #Same as pd.merge(left,right,how='inner')
print(res_merge_1)
print(res_merge_2)
print('\n')

# merge DataFrame base on [key1, key2].
left_2 = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                      'key2':['K0','K1','K0','K1'],
                      'A':['A0','A1','A2','A3'],
                      'B':['B0','B1','B2','B3']})
right_2 = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                        'key2':['K0','K0','K0','K0'],
                        'C':['C0','C1','C2','C3'],
                        'D':['D0','D1','D2','D3']})
res_merge_3 = pd.merge(left_2, right_2, on=['key1','key2'], how='inner')
print(left_2)
print(right_2)
print(res_merge_3)


# <h3 style='color: blue'>pandas & matplotlib.pyplot</h3>
# <br>
# <ol>
#     <li style="line-height: 1rem">
#         <p>plt<b>.plot</b>(data)</p>
#         <p><b>PS:</b> If we provide a single list or array to the plot() command, matplotlib assumes it is a sequence
#         of y values, and automatically generates the x values for us.</p>
#     </li>
# 

# In[27]:

# Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create 100 random numbers from normal distribution
data = pd.Series(np.random.randn(100), index=np.arange(100))
print(data)

# plot a graph from pandas data
data.plot() # Same as plt.plot(data)
plt.show()


# In[69]:

# DataFrame
data_df = pd.DataFrame(
    np.random.randn(10,4),
    index=np.arange(10),
    columns=list('ABCD')
)

print(data_df)

plt.plot(data, 'ko')
plt.show()

#scatter
area = np.pi * (10)**2
ax = data_df.plot.scatter(x='A',y='B',s= area,label='class1',color='lightgreen')
data_df.plot.scatter(x='C',y='D', label='class2',color='darkred', ax=ax)
plt.show()
#bar

menMeans = [20, 35, 30, 35, 37]
womenMeans = [25, 32, 34, 20, 25]
menStd = (2, 3 ,4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(5) # the x location for the groups
width = 0.35 # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, color='#d62728', yerr=menStd)
p2 = plt.bar(ind, womenMeans, width, bottom=menMeans, yerr=womenStd)
print(p1[0])

# p1.set_label('Men')
# p2.set_label('Women')
# plt.legend(loc=0, shadow=True) 
plt.legend((p1, p2),('Men', 'Women'),loc=2,shadow=True,fontsize='medium')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5')) # Get or set the x-limits of the current ticks and labels
plt.yticks(np.arange(0, 81, 10)) # Get or set the y-limits of the current ticks and labels
plt.show()


# In[ ]:

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
