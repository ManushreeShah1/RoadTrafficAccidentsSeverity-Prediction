#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


myfile=r'RTA Dataset.csv'
df=pd.read_csv(myfile)
df.head()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.isnull()


# In[10]:


df.isnull().any()


# In[11]:


df.isnull().sum()


# In[12]:


df.info


# In[13]:


df.duplicated()


# In[14]:


df.duplicated().sum()


# In[15]:


df.describe()  #By default, ".describe()" method will show the statistical summary for numerical features only


# In[ ]:


#df.describe(include="category")


# In[18]:


df.describe(include="all")


# In[ ]:





# In[20]:


plt.figure(figsize=(10,5))
sns.histplot(df['Accident_severity'],bins=80)
plt.show()


# In[22]:


df.hist(layout=(1,6), figsize=(18,3))
plt.show()  #histogram of all the numerical features in the data, use the ".hist()


# In[33]:


plt.figure(figsize=(10,7))
sns.boxplot(data=df,x='Number_of_vehicles_involved',y='Accident_severity')
plt.show()


# In[35]:


plt.figure(figsize=(10,7))
sns.boxplot(data=df, y='Number_of_vehicles_involved', x='Accident_severity', hue='Age_band_of_driver')
plt.show()


# In[38]:


sns.boxenplot(data=df, y='Number_of_vehicles_involved')
plt.show()


# In[39]:


plt.figure(figsize=(10,7))
sns.boxenplot(data=df, y='Number_of_vehicles_involved', x='Accident_severity')
plt.show()


# In[40]:


plt.figure(figsize=(10,7))
sns.boxenplot(data=df, y='Number_of_vehicles_involved', x='Accident_severity', hue='Age_band_of_driver')
plt.show()


# In[44]:


sns.violinplot(data=df, y='Number_of_vehicles_involved', x='Accident_severity')
plt.show()


# In[43]:


plt.figure(figsize=(10,7))
sns.violinplot(data=df, y='Number_of_vehicles_involved', x='Accident_severity', hue='Age_band_of_driver')
plt.show()


# In[45]:


sns.scatterplot(x=df['Accident_severity'], y=df['Pedestrian_movement'])
plt.show()


# In[50]:


sns.scatterplot(data=df,x=df['Accident_severity'], y=df['Pedestrian_movement'],hue='Road_surface_type')
plt.show()


# In[51]:


x=list(np.arange(100))
y1=[i+(43.17*np.random.random()) for i in x]
y2=[(57.39*np.random.random())-i for i in x]
plt.subplots(figsize=(15,5))
ax1 = plt.subplot(1,2,1)
plt.xlabel('Positively Correlated Numerical Variables')
sns.regplot(x=x,y=y1, color='green')
ax2 = plt.subplot(1,2,2)
sns.regplot(x=x,y=y2, color='tomato')
plt.xlabel('Negatively Correlated Numerical Variables')
plt.show()


# In[53]:


sns.pairplot(df[['Number_of_vehicles_involved','Number_of_casualties']])  #numerical features only
plt.show()


# In[54]:


correlation_matrix = df[['Number_of_vehicles_involved','Number_of_casualties']].corr()
correlation_matrix


# In[55]:


sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[56]:


sns.scatterplot(x=df['Number_of_vehicles_involved'], y=df['Accident_severity'])
plt.show()


# In[57]:


df['Accident_severity'].value_counts()


# In[58]:


sns.barplot(x=df['Accident_severity'].value_counts().index,
            y=df['Accident_severity'].value_counts().values)
plt.show()


# In[59]:


sns.countplot(data=df, x='Accident_severity')
plt.show()


# In[60]:


plt.figure(figsize=(10,7))
plt.pie(x=df['Accident_severity'].value_counts().values,
        labels=df['Accident_severity'].value_counts().index,
        autopct='%2.2f%%')
plt.show()


# In[61]:


pd.crosstab(index=df['Number_of_vehicles_involved'], columns=df['Accident_severity'])


# In[62]:


# creating a facet grid -- a bar visualization pf crosstabs
grid = sns.FacetGrid(data=df, col='Number_of_vehicles_involved', height=4, aspect=1, sharey=False)
# mapping bar plot and the data on to the grid
grid.map(sns.countplot, 'Accident_severity', palette=['black', 'brown', 'orange'])
plt.show()


# In[63]:


# creating the histogram of 'age'   #numerical and categorical
plt.figure(figsize=(7,5))
sns.histplot(data=df, x='Accident_severity', bins=16, hue='Road_surface_type')
plt.show()


# In[65]:


# visualizing a strip plot
plt.figure(figsize=(7,5))
sns.stripplot(data=df, x='Accident_severity', y='Number_of_vehicles_involved')
plt.show()


# In[66]:


plt.figure(figsize=(7,5))
sns.stripplot(data=df, x='Accident_severity', y='Number_of_vehicles_involved', hue='Road_surface_type')
plt.show()


# In[ ]:


# visualizing a swarm plot
plt.figure(figsize=(7,5))
sns.swarmplot(data=df, x='Accident_severity', y='Number_of_vehicles_involved')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Accident_severity', y='Number_of_vehicles_involved',hue='Road_surface_type')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Accident_severity', y='Number_of_vehicles_involved',hue='Road_surface_type')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Accident_severity', y='Number_of_vehicles_involved',hue='Road_surface_type', size='Sex_of_driver')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Accident_severity', y='Number_of_vehicles_involved',hue='Road_surface_type', size='Sex_of_driver', style='Driving_experience')
plt.show()


# In[ ]:


grid = sns.FacetGrid(data=df, row='sex', col='embark_town', height=5, aspect=1, hue='pclass')
grid.map(sns.scatterplot, 'age', 'fare').add_legend()
plt.show()


# In[ ]:





# In[ ]:


grid = sns.FacetGrid(data=df, col='survived', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'sex', palette='inferno')
plt.show()


# In[ ]:


print("Number of females that survived: ", df[df['survived']==1]['sex'].value_counts().values[0])
print("Number of females that survived: ", df[df['survived']==1]['sex'].value_counts().values[1])


# In[ ]:


grid = sns.FacetGrid(data=df, col='survived', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'pclass', palette='inferno')
plt.show()


# In[ ]:


print("Number of people who travelled in 'First Class' and survived: ", df[df['survived']==1]['pclass'].value_counts().values[0])
print("Number of people who travelled in 'Second Class' and survived: ", df[df['survived']==1]['pclass'].value_counts().values[2])
print("Number of people who travelled in 'Third Class' and survived: ", df[df['survived']==1]['pclass'].value_counts().values[1])


# In[ ]:


grid = sns.FacetGrid(data=df, col='survived', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'embark_town', palette='inferno')
plt.show()


# In[ ]:


print("Number of people boarded from Southampton and survived: ", df[df['survived']==1]['embark_town'].value_counts().values[0])


# In[ ]:


df1 = df[df['parch']>0]
grid = sns.FacetGrid(data=df1, col='survived', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'sex', palette='inferno')
plt.show()


# In[ ]:


print("Number of males who travelled along with their parent or child and survived: ", df[df['survived']==1][df['parch']>0]['sex'].value_counts().values[1])
print("Number of females who travelled along with their parent or child and survived: ", df[df['survived']==1][df['parch']>0]['sex'].value_counts().values[0])


# In[ ]:


df1 = df[df['sibsp']>0]
grid = sns.FacetGrid(data=df1, row='survived', col='pclass', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'sex', palette='inferno')
plt.show()


# In[ ]:


print("Number of females from first class who travelled along with a sibling or a spouse with them and survived: ",  df[df['survived']==1][df['sibsp']>0][df['pclass']==1]['sex'].value_counts()[0])


# In[ ]:


df1 = df[df['age']<20]
grid = sns.FacetGrid(data=df1, col='pclass', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'survived', palette='inferno')
plt.show()


# In[ ]:


print("Number of teenagers in the third class who failed to survive: ", df[df['survived']==0][df['age']<20]['pclass'].value_counts().values[0])


# In[ ]:


df1 = df[df['age']>30][df['pclass']==1]
grid = sns.FacetGrid(data=df1, col='embark_town', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'sex', palette='inferno')
plt.show()


# In[ ]:


print("Number of females above 30yrs age from Cherbourg who travelled in first class: ", df[df['age']>30][df['pclass']==1][df['embark_town']=='Cherbourg']['sex'].value_counts().values[0])


# In[ ]:


mean_fare = df['fare'].mean()
df1 = df[df['fare']>mean_fare][df['pclass']==1]
grid = sns.FacetGrid(data=df1, col='embark_town', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'sex', palette='inferno')
plt.show()


# In[ ]:


print("Number of females from Southampton who paid a higher fare to travel in the first class: ", df[df['fare']>mean_fare][df['pclass']==1][df['embark_town']=='Southampton']['sex'].value_counts().values[0])


# In[2]:


import pandas as pd
import numpy as np
train=pd.read_csv('RTA Dataset.csv')
test=pd.read_csv('RTA Dataset.csv')

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

# First few rows of the training dataset
train.head()


# In[6]:


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[7]:


train_missing= missing_values_table(train)
train_missing


# In[8]:


test_missing= missing_values_table(test)
test_missing


# In[13]:


get_ipython().system('pip install missingno')
import missingno as msno
msno.bar(train)


# In[14]:


msno.matrix(train)


# In[15]:


msno.matrix(train.sample(100))


# 1. Missing Completely at Random (MCAR)
# 
# The missing values on a given variable (Y) are not associated with other variables in a given data set or with the variable (Y) itself. In other words, there is no particular reason for the missing values.
# 
# 2. Missing at Random (MAR)
# 
# MAR occurs when the missingness is not random, but where missingness can be fully accounted for by variables where there is complete information.
# 
# 3. Missing Not at Random (MNAR)
# 
# Missingness depends on unobserved data or the value of the missing data itself.
# 
# All definitions taken from Wikipedia: https://en.wikipedia.org/wiki/Missing_data
# 
# Now let us look at nullity matrix again to see if can find what type of missingness is present in the dataset.

# In[16]:


#Finding reason for missing data using matrix plot
msno.matrix(train)


# In[20]:


#sorted by Age
sorted = train.sort_values('Number_of_vehicles_involved')
msno.matrix(sorted)


# In[21]:


#Finding reason for missing data using a Heatmap
msno.heatmap(train)


# In[22]:


sorted = train.sort_values('Number_of_vehicles_involved')
msno.matrix(sorted)


# In[23]:


msno.dendrogram(train)


# Let's read the above dendrogram from a top-down perspective:
# 
# Cluster leaves which linked together at a distance of zero fully predict one another's presence—one variable might always be empty when another is filled, or they might always both be filled or both empty, and so on(missingno documentation)
# Screenshot%202020-04-25%20at%208.19.56%20AM.png
# 
# the missingness of Embarked tends to be more similar to Age than to Cabin and so on.However, in this particluar case, the correlation is high since Embarked column has a very few missing values.
# This dataset doesn't have much missing values but if you use the same methodology on datasets having a lot of missing values, some interesting pattern will definitely emerge.

# In[24]:


#Pairwise Deletion
#Parwise Deletion is used when values are missing completely at random i.e MCAR. During Pairwise deletion, only the missing values are deleted. All operations in pandas like mean,sum etc intrinsically skip missing values.

train.isnull().sum()


# In[25]:


train_1 = train.copy()
train_1['Number_of_vehicles_involved'].mean() 


# In[26]:


train_1.dropna(subset=['Number_of_vehicles_involved'],how='any',inplace=True)
train_1['Number_of_vehicles_involved'].isnull().sum()


# In[27]:


#Basic Imputation Techniques
#Imputating with a constant value
#Imputation using the statistics (mean, median or most frequent) of each column in which the missing values are located
#For this we shall use the The SimpleImputer class from sklearn.
# imputing with a constant

from sklearn.impute import SimpleImputer
train_constant = train.copy()
#setting strategy to 'constant' 
mean_imputer = SimpleImputer(strategy='constant') # imputing using constant value
train_constant.iloc[:,:] = mean_imputer.fit_transform(train_constant)
train_constant.isnull().sum()


# In[30]:


from sklearn.impute import SimpleImputer
train_most_frequent = train.copy()
#setting strategy to 'mean' to impute by the mean
mean_imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 
train_most_frequent.iloc[:,:] = mean_imputer.fit_transform(train_most_frequent)


# In[32]:


train_most_frequent.isnull().sum()


# In[ ]:




