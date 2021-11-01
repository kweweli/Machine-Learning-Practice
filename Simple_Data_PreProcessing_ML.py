
# coding: utf-8

# Q1) Load the dataset into a pandas dataframe and display the first 5 lines of the dataset along with the column
# headings. Note that the data does not come along with the column headings, so you should be adding that to the
# data frame.

# In[2]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler

dat = pd.read_stata('Panel101.dta')
dat


# In[1]:


import pandas as pd

dataset = pd.read_csv('people.csv', names=['firstname', 'lastname', 'age', 'company name', 'street name', 'city','county','state', 'zipcode','Phone','email', 'url'])
dataset.head()


# Q2) Drop the county column

# In[2]:


dataset = dataset.drop(columns="county")
dataset


# Q3) Keep only those rows that have a minimum of 4 values, otherwise delete them.

# In[3]:


import numpy as np

dataset = dataset.fillna(value= "NaN", axis = 0)
dataset = dataset.dropna(thresh = 4)
dataset


# Q4) Delete rows with email missing.

# In[4]:


dataset = dataset[dataset['email'].values !="NaN" ]
dataset


# Q5) Impute the missing values in age with the mean of the column.

# In[5]:


dataset['age'] = pd.to_numeric(dataset['age'], errors='coerce') # converted ages to numeric
dataset.iloc[:, 2:3] = dataset.fillna(dataset.iloc[:, 2:3].mean())
dataset


# Q6) Perform one hot encoding for the state column

# In[6]:


oneHotEn = OneHotEncoder(categories = 'auto')
stateEncoded = oneHotEn.fit_transform(dataset.iloc[:, 6:7].values).toarray()[:,:-1]
stateEncoded = pd.DataFrame(data = stateEncoded, columns=['AZ','CA','FL']) #created dataframe for encoded dataset
combinedDat = dataset.join(stateEncoded) # join the oneHotencoder data to the old data set
combinedDat

