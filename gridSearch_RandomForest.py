
# coding: utf-8

# In[16]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Loading the 
dataset = pd.read_csv('carPurchase.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values 

#Split data into Training Set and Test Set
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.0001,random_state=0)

# Normalize Features
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test) #Fitting Classifier to Training Set. Create a classifier object here and call it classifierObj

#Build Model
classifierObj= RandomForestClassifier()

grid_param= {'n_estimators': [10, 15, 20, 25, 30, 40, 50],'criterion': ['gini', 'entropy'],'bootstrap': [True, False]}

# Grid search
gd_sr= GridSearchCV(estimator=classifierObj, param_grid=grid_param, scoring='accuracy', cv=5, n_jobs=-1)
gd_sr.fit(X_train, y_train) 
print(gd_sr.best_params_)  
print(gd_sr.best_score_)

