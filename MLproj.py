#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalanceCascade
from imblearn.ensemble import EasyEnsemble
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#import google.datalab.storage as storage
#from io import BytesIO
#mybucket= storage.Bucket('onyx-elevator-237316')
#
#data_csv= mybucket.object('creditcard.csv')
#uri = data_csv.uri
#%gcs read --object $uri --variable data_csv
#
##Loading the dataset
dataset = pd.read_csv('creditcard.csv')
datasetF = dataset[dataset['Class']==1]
datasetN = dataset[dataset['Class']==0].sample(n=15000)
X1 = datasetF.merge(datasetN, how='outer')
X = X1.iloc[:,0:30] # sliced data to make the model run faster
y = X1.iloc[:,-1]
X.shape


# In[2]:


# Classes and Functions
class sampling:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def undersampling(self):
        rus = RandomUnderSampler(random_state=42)
        X_us, y_us = rus.fit_resample(self.X, self.y)
        return X_us, y_us
    def oversampling(self):
        ros = RandomOverSampler()
        X_os, y_os = ros.fit_resample(self.X, self.y)
        return X_os, y_os
    
    def balanceCad(self):
        bc = BalanceCascade(random_state=42)
        X_bc, y_bc = bc.fit_resample(self.X, self.y)
        return X_bc, y_bc
    
    def easyEnsem(self):
        easyEn = EasyEnsemble(random_state=42) # doctest: +SKIP
        X_es, y_es = easyEn.fit_resample(self.X, self.y) 
        return X_es, y_es
    
    def smote(self):
        smot = SMOTE()
        X_sm, y_sm = smot.fit_resample(self.X, self.y)
        return X_sm, y_sm
    
    def condensNN(self):
        cnn = CondensedNearestNeighbour(random_state=42) # doctest: +SKIP
        X_cnn, y_cnn = cnn.fit_resample(self.X, self.y)
        return X_cnn, y_cnn
    

class supervisedmodel:
    def __init__(self, X, y, scoring, random_state=1, cv = 5):
        self.random_state = random_state
        self.cv = cv
        self.X = X
        self.y = y
        self.scoring = scoring
        
    def hyperParTuning(self, model="knn"):
        self.model = model       
        
        if self.model == 'RF':
            rfc = RandomForestClassifier()
            grid_param = {'n_estimators': [10, 15, 20, 25],'criterion': ['entropy'],'bootstrap': [True]}
            grid = GridSearchCV(estimator= rfc,param_grid = grid_param, cv=self.cv, scoring= self.scoring).fit(X = self.X, y=self.y)
            return grid#.best_params_
        
        elif self.model == 'DT':
            dtc = DecisionTreeClassifier()
            params = {'criterion':['entropy']}
            grid = GridSearchCV(estimator= dtc, param_grid = params, cv=self.cv, scoring= self.scoring).fit(X = self.X, y=self.y)
            return grid#.best_params_
        
        elif self.model == 'SVC':
            modelSvc = SVC()
            grid_param ={ 'kernel':['linear', 'rbf']}
            gridSvc = GridSearchCV(estimator = modelSvc, param_grid = grid_param, cv=self.cv, scoring= self.scoring).fit(X = self.X, y=self.y)
            return gridSvc#.best_params_
        
        elif self.model == 'MLP':
            mlp = MLPClassifier(random_state = self.random_state)
            grid_params = {'activation' :['identity', 'relu'], 'hidden_layer_sizes': [(50,) ,(100,)]}
            gridMlp = GridSearchCV(estimator= mlp, param_grid= grid_params, cv=self.cv, scoring= self.scoring).fit(X =self.X, y=self.y)
            return gridMlp#.best_params_
        
        elif self.model == 'LR':
            Lr = LogisticRegression()
            grid_params3 = {'C':[1, 1.5]} #reminder--look for diff dist metrices
            gridLr = GridSearchCV(estimator= Lr, param_grid= grid_params3, cv=self.cv, scoring= self.scoring).fit(X =self.X, y=self.y)
            return gridLr#.best_params_ 
        
        elif self.model == 'NB':
            gnb = GaussianNB()
            grid_param ={}
            gridNb = GridSearchCV(estimator= gnb, param_grid= grid_param, cv=self.cv, scoring= self.scoring).fit(X =self.X, y=self.y)
            return gridNb#.best_params_ 
         
        elif self.model == 'Xgb':
            xgb =XGBClassifier()
            grid_params2 = {'booster':['dart']}
            gridXgb = GridSearchCV(estimator= xgb, param_grid= grid_params2, cv=self.cv, scoring= self.scoring).fit(X =self.X, y=self.y)
            return gridXgb#.best_params_
               
        else:
            knn = KNeighborsClassifier()
            grid_params3 = {'n_neighbors':[2,3,4,5,6,7]} #reminder--look for diff dist metrices
            gridKnn = GridSearchCV(estimator= knn, param_grid= grid_params3, cv=self.cv, scoring= self.scoring).fit(X =self.X, y=self.y)
            return gridKnn#.best_params_
        
    def buildmodel(self, model, params):
        self.model = model
        self.params = params
        for i,j in self.params.items():
            self.params[i] = list([j])
            
        par = ParameterGrid(self.params)
        
        if self.model == 'LR':
            for i in par:
                modelLr = LogisticRegression(**i).fit(X = self.X, y=self.y)
            return modelLr
        
        elif self.model == 'NB':
            for i in par:
                modelNb = GaussianNB(**i).fit(X = self.X, y=self.y)
            return modelNb      
            
        elif self.model == 'SVC':
            for i in par:
                modelSvc = SVC(**i).fit(X = self.X, y=self.y)
            return modelSvc
               
        elif self.model == 'Xgb':
            for i in par:
                modelXgb =XGBClassifier(**i).fit(X = self.X, y=self.y)
            return modelXgb
        
        elif self.model == 'MLP':
            for i in par:
                modelMlp = MLPClassifier(**i).fit(X = self.X, y=self.y)
            return modelMlp
        
        elif self.model == 'DT':
            for i in par:
                modelDtc = DecisionTreeClassifier(**i).fit(X = self.X, y=self.y) 
            return modelDtc
        
        elif self.model== 'KNN':
            for i in par:
                modelKnn = KNeighborsClassifier(**i).fit(X = self.X, y=self.y) 
            return modelKnn
        
        elif self.model == 'RF':
            for i in par:
                modelRfc = RandomForestClassifier(**i).fit(X = self.X, y=self.y) 
            return modelRfc
        
        
# FUNCTIONS
def ci(accuracy, std, level= .95, n=10):
    from scipy.stats import norm
    ciUpper = accuracy + norm.ppf(level)*std/np.sqrt(n)
    ciLower = accuracy - norm.ppf(level)*std/np.sqrt(n)
    return (ciLower, ciUpper)
    

def combineList(a):
    '''combines a list of lists'''
    models = []
    for i in a:
        for j in i:
            models.append(j)
    return models

def buildTable(modelObj):
    t = PrettyTable()
    t.clear_rows()
    column = ['Models', 'Rand. Undersampled','Rand. Oversampled', 'SMOTE', 'EasyEnsemble', 'BalanceCascade', 'Condensed NN']
    t.add_column(column[0],['Naive Bayes', 'Logistic Reg ','SV Classifier', 'Xgboost   ', 'ML Perceptron', 'Decision Tree ','Random Forest', 'KN Neighbors'])
    count = 0
    for i in range(0,len(modelObj),8):
        modelObjSlice = modelObj[i : i + 8]
        count +=1
        t.add_column(column[count], ["{:.3f} \u00B1 {:.3f}".format(i[0], i[1]) for i in modelObjSlice])

    return t


# In[3]:


# Below is how you install a module in jupyter notebook
#import sys
#!{sys.executable} -m pip install xgboost

#running all sampling methods
sampl = sampling(X, y)
X_us, y_us = sampl.undersampling()
X_trainUS, X_testUS, y_trainUS, y_testUS = train_test_split(X_us, y_us, test_size=0.10, random_state=0)

X_e, y_e   = sampl.easyEnsem()
X_trainE, X_testE, y_trainE, y_testE = train_test_split(X_e[1], y_e[1], test_size=0.10, random_state=0)

X_s, y_s   = sampl.smote()
X_trainS, X_testS, y_trainS, y_testS = train_test_split(X_s, y_s, test_size=0.10, random_state=0)

X_b, y_b   = sampl.balanceCad()
X_trainB, X_testB, y_trainB, y_testB = train_test_split(X_b[1], y_b[1], test_size=0.10, random_state=0)

X_cnn, y_cnn = sampl.condensNN()
X_trainC, X_testC, y_trainC, y_testC = train_test_split(X_cnn, y_cnn, test_size=0.10, random_state=0)

X_os, y_os = sampl.oversampling()
X_trainO, X_testO, y_trainO, y_testO = train_test_split(X_os, y_os, test_size=0.10, random_state=0)


# In[ ]:


#Loop for running models

import sklearn
scoring = ['accuracy', 'roc_auc', 'recall']
scoreMethod = [accuracy_score, roc_auc_score, recall_score]

for m, n in zip(scoreMethod, scoring):
    obj  = supervisedmodel(X_trainUS, y_trainUS, scoring=n)
    obje = supervisedmodel(X_trainE, y_trainE, scoring= n)
    objs = supervisedmodel(X_trainS, y_trainS, scoring= n)
    objb = supervisedmodel(X_trainB, y_trainB, scoring= n)
    objc = supervisedmodel(X_trainC, y_trainC, scoring= n)
    objo = supervisedmodel(X_trainO, y_trainO, scoring= n)
    #Running models on the undersampled data
    modelParam = ['NB', 'LR','SVC', 'Xgb', 'MLP', 'DT','RF', 'KNN']
    modelObj = []
    for i, j in zip(modelParam, modelParam):
        k = obj.hyperParTuning(model = i)
        h = obj.buildmodel(model = i , params = k.best_params_)
        j = m(y_testUS, h.predict(X_testUS)), k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObj.append(j)
    
    # Running models on the easy Ensembled data
    modelParam1 = ['NB1', 'LR1','SVC1', 'Xgb1', 'MLP1', 'DT1','RF1', 'KNN1']
    modelObje = []
    for i, j in zip(modelParam, modelParam1):
        k = obje.hyperParTuning(model = i)
        h1 = obje.buildmodel(model = i , params = k.best_params_)
        j = m(y_testE, h1.predict(X_testE)), k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObje.append(j)
    
    # Running model on Smote -- note I significantly reduced the sample size so the model could run
    modelParam2 = ['NB2', 'LR2','SVC2', 'Xgb2', 'MLP2', 'DT2','RF2', 'KNN2']
    modelObjs = []
    for i, j in zip(modelParam, modelParam2):
        k = objs.hyperParTuning(model = i)
        h3 = objs.buildmodel(model = i , params = k.best_params_)
        j = m(y_testS, h3.predict(X_testS)), k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObjs.append(j)
    
    # Running model on balancade data
    modelParam3 = ['NB3', 'LR3','SVC3', 'Xgb3', 'MLP3', 'DT3','RF3', 'KNN3']
    modelObjb = []
    for i, j in zip(modelParam, modelParam3):
        k = objb.hyperParTuning(model = i)
        h4 = objb.buildmodel(model = i , params = k.best_params_)
        j = m(y_testB, h4.predict(X_testB)),k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObjb.append(j)
    
    # Running model on Condensed sample
    modelParam4 = ['NB4', 'LR4','SVC4', 'Xgb4', 'MLP4', 'DT4','RF4', 'KNN4']
    modelObjc = []
    for i, j in zip(modelParam, modelParam4):
        k = objc.hyperParTuning(model = i)
        h5 = objc.buildmodel(model = i , params = k.best_params_)
        j = m(y_testC, h5.predict(X_testC)),k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObjc.append(j)
    
    #Running model of random oversampled model
    modelParam5 = ['NB5', 'LR5','SVC5', 'Xgb5', 'MLP5', 'DT5','RF5', 'KNN5']
    modelObjo = []
    for i, j in zip(modelParam, modelParam5):
        k = objo.hyperParTuning(model = i)
        h6 = objo.buildmodel(model = i , params = k.best_params_)
        j = m(y_testO, h6.predict(X_testO)),k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObjo.append(j)
        
    #combined and print List
    print(str.upper(n))
    combinedList = combineList([modelObj, modelObjo, modelObjs, modelObje, modelObjb, modelObjc])
    print(buildTable(combinedList))
    


# In[ ]:


print(modelObj)


# In[ ]:


## trying different approach to our test data. 
## Test data is data from the undersampling method
#scoring = ['accuracy', 'precision', 'roc_auc', 'recall']
#scoreMethod = [accuracy_score, precision_score, roc_auc_score, recall_score]

for m, n in zip(scoreMethod, scoring):
    #obj  = supervisedmodel(X_trainUS, y_trainUS, scoring=n)
    obje = supervisedmodel(X_e[1], y_e[1], scoring= n)
    objs = supervisedmodel(X_s, y_s, scoring= n)
    objb = supervisedmodel(X_b[1], y_b[1], scoring= n)
    objc = supervisedmodel(X_cnn, y_cnn, scoring= n)
    objo = supervisedmodel(X_os, y_os, scoring= n)
    #Running models on the undersampled data
    #modelParam = ['NB', 'LR','SVC', 'Xgb', 'MLP', 'DT','RF', 'KNN']
    modelObj = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
    #for i, j in zip(modelParam, modelParam):
    #    k = obj.hyperParTuning(model = i)
    #    h = obj.buildmodel(model = i , params = k.best_params_)
    #    j = m(y_testUS, h.predict(X_testUS)), k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
    #    modelObj.append(j)
    
    # Running models on the easy Ensembled data
    modelParam1 = ['NB1', 'LR1','SVC1', 'Xgb1', 'MLP1', 'DT1','RF1', 'KNN1']
    modelObje = []
    for i, j in zip(modelParam, modelParam1):
        k = obje.hyperParTuning(model = i)
        h1 = obje.buildmodel(model = i , params = k.best_params_)
        j = m(y_testUS, h1.predict(X_testUS)), k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObje.append(j)
    
    # Running model on Smote -- note I significantly reduced the sample size so the model could run
    modelParam2 = ['NB2', 'LR2','SVC2', 'Xgb2', 'MLP2', 'DT2','RF2', 'KNN2']
    modelObjs = []
    for i, j in zip(modelParam, modelParam2):
        k = objs.hyperParTuning(model = i)
        h3 = objs.buildmodel(model = i , params = k.best_params_)
        j = m(y_testUS, h3.predict(X_testUS)), k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObjs.append(j)
    
    # Running model on balancade data
    modelParam3 = ['NB3', 'LR3','SVC3', 'Xgb3', 'MLP3', 'DT3','RF3', 'KNN3']
    modelObjb = []
    for i, j in zip(modelParam, modelParam3):
        k = objb.hyperParTuning(model = i)
        h4 = objb.buildmodel(model = i , params = k.best_params_)
        j = m(y_testUS, h4.predict(X_testUS)),k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObjb.append(j)
    
    # Running model on Condensed sample
    modelParam4 = ['NB4', 'LR4','SVC4', 'Xgb4', 'MLP4', 'DT4','RF4', 'KNN4']
    modelObjc = []
    for i, j in zip(modelParam, modelParam4):
        k = objc.hyperParTuning(model = i)
        h5 = objc.buildmodel(model = i , params = k.best_params_)
        j = m(y_testUS, h5.predict(X_testUS)),k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObjc.append(j)
    
    #Running model of random oversampled model
    modelParam5 = ['NB5', 'LR5','SVC5', 'Xgb5', 'MLP5', 'DT5','RF5', 'KNN5']
    modelObjo = []
    for i, j in zip(modelParam, modelParam5):
        k = objo.hyperParTuning(model = i)
        h6 = objo.buildmodel(model = i , params = k.best_params_)
        j = m(y_testUS, h6.predict(X_testUS)),k.cv_results_['std_test_score'][list(k.cv_results_['rank_test_score']).index(1)] 
        modelObjo.append(j)
        
    #combined and print List
    print(str.upper(n))
    combinedList = combineList([modelObj, modelObjo, modelObjs, modelObje, modelObjb, modelObjc])
    tableOfModels = buildTable(combinedList)
    print(tableOfModels)





