
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 
digits = load_digits()


# 1. Load the dataset and split it into a training set (75%) and a test set (25%).

# In[2]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


# 2. Train Logistic Regression model on the dataset, and print the accuracy of the model using the score method.

# In[ ]:


from sklearn.linear_model import LogisticRegression
objLR = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=0)
objLR.fit(x_train, y_train)

print('Score: {0:.3f}'.format(objLR.score(x_test, y_test)))


# 3. Train SVM with linear kernel, and print the accuracy of the model. 

# In[18]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
objSvc = SVC(kernel ='linear')
objSvc.fit(x_train, y_train)

confMatrix = np.array(confusion_matrix(y_test, objSvc.predict(x_test)))
print('Score ConfMatrix Method:    ', confMatrix.diagonal().sum() / y_test.shape[0]) # will use score from this moment forth
print('Score Accuracy_score Method:',accuracy_score(y_test, objSvc.predict(x_test))) # will use score from this moment forth
print('Score Method:                {0:.16f}'.format(objSvc.score(x_test, y_test)))


# 4. Write a loop trying different values of degree and train SVM with poly kernel. For every value of degree, you should have an accuracy. Plot a graph with degree on x-axis and test accuracy on the y-axis. What value of degree gives you the best accuracy? 
# # linear kernel (deg 1) wins

# In[ ]:


from sklearn.svm import SVC
deg = [1,2,3,4,5,6,7,8,9,10]
svcDict = []
for i in deg:
    objSvc2 = SVC(kernel ='poly', degree = i, gamma='auto')
    objSvc2.fit(x_train, y_train)
    svcDict.append(objSvc2.score(x_test, y_test))
plt.plot(deg,svcDict)
plt.ylabel('Accuracy')
plt.xlabel('Degree of Polynomial')
plt.title('SVC Accuracy with Different Poly Degree')
plt.show()


# 5. Train SVM with RBF kernel, and print the accuracy of the model. 

# In[ ]:


objSvc3 = SVC(kernel ='rbf', gamma='auto')
objSvc3.fit(x_train, y_train)

print('Score: {0:.3f}'.format(objSvc3.score(x_test, y_test)))


# 6. Write a loop trying different values of k and perform classification using k-NN. For every value of k, you should have an accuracy. Plot a graph with k on the x-axis and the test accuracy on the y-axis. What value of k gives you the best accuracy?
# # k = 7

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
knnList = []
for i in k:
    knn = KNeighborsClassifier(n_neighbors = i, p=2, metric='minkowski')
    knn.fit(x_train, y_train)
    knnList.append(knn.score(x_test, y_test))
plt.plot(k,knnList)
plt.ylabel('Accuracy')
plt.xlabel('Number of Nearest Neighbors')
plt.title("KNN Accuracy with Different Ks'")
plt.show()


# 7. Train Naïve Bayes Model, and print the accuracy of the model.

# In[ ]:


from sklearn.naive_bayes import GaussianNB
objNb = GaussianNB()
objNb.fit(x_train, y_train)

print('Score: {0:.3f}'.format(objNb.score(x_test, y_test)))


# 8. Train Decision Tree, and print the accuracy of the model.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
objDt = DecisionTreeClassifier(criterion='entropy')
objDt.fit(x_train, y_train)

print('Score: {0:.3f}'.format(objDt.score(x_test, y_test)))


# 9. Write a loop to train Random Forest with different values of n_estimators. For every value of n_estimators you should have an accuracy. Plot a graph with n_estimators on the x-axis and the test accuracy on the y-axis. What value of n_estimators gives you the best accuracy?

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import operator
import collections
Trees = [i for i in range(1, 81)]
rfcList = {}
count = 1
for i in Trees:
    rfc = RandomForestClassifier(n_estimators = i)
    rfc.fit(x_train, y_train)
    rfcList[count]= rfc.score(x_test, y_test)
    count +=1
sorted_x = sorted(rfcList.items(), key=operator.itemgetter(1), reverse=True)
print("Best Accuracy n_estimator: ",list(collections.OrderedDict(sorted_x))[0])
plt.plot(rfcList.keys(), rfcList.values())
plt.ylabel('Accuracy')
plt.xlabel('Number of Forest')
plt.title("Random Forest Accuracy")
plt.show()

