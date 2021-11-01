
# coding: utf-8

# Q1) Load the dataset and print the shape of the data that you imported.

# In[21]:


import pandas as pd
import numpy as np
data = pd.read_csv('energy_output.csv', names=['AT','EV', 'AP', 'RH', 'PE'])
print(data.shape)


# Q2) Display the first 12 lines of the dataset with all the columns.

# In[22]:


data.head(12)


# Q3) Separate out the independent and dependent variables and store them into X and y.

# In[23]:


x = data.iloc[:,:4]
y = data.iloc[:,4:5]


# Q4) Split the dataset into training (80%) and testing (20%).

# In[24]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, train_size=.80, random_state=0)


# Q5) Plot a scatter plot of Ambient Temperature (X-axis) and energy output (Y-axis). What kind of a relationship exists between the two? (i.e. linear, non-linear, strong, weak).
# #### Ans: A strong negative linear relationship between Ambient Temp and Energy Output

# In[25]:


import matplotlib.pyplot as plt

colors = np.random.rand(9568)
plt.scatter(x['AT'], y['PE'], c=colors, marker='+')
plt.title('Ambient Temp vs. Energy Output')
plt.xlabel('Ambient Temp')
plt.ylabel('Energy Output')
plt.show()


# Q6) Plot a scatter plot of Exhaust Vacuum (X-axis) and energy output (Y-axis). What kind of a relationship exists
# between the two? (i.e. linear, non-linear, strong, weak).
# #### Ans: A weak negative linear relationship(or quadratic) between Exhaust Vacuum and Energy Output

# In[26]:


plt.scatter(x['EV'], y['PE'], c=colors)
plt.title('Exhaust Vacuum vs. Energy Output')
plt.xlabel('Exhaust Vacuum')
plt.ylabel('Energy Output')
plt.show()


# Q7) Plot a scatter plot of Ambient Pressure (X-axis) and energy output (Y-axis). What kind of a relationship exists between the two? (i.e. linear, non-linear, strong, weak).
# #### Ans: A weak positive linear relationship between Ambient Pressure and Energy Output

# In[32]:


plt.scatter(x['AP'], y['PE'], c=colors, marker='^')
plt.title('Ambient Pressure vs. Energy Output')
plt.xlabel('Ambient Pressure')
plt.ylabel('Energy Output')
plt.show()


# Q8) Plot a scatter plot of Relative Humidity (X-axis) and energy output (Y-axis). What kind of a relationship exists
# between the two? (i.e. linear, non-linear, strong, weak).
# #### Ans: A somewhat positive linear relationship between Relative Humidity and Energy Output

# In[33]:


plt.scatter(x['RH'], y['PE'], c=colors, marker='+')
plt.title('Relative Humidity vs. Energy Output')
plt.xlabel('Rel Humidity')
plt.ylabel('Energy Output')
plt.show()


# Q9) Fit a LinearRegression Model using the 4 independent variables

# In[29]:


from sklearn.linear_model import LinearRegression
objlinear = LinearRegression().fit(x_train, y_train)


# Q10) Print the model coefficients. That is, the intercept and 4 coefficients for the 4 independent variables

# In[30]:


names=['Ambient Temp','Exhaust V   ', 'Ambient Pres', 'Rel Humidity']
for i in objlinear.intercept_:
    print('Constant     :', i)
for i in range(len(names)):
    print(names[i],':', objlinear.coef_[0][i])
       


# Q11) What is the Mean Squared Error of the model on the test set?

# In[31]:


y_pred = objlinear.predict(x_test)
se     = (y_pred - y_test.iloc[:,:].values).T @ (y_pred - y_test.iloc[:,:].values)
mse    = se/y_test.shape[0]
print('MSE_TEST:', mse[0][0])

