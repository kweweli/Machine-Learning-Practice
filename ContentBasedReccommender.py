
# coding: utf-8

# import dependencies
import numpy as np
import pandas as pd

# creating Artificial user movies and ratings
movie  = np.array(['Love', 'Hani goes to Kenya', 'Batman', 'Johnny be good'])
rating = np.array([8, 10, 5, 4])
user   = pd.DataFrame({'movie': movie, 'rating': rating})
display(user)

# onehot encoding of genre for the user's movies
genreUser  = {'Romance': [1,0,0,0], 'Action':[1,1,1,0], 'comedy':[0,1,1,1], 'Scifi':[0,1,1,0]} # oneHotEncoder for genre
genreUser  = pd.DataFrame(genreUser) # making genre a dataframe
userMovies = user.join(genreUser) # join the user movie rating with genre
userMovies

# creating a weighted genre matrix by multiplying the ratings with the genre. Each row in the genre matrix is multiplied
# by the corresponding col of the rating. 
weightedGenre = np.array(userMovies.iloc[:,[1]]) * np.array(userMovies.iloc[:,2:])

view = pd.DataFrame(weightedGenre, columns=['Romance','Action', 'Comedy','Sci-fi']) # creating view of weightedGenre matrix
view


#normalize the weightedGenre matrix by adding accross columns and dividing by the sum of the entire matrix
normalizedWeightedGenre = np.sum(weightedGenre, axis=0)/ np.sum(np.sum(weightedGenre))

#We see that our user loves action(0.353) movies the most
view2 = pd.DataFrame(np.array(normalizedWeightedGenre).reshape(-1,4),  columns=['Romance','Action', 'Comedy','Sci-fi']) 
view2


# In[91]:


# Getting new movies for reccommendation
moviePred = pd.DataFrame(np.array(['Michaels', 'Good Homes', 'Batman 3', 'Johnny be good 2'])) 
genrePred = pd.DataFrame({'Romance': [1,0,1,0], 'Action':[0,1,1,0], 'comedy':[0,0,1,1], 'Scifi':[0,1,0,1]}) # ratings
userPred= moviePred.join(genrePred)
userPred


# In[64]:


# creating a weighted genre matrix for the new movies we need to recomend to the user
weightedPred = np.array(normalizedWeightedGenre) * np.array(userPred.iloc[:,1:])
weightedPred


# In[86]:


# we get the reccommended weights for each of the movies by summing accross the weighted genre matrix/weightedPred
recommend = np.sum(weightedPred, axis=1)
recommendData = np.array(recommend).reshape(-1, 1)
rec =  pd.DataFrame({'Movies': np.array(['Michaels', 'Good Homes', 'Batman 3', 'Johnny be good 2']), 'Weights': recommend})

#rec
rec


# We reccommend Batman 3, followed by Good Homes to the User
