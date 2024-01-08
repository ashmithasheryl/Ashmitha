#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px


# In[33]:


# importing pandas module for data frame
import pandas as pd

# loading dataset and storing in train variable
train=pd.read_csv('shark tank - Pollution.csv')
train



# In[28]:


# importing Randomforest
from sklearn.ensemble import RandomForestRegressor


# In[29]:


# importing Randomforest
from sklearn.ensemble import RandomForestRegressor

# creating model
m1 = RandomForestRegressor()

# separating class label and other attributes
train1 = train.drop(['Date','Time','Location','Wind Direction','AQI','Pollution Level'], axis=1)
target = train['AQI']

# Fitting the model
m1.fit(train1, target)


# calculating the score and the score is 97.96360799890066%
m1.score(train1, target) * 100


# In[30]:


train1


# In[18]:


target


# In[19]:



# predicting the model with other values (testing the data)
#assuming the value of PM.2.5,PM.10,O3,NO2,SO2,CO,Temp,humidity and windspeed.
m1.predict([[81,110,13,45,13,8,32,55,7]])


# In[ ]:




