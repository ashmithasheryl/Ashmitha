#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px


# In[2]:


data=pd.read_csv('food delivery.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data.dropna(inplace=True)


# In[5]:


data.duplicated().sum()


# In[6]:


lat1=data['Restaurant_latitude']
lat2=data['Delivery_location_latitude'] 
long1=data['Restaurant_longitude'] 
long2=data['Delivery_location_longitude']


# In[7]:


import math


# In[8]:


R=6371     #earth's radius in km
def deg_to_rad(degrees):
    return degrees*(np.pi/180)
def distance(lat1, long1, lat2, long2):
    dlat=math.radians(lat2-lat1)
    dlong=math.radians(long2-long1)
    a=math.sin(dlat/2)*math.sin(dlat/2)+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlong/2)*math.sin(dlong/2)
    c=2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d=R*c
    return d

data['distance(km)'] = np.nan      #distance between each points

for i in range(len(data)):
    data.loc[i,'distance(km)']=distance(data.loc[i,'Restaurant_latitude'],data.loc[i,'Restaurant_longitude'],data.loc[i,'Delivery_location_latitude'], data.loc[i,'Delivery_location_longitude'])


# In[9]:


data['Time_taken(min)'] = pd.to_numeric(data['Time_taken(min)'], errors='coerce')
data['Time_taken(min)']= data['Time_taken(min)'].astype(float)


# With distance now lets calculate speed using time.

# In[10]:


data['speed(km)']=round(data['distance(km)']*60/data['Time_taken(min)'])
data.head()


# In[11]:


import matplotlib.pyplot as plt


# Let's drop the columns that we wont be using for our analysis.

# In[12]:


df1=data.drop(['Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Vehicle_condition'],axis=1)
df1.head()


# In[13]:


#Using box-cox transormation.
from sklearn.preprocessing import PowerTransformer
y=data['Time_taken(min)']    #transforming response variable
pt = PowerTransformer(method='box-cox')
transformed_data = pt.fit_transform(np.array(y).reshape(-1, 1))


# In[14]:


# Generate a random dataset
transformed_data = np.random.normal(size=45593)

# Plot the histogram of the dataset
plt.hist(transformed_data, density=True, bins=30)

# Plot the bell curve using the mean and standard deviation of the dataset
mu, std = np.mean(transformed_data), np.std(transformed_data)
x = np.linspace(mu - 3*std, mu + 3*std, 100)
plt.plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * std**2)), linewidth=2, color='r')

# Set the plot title and axis labels
plt.title('Bell Curve Plot')
plt.xlabel('Data')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[15]:


#removing negative values from fitted value
y1=abs(transformed_data)


# In[16]:


#substituted the fitted values
data['Time_taken(min)']=y1
data.head() 


# In[17]:


data['Delivery_person_Age'] = pd.to_numeric(data['Delivery_person_Age'], errors='coerce')
data['Delivery_person_Age']= data['Delivery_person_Age'].astype(float)

data['Delivery_person_Ratings'] = pd.to_numeric(data['Delivery_person_Ratings'], errors='coerce')
data['Delivery_person_Ratings']= data['Delivery_person_Ratings'].astype(float)


# In[18]:


#Checking for Multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

# the independent variables set
X = data[['Delivery_person_Ratings','distance(km)']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)


# # Multiple Linear Regression model

# In[19]:


from sklearn import linear_model
import statsmodels.api as sm

X = data[['Delivery_person_Ratings','distance(km)']]
y2 = data['Time_taken(min)']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, y2)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# with statsmodels
x = sm.add_constant(X) # adding a constant
 
model = sm.OLS(y2, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# In[21]:


from sklearn.metrics import mean_squared_error
model = sm.OLS(y2, X).fit()
predictions = model.predict(X)
mse = mean_squared_error(y2, predictions)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)


# In[115]:


#prediction
X1=4
X2=6
Time_taken= 31.455051415478746+(-1.15600210e+00)*X1+(-2.81491272e-04)*X2
Time_taken


# In[147]:


# calculate residuals
residuals = (y2) - np.mean(predictions)

# plot residuals against predicted values
plt.figure(figsize=(20,10))
plt.scatter(predictions, residuals)
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[ ]:


import statsmodels.formula.api as smf


# In[78]:



import numpy as np
import statsmodels.api as sm
import pylab as py
  
# np.random generates different random numbers
# whenever the code is executed
# Note: When you execute the same code 
# the graph look different than shown below.
  
# Random data points generated
data_points = np.random.normal(0, 1, 45593)    
  
sm.qqplot(transformed_data, line ='45')
py.show()


# # Naive Bayes Classification Model

# In[130]:


from sklearn.model_selection import train_test_split

X = data[['Delivery_person_Ratings','distance(km)']]
y2 = data['Time_taken(min)']

X_train, X_test, y_train, y_test = train_test_split(X,y2,test_size=0.3)


# In[131]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[132]:


model.fit(X_train,y_train)


# In[133]:


n = np.array([[3.9,5]])


# In[134]:


model.predict(n)


# In[135]:


model.score(X_test,y_test)


# # Random Forest Classification

# In[112]:


# importing Randomforest
from sklearn.ensemble import RandomForestRegressor


# In[136]:


# creating model
m1 = RandomForestRegressor()

# separating class label and other attributes
X = data[['Delivery_person_Ratings','distance(km)']]
y2 = data['Time_taken(min)']

# Fitting the model
m1.fit(X,y2)


# In[137]:


m1.predict([[4,10]])


# In[138]:


# calculating the score and the score is 97.96360799890066%
m1.score(X,y2) * 100


# # LSTM model

# In[22]:


#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[[ "Delivery_person_Ratings", 
                   "distance(km)"]])
y = np.array(data[["Time_taken(min)"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

# creating the LSTM neural network model
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()


# In[23]:


# training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=9)


# In[29]:


y_pred = model.predict(xtrain)
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(ytrain, y_pred))

print('RMSE: ', rmse)


# In[24]:


print("Food Delivery Time Prediction")
a = float(input("Ratings of Previous Deliveries: "))
b = int(input("Total Distance: "))

features = np.array([[a,b]])
print("Predicted Delivery Time in Minutes = ", model.predict(features))


# In[123]:


y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print('R2 score:', r2)


# In[ ]:




