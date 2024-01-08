#!/usr/bin/env python
# coding: utf-8

# # DEMAND AND SUPPLY ANALYSIS

# Demand of a product or a service is the quantity of that product or service the customers are willing to buy in a particular period of time. Supply of a product or a service is the quantity that the producers are willing to provide to the market in a particular period. In demand and supply analysis we see the relationship between the quantity provided and quantity supplied. 

# In[1]:


import pandas as pd
import plotly.express as px


# In[2]:


data=pd.read_csv('rides.csv')      #importing the data
data.head()


# In[3]:


data.isnull().sum()      # checking for missing values in the data


# In[4]:


data.dropna().head()     #dropping the null values


# Now, let us check the relationship between the number of drivers active per hour and riders active per hour.

# In[5]:


demand=data['Riders Active Per Hour']
supply=data['Drivers Active Per Hour']


# In[6]:


fig=px.scatter(data,x="Drivers Active Per Hour",y="Riders Active Per Hour",trendline='ols',title=" Demand and Supply Analysis")
fig.update_layout(xaxis_title="No.of Drivers active per hour (Supply)",yaxis_title="No.of Riders active per hour (Demand)")
fig.show()


# From the above graph we can see that there is a constant relationship between the demand and the supply. Thus, we can say that there is no shortage.

# Now, lets check the elasticity of the demand (small change in quantity due to change in price)

# In[7]:


#calculating elasticity
avg_demand=data['Riders Active Per Hour'].mean()
avg_supply=data['Drivers Active Per Hour'].mean()
change_demand=(max(data['Riders Active Per Hour'])-min(data['Riders Active Per Hour']))/avg_demand*100
change_supply=(max(data['Drivers Active Per Hour'])-min(data['Drivers Active Per Hour']))/avg_supply*100
elasticity=change_demand/change_supply
print("The Elasticity of the demand is", elasticity)


# This signifies that the relationship between the demand and supply is moderate, which means that with 1% increase in the supply would lead to 0.87% decrease in demand for rides and vise versa.

# Now lets add a new column in the dataset by calculting the supply ratio.

# In[8]:


data['Supply Ratio']=data['Rides Completed']/data['Drivers Active Per Hour']
data.head()


# In[9]:


#Visualizing the supply ratio
fig1=px.scatter(data,x=data['Drivers Active Per Hour'],y=data['Supply Ratio'],title='Supply Ratio vs Driver Activity')
fig1.update_layout(xaxis_title="Driver Activity",yaxis_title="Supply Ratio(Rides completed per driver active per hour)")
fig1.show()


# In[10]:


#visualisation
px.bar(data,x='Drivers Active Per Hour',y='Rides Completed',title='Rides completed with active drivers')


# From the graphs we can see that if there are more active drivers available then there are more rides that can be completed. Thus, if supply is more we can get a good profit as the demand will be met.

# In[ ]:




