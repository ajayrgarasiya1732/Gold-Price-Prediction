#!/usr/bin/env python
# coding: utf-8

# # Project : Gold Price Prediction Model

# # 1) Data Collection :

# In[1]:


# Importing Relevant Libraries


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


# Importing CSV data file into dataframe


# In[6]:


df = pd.read_csv("/Users/ajaygarasiya/Library/Containers/com.microsoft.Excel/Data/Desktop/ML/gld_price_data.csv")
df


# # 2) Data Processing

# In[7]:


# Dataframe structure


# In[8]:


df.info()


# In[9]:


# No. of rows and columns


# In[11]:


df.shape


# In[17]:


# Cheaking null values


# In[20]:


df.isnull().sum()


# In[12]:


# Discribe the numerical columns


# In[14]:


df.describe()


# ***Data Discription***
# 
# This is gold price dataset. The dataset gives you information about a gold prices based on several other stock prices.

# ***Feature***
# 
# * Date = mm/dd/yyyy
# * SPX = It is a free-float weighted measurement stock market index of the 500 largest companies listed on stock exchangers in the United States
# * USO = United States Oil Fund 
# * SLV = Silver Price
# * EUR/USD = Currency pair quotation of the Euro against the US

# ***Label***
# 
# * GLD = Gold Price

# # 3) Correlation between features

# In[21]:


# There are two types of correlation:
# 1) Positive Correlation
# 2) Negative Correlation


# In[23]:


correlation = df.corr()
correlation


# In[25]:


# Constructing the heatmap to understand the correlation


# In[36]:


plt.figure(figsize = (10,8))
sns.heatmap(correlation, annot = True,fmt = '.1f', annot_kws = {'size':10})


# In[37]:


# Correlation Values of Gold


# In[39]:


print(correlation['GLD'])


# In[40]:


# Cheacking the distribution of GLD prices


# In[46]:


# Histogram
sns.histplot(df['GLD'], color = 'green')


# # 4) Spliting the features and targets

# In[47]:


X = df.drop(['Date','GLD'],axis = 1)
X


# In[49]:


y = df['GLD']
y


# # 5) Spliting the dataset into Training and Testing Data

# In[63]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state = 50)


# In[64]:


len(X_train)


# In[65]:


len(X_test)


# # 6) Model Selection

# In[91]:


# Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train,y_train)


# In[67]:


rf_model.score(X_train,y_train)


# In[92]:


# Linear Regression Model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)


# In[93]:


lr_model.score(X_train,y_train)


# In[94]:


# Decision Tree Regression Model
from sklearn import tree
t_model = tree.DecisionTreeRegressor()
t_model.fit(X_train,y_train)


# In[95]:


t_model.score(X_train,y_train)


# In[96]:


# K Nearest Neighbors Model
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=10)
knn_model.fit(X_train,y_train)


# In[97]:


knn_model.score(X_train,y_train)


# In[98]:


# Support Vector Regression Model
from sklearn.svm import SVR
svr_model = SVR(kernel = 'rbf')
svr_model.fit(X_train, y_train)


# In[99]:


svr_model.score(X_train,y_train)


# From this 5 models, Decision Tree Regression model and Random Forest Regression model secures highest score and Support Vector Regression model secures lowest score.

# So, We will use Random Forest Regression Model.

# # 7) Applying the Random Forest Regression Model

# In[102]:


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 100)
rf_model.fit(X_train,y_train)


# In[103]:


rf_model.score(X_train,y_train)


# In[107]:


rf_model.predict(X_test)


# # 8) Error Calculation

# In[157]:


# R Squared Error
from sklearn import metrics
y_predicted = list(rf_model.predict(X_test))
error_score = metrics.r2_score(y_test,y_predicted)
print("R Squared error : ", error_score)


# # 9) Compare the actual values and predicted values

# In[158]:


y_test = list(y_test)


# In[159]:


y_predicted = list(rf_model.predict(X_test))


# In[160]:


plt.plot(y_test,color = 'blue', label = 'Actual Value' )
plt.plot(y_predicted, color = 'red', label = 'Predicted Value')

plt.title("Actual Value vs Predicted Value")
plt.xlabel("Number of values")
plt.ylabel("Gold Prices")
plt.legend()
plt.show()


