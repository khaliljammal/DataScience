#!/usr/bin/env python
# coding: utf-8

# # Import Libraries & Reading Data

# In[1]:


#importing libraries used to create dataframe,to do calculations, and machine learning library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns


from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


#reading data

test_data = pd.read_csv('https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/test.csv')

df = pd.read_csv('https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv')



# # Data Explorations

# In[3]:


#Basic Stats
df['SalePrice'].describe()


# In[4]:


sns.distplot(df['SalePrice'])


# In[7]:


corrmatrix = df.corr()
top_features = corrmatrix.index[abs(corrmatrix["SalePrice"])>0.5]
plt.figure(figsize=(8,8))
graph = sns.heatmap(df[top_features].corr(),annot=True,cmap="RdYlGn")

#visually see overall quality is related to price


# In[8]:


# features correlation scores

correlation = df.corr()
correlation.sort_values(["SalePrice"], ascending = False, inplace = True)
print(correlation.SalePrice)


# In[9]:


#fill NaNs with 0 using pandas
df.fillna(0,inplace = True)

#drop Id and Sale Price from X columns
X = df.drop(['Id','SalePrice'], axis=1)

#set X and y for data
#feature_cols = list(X.columns)

feature_cols = ['OverallQual','GrLivArea','GarageCars', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
                'TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1',
               'LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF',
               'BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch']

X = X[feature_cols]
y = df['SalePrice']




# In[10]:


#selecting all numerical cols and categorical columns
#num_cols = X._get_numeric_data().columns
#cat_cols = list(set(feature_cols) - set(num_cols))

#train_cat = X[cat_cols]
#train_num = X[num_cols]



#convert ONLY categorial dataframe using onehot encoding
#train_cat = pd.get_dummies(train_cat)


#combine the datagrames
#train = pd.concat([train_cat,train_num],axis=1)

#convert categorical columns into numerical values using one hot encoding/dummy
#X = pd.get_dummies(X, columns = cat_cols, drop_first = True)


#normalizing X with Pandas
#X = (X - X.mean()) / (X.max() - X.min())




# In[11]:


#split training set and data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state= 0)


# In[12]:


# from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_regression Feature extraction
#test = SelectKBest(score_func=f_regression, k=5)
#fit = test.fit(X_train, y_train)

# Summarize scores
#np.set_printoptions(precision=4)
#print(fit.scores_)

#features = fit.transform(X_train)

# Summarize selected features
#top_features = features[0:6,:]
#print(top_features)


# In[13]:


# In the following line, "my_linreg" is instantiated as an "object" of LinearRegression "class". 

my_linreg = LinearRegression()

# fitting the model to the training data:
my_linreg.fit(X_train, y_train)


#Check CoEfficients Theta0 
print("Intercept: ", my_linreg.intercept_)

# print the thetas
#print("Coef: ",my_linreg.coef_)


# In[14]:


# make predictions on the testing set
y_prediction = my_linreg.predict(X_test)

#print("Prediction: ", y_prediction)


# In[15]:


mse = metrics.mean_squared_error(y_test, y_prediction)

# Using numpy sqrt function to take the square root and calculate "Root Mean Square Error" (RMSE)
rmse = np.sqrt(mse)

print("RMSE: ",rmse)

print("MAX Y:",y_prediction.max())
print("MIN Y:",y_prediction.min())


# In[ ]:





# In[18]:


my_logreg = LogisticRegression()

# function cross_val_score performs Cross Validation:
accuracy_list = cross_val_score(my_logreg, X_train, y_train, cv=10, scoring='accuracy')

#print(accuracy_list)


# In[19]:


# use average of accuracy values as final result
accuracy_cv = accuracy_list.mean()

print(accuracy_cv)


# In[20]:


# Applying 10-fold cross validation with "linear regression":

# In the following line, "my_linreg" is instantiated as an "object" of LinearRegression "class". 
my_linreg = LinearRegression()

mse_list = cross_val_score(my_linreg, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

print(mse_list)


# In[21]:


mse_list_positive = -mse_list

# using numpy sqrt function to calculate rmse:
rmse_list = np.sqrt(mse_list_positive)
print(rmse_list)


# In[22]:


print(rmse_list.mean())


# In[23]:


from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 0)
# Train the model on training data
rf.fit(X_train, y_train)


# In[24]:


# Use the forest's predict method on the test data
rf_predictions = rf.predict(X_test)


# In[25]:


mse = metrics.mean_squared_error(y_test, rf_predictions)

# Using numpy sqrt function to take the square root and calculate "Root Mean Square Error" (RMSE)
rmse = np.sqrt(mse)

print("RMSE: ",rmse)


# In[ ]:





# In[ ]:




