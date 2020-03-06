#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# In[14]:


#A 
#Read Dataset into DataFrame and assign Features and Label to variables
df = pd.read_csv("https://github.com/mpourhoma/CS4662/raw/master/Cancer.csv")

feature_cols = ['Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion',
                'Single_Epithelial_Cell_Size','Bare_Nuclei', 'Bland_Chromatin','Normal_Nucleoli','Mitoses']

X = df[feature_cols]
y = df['Malignant_Cancer']

len(df)


# In[15]:


#B Use SciKit Learn to split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 2)


# In[16]:


#C LOAD MODEL
my_Decisiontree = DecisionTreeClassifier(random_state=2)
my_Decisiontree.fit(X_train, y_train)


# In[17]:


#C GET SCORE
y_predict_dt = my_Decisiontree.predict(X_test)
score_dt = accuracy_score(y_test, y_predict_dt)
print("DescitionTree" ,score_dt)


# In[32]:


#C GET FPR, TPR
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_dt, pos_label=1)

print(fpr)
print(tpr)


# In[33]:


# GET AUC SCORE
AUC = metrics.auc(fpr, tpr)
print(AUC)


# In[34]:


#Plot ROC Curve
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()

# Roc Curve:
plt.plot(fpr, tpr, color='red', lw=2, 
         label='ROC Curve (area = %0.2f)' % AUC)

# Random Guess line:
plt.plot([0, 1], [0, 1], color='blue', lw=1, linestyle='--')

# Defining The Range of X-Axis and Y-Axis:
plt.xlim([-0.005, 1.005])
plt.ylim([0.0, 1.01])

# Labels, Title, Legend:
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.show()


# In[25]:


#D 
bootstarp_size = 0.8*(len(df))
for i in range(19):
    df.resample(X_train, n_samples = bootstarp_size , random_state=i , replace = True)
    Base_DecisionTree = DecisionTreeClassifier(random_state=2)
   


# In[52]:


#E
from sklearn.ensemble import AdaBoostClassifier
my_AdaBoost = AdaBoostClassifier(n_estimators = 29,random_state=2)

model = my_AdaBoost.fit(X_train, y_train)


# In[56]:


y_pred_dt = model.predict(X_test)
score_dt = accuracy_score(y_test, y_pred_dt)
print("DescitionTree w/ AdaBoost" ,score_dt)

#C GET FPR, TPR
fpr_ADA, tpr_ADA, thresholds = metrics.roc_curve(y_test, y_pred_dt, pos_label=1)

AUC = metrics.auc(fpr_ADA, tpr_ADA)
print("AUC with Ada Boost",AUC)


# In[1]:


#f
from xgboost import XGBClassifier
my_XGBoost = XGBClassifier(n_estimators = 29,random_state=2)

model = my_XGBoost.fit(X_train, y_train)

y_pred_dt = model.predict(X_test)
score_dt = accuracy_score(y_test, y_pred_dt)
print("DescitionTree w/ XGBoost" ,score_dt)

#C GET FPR, TPR
fpr_XG, tpr_XG, thresholds = metrics.roc_curve(y_test, y_pred_dt, pos_label=1)

AUC = metrics.auc(fpr_XG, tpr_XG)
print("AUC with XGBoost",AUC)


# In[27]:


#G
from sklearn.ensemble import RandomForestClassifier
my_RandomForest = RandomForestClassifier(n_estimators = 29, bootstrap = True, random_state=2)

model = my_RandomForest.fit(X_train, y_train)

y_pred_dt = model.predict(X_test)
score_dt = accuracy_score(y_test, y_pred_dt)
print("Random Forrest Score" ,score_dt)

#C GET FPR, TPR
fpr_RF, tpr_RF, thresholds = metrics.roc_curve(y_test, y_pred_dt, pos_label=1)

AUC = metrics.auc(fpr_RF, tpr_RF)
print("AUC with Random Forest",AUC)


# In[ ]:





# In[ ]:




