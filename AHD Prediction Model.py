

#importing libraries used to create dataframe,to do calculations, and machine learning library
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Read Heart Disease Data

df = pd.read_csv('https://raw.githubusercontent.com/mpourhoma/CS4661/master/Heart_s.csv')


df.head(5)


#Keep Numerical Features in database
feature_cols = ['Age','RestBP','Chol','RestECG','MaxHR','Oldpeak']

# Set x to features and y to labels
X = df[feature_cols]
y = df['AHD']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)


#Predicit Using Different Models
k = 3
knn = KNeighborsClassifier(n_neighbors=k) 

knn.fit(X_train,y_train)

y_predict = knn.predict(X_test)
y_actually = y

accuracy = accuracy_score(y_test, y_predict)

print("KNN:", accuracy)




#Train Data Using Tree & Logicisitc Regression


#Make Object of Logistic Regression
my_logreg = LogisticRegression()

#Make Object of Tree
my_decisiontree = DecisionTreeClassifier(random_state = 5)


#Train Models
my_logreg.fit(X_train, y_train)
my_decisiontree.fit(X_train, y_train)


#Testing on Testing Sets
y_predict_lr = my_logreg.predict(X_test)
y_predict_dt = my_decisiontree.predict(X_test)

#Get Accuracy Scores
score_lr = accuracy_score(y_test, y_predict_lr)
score_dt = accuracy_score(y_test, y_predict_dt)

print("LogisticReg:" ,score_lr)
print("DescitionTree" ,score_dt)


#BEST ONE IS LOG REG 
#WORST IS Descition Tree


#Using One Hot Encoding on the Gender, Thal and Chest Pain columns
one_hot_df = pd.get_dummies(df[['Gender','Thal','ChestPain']])

#Merge One Hot DF with previous DF
merged_df = pd.concat([df,one_hot_df],axis='columns')

#Remove Categorical Columns
new_df = merged_df.drop(['Gender','ChestPain','Thal'], axis='columns')

#train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)


k = 3
knn = KNeighborsClassifier(n_neighbors=k) 

knn.fit(X_train,y_train)

y_predict = knn.predict(X_test)
y_actually = y

accuracy = accuracy_score(y_test, y_predict)

print("KNN:", accuracy)



#Train Data Using Tree & Logicisitc Regression


#Make Object of Logistic Regression
my_logreg = LogisticRegression()

#Make Object of Tree
my_decisiontree = DecisionTreeClassifier(random_state = 5)


#Train Models
my_logreg.fit(X_train, y_train)
my_decisiontree.fit(X_train, y_train)


#Testing on Testing Sets
y_predict_lr = my_logreg.predict(X_test)
y_predict_dt = my_decisiontree.predict(X_test)

#Get Accuracy Scores
score_lr = accuracy_score(y_test, y_predict_lr)
score_dt = accuracy_score(y_test, y_predict_dt)

print("LogisticReg:" ,score_lr)
print("DescitionTree" ,score_dt)





accuracy_list = cross_val_score(my_logreg, X, y, cv=10, scoring='accuracy')
print(accuracy_list)


accuracy_cv = accuracy_list.mean()

print(accuracy_cv)


