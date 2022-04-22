# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:50:46 2022

@author: anilk
"""

"""
When multiple input features present.
y=a1 * x1 + a2 * x2+...+an * xn + e
"""

#modules
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#load data
data = pd.read_csv("dataset.csv")
countries = data.iloc[:, 0:1].values
genders = data.iloc[:,-1:].values

#encoding
#label encoding
le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(data.iloc[:,0])
print(countries)

genders[:,0] = le.fit_transform(data.iloc[:,-1])
print(genders)

#one-hot encoding
ohe = preprocessing.OneHotEncoder()
genders = ohe.fit_transform(genders).toarray()
countries = ohe.fit_transform(countries).toarray()

"""
One-hot encoding creates dummy variable when it creates
two column. To tackle that, we need to pick only one
column of encoded data.
"""

#updating data
data = data.drop(['ulke', 'cinsiyet'], axis=1) # dropping saved columns
data["cinsiyet"] = genders[:,:1] #adding new column to dataframe

#np.array to dataframe
#genders = pd.DataFrame(genders, columns=["cinsiyet"])
countries = pd.DataFrame(countries, columns=['fr', 'tr', 'us'])
data = pd.concat([data, countries], axis = 1)
#data=pd.concat([data,countries,genders], axis=1)


#train-test split
X = data.drop(["cinsiyet"], axis=1)
y = data.iloc[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33, random_state=0)

#creating model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction
y_pred = regressor.predict(X_test)

# Backward Elimination
"""
Elimination is done to eliminate the unnecessary row which
have less or do not have any effect.
"""


"""
regression for height
"""

#data load
data = pd.read_csv("dataset.csv")
countries = data.iloc[:, 0:1].values
genders = data.iloc[:,-1:].values

y = data.iloc[:,1:2].values # X and y data is being prepared here.
X = data.drop(['boy'], axis=1)


#encoding
#label encoding
le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(X.iloc[:,0])


genders[:,0] = le.fit_transform(X.iloc[:,-1])


#one-hot encoding
ohe = preprocessing.OneHotEncoder()
genders = ohe.fit_transform(genders).toarray()
countries = ohe.fit_transform(countries).toarray()

"""
One-hot encoding creates dummy variable when it creates
two column. To tackle that, we need to pick only one
column of encoded data.
"""

#updating data
X = X.drop(['ulke', 'cinsiyet'], axis=1) # dropping saved columns
X["cinsiyet"] = genders[:,:1] #adding new column to dataframe

#np.array to dataframe
#genders = pd.DataFrame(genders, columns=["cinsiyet"])
countries = pd.DataFrame(countries, columns=['fr', 'tr', 'us'])
X = pd.concat([X, countries], axis = 1)
#data=pd.concat([data,countries,genders], axis=1)


#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=0)

#creating model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction
y_pred = regressor.predict(X_test)

# we have 6 features so our linear regression model is
# y = a0 + a1*x1 + a2*x2+...+a6*x6
# to have a0, we add the statement below
# axis = 1 adds the array as column vector
x = np.append(arr = np.ones((22, 1)).astype(int), values = data, axis=1)

x_l = X.iloc[:, :].values
x_l = np.array(x_l, dtype=float)

import statsmodels.api as sm
model = sm.OLS(y, x_l).fit()
print(model.summary())
"""
By inspecting model.summary(), we can eliminate the feature
whose P>|t| value is the largest(for backward elimination).
"""

x_l = X.iloc[:, [0,2,3,4,5]].values
x_l = np.array(x_l, dtype=float)

import statsmodels.api as sm
model = sm.OLS(y, x_l).fit()
print(model.summary())