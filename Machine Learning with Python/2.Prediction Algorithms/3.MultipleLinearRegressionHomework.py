# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:53:25 2022

@author: anilk
"""

"""
Creating a model which predicts if tennis can be played based on given conditions.
"""


#modules
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#load data
data = pd.read_csv("hw_data.csv")
play = data.iloc[:, -1:].values
outlook = data.iloc[:, :1].values
data.iloc[:,3] = data.iloc[:,3].astype(int)

#preprocessing
#label encoding
le = preprocessing.LabelEncoder()
play[:, 0] = le.fit_transform(data.iloc[:, -1])
outlook[:,0] = le.fit_transform(data.iloc[:,0])

#one-hot encoding
ohe = preprocessing.OneHotEncoder()
play = ohe.fit_transform(play).toarray()
outlook = ohe.fit_transform(outlook).toarray()


#np.array to pd.dataframe and updating data
play = pd.DataFrame(play, columns=["yes", "no"])
output = play.iloc[:, :1]
outlook = pd.DataFrame(outlook, columns=["overcast", "rainy", "sunny"])
data = data.drop(["play", "outlook"], axis=1)
data = pd.concat([outlook, data], axis=1)
#train-test split
X_train, X_test, y_train, y_test = train_test_split(data, output, train_size=0.67, random_state=0)

#Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

#prediction
preds = lr.predict(X_test)

#backward elimination
bias = pd.DataFrame(np.ones((data.shape[0], 1)), columns=["bias"])
data = pd.concat([bias, data], axis=1)
x_l = data.values.astype(float)
model = sm.OLS(output.iloc[:, 0], x_l).fit()
print(model.summary())

"""
it seems sunny column has the least effect. We can drop sunny column from data, and perform train-test splitting again,
however, we can add sunny column directly from X_train.
"""

X_train.drop(["sunny"], inplace=True, axis=1)
X_test.drop(["sunny"], inplace=True, axis=1)
print(X_train)

lr.fit(X_train, y_train)

#prediction
preds = lr.predict(X_test)
