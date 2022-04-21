# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:49:18 2022

@author: anilk
"""

#modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#load data
data = pd.read_csv("sales_data.csv")
print(data)

#preprocessing
months = data[['Aylar']]
sales = data[['Satislar']]

#train-test split
X_train, X_test, y_train, y_test=train_test_split(months,sales,test_size=0.33,random_state=0)

#feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

#model
lr=LinearRegression()
lr.fit(X_train, y_train)

#prediction
prediction = lr.predict(X_test)

#Visualization
plt.plot(y_test, y_test, "b*-")
plt.plot(prediction, y_test, "r-o")
plt.show()