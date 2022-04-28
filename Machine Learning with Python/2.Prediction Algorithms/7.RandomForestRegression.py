# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:01:59 2022

@author: anilk
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv("salary_data.csv")

# dataframe slicing
x = data.iloc[:, 1:2]
y = data.iloc[:, 2:]

# dataframe to np.array
X = x.values
Y = y.values

# Decision Tree
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X, y)

# Random Forest
rf_reg = RandomForestRegressor(10, random_state=0)
rf_reg.fit(X,Y.ravel())

# Prediction of both Random Forest and Decision Tree
print(f"Random Forest Prediction:{rf_reg.predict([[10.4]])}, Decision Tree Prediction:{dt_reg.predict([[10.4]])}")
print(f"Random Forest Prediction:{rf_reg.predict([[6.4]])}, Decision Tree Prediction:{dt_reg.predict([[6.4]])}")

#Visualization
plt.scatter(X, Y, color="red")
plt.plot(X, rf_reg.predict(X), color="blue")
plt.title("Random Forest")
plt.xlabel("Training Level")
plt.ylabel("Salary")

# Comments
"""
Decision tree splits the data into parts, and and predicts the data accordingly. Random forest is a type of
ensemble learning, which is combining different type of learning alghorithms. Here, Random Forest contains
a few(10 in our case) different decision trees, and actually the data is being seen different eyes, and thus the data generalized
well.
"""