# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:34:04 2022

@author: anilk
"""

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

# Decision tree regressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X, y)

# Visualization
plt.scatter(X,Y, color='red')
plt.plot(X, dt_reg.predict(X), color='green')
plt.xlabel("Training Level")
plt.ylabel("Salary")
plt.title("Decision Tree")

"""
Decision tree splits the data points into regions. Since
we do not have large data and it is well-seperable, it 
seperates the data into 9 parts, and predicts the data
accordingly.
"""


print(dt_reg.predict([[11]]))
print(dt_reg.predict([[6.6]]))