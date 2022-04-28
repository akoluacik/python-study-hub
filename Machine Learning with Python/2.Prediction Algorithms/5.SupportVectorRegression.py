# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:08:50 2022

@author: anilk
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:31:10 2022

@author: anilk
"""

#modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


data = pd.read_csv("salary_data.csv")

# dataframe slicing
x = data.iloc[:, 1:2]
y = data.iloc[:, 2:]

# dataframe to np.array
X = x.values
Y = y.values


# feature scaling
"""
SVR algorithm is very sensitive to data whose correlation
is very low. That's why we need to use scaling.
"""
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(X)

sc2 = StandardScaler()
y_scaled = sc2.fit_transform(Y)

# Support Vector Regression
svr_reg = SVR(kernel = 'precomputed')
svr_reg.fit(x_scaled, y_scaled)

# Visualization
plt.scatter(x_scaled, y_scaled, color='red')
plt.plot(x_scaled, svr_reg.predict(x_scaled), color='blue')