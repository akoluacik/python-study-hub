# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:31:10 2022

@author: anilk
"""

#modules
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("salary_data.csv")

# dataframe slicing
x = data.iloc[:, 1:2]
y = data.iloc[:, 2:]

# dataframe to np.array
X = x.values
Y = y.values

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# polynomial regression
poly_reg2 = PolynomialFeatures(degree = 2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, Y)

poly_reg4 = preprocessing.PolynomialFeatures(degree=4)
x_poly4 = poly_reg4.fit_transform(X)
#print(x_poly)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4, Y)

# prediction 

print(lin_reg.predict(6.6))
print(lin_reg.predict(11))

print(lin_reg2.predict(poly_reg2.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg2.fit_transform([[11]])))

print(lin_reg4.predict(poly_reg4.fit_transform([[6.6]])))
print(lin_reg4.predict(poly_reg4.fit_transform([[11]])))

# Visualization
plt.scatter(X,Y) # increases parabolic
plt.plot(x, lin_reg.predict(X), color="blue")
plt.title("linear regression")
plt.xlabel("Training Degree")
plt.ylabel("Salary")
plt.show()

plt.scatter(X, Y, color="red")
plt.plot(X, lin_reg2.predict(poly_reg2.fit_transform(X)), color="blue")
plt.title("2nd degree polynomial regression")
plt.xlabel("Training Degree")
plt.ylabel("Salary")
plt.show()

plt.scatter(X,Y, color="red")
plt.plot(X, lin_reg4.predict(poly_reg4.fit_transform(X)), color="blue")
plt.title("4th degree polynomial regression")
plt.xlabel("Training Degree")
plt.ylabel("Salary")
plt.show()