# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:22:10 2022

@author: anilk
"""

"""
1.Find the unrelated/unnecessary/redundant columns
2.Create a model of Linear Regression, Multinomial Regression, Support Vector Regression, Decision Tree and Random Forest.
3.Compare the results.
4.Predict the salary of the manager who is experienced for 10 years has 100 points, and compare the his/her salary with 
the CEO who is experienced for 10 years has 100 points.
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

errors = {"MLP":[], "PR":[], "SVR":[], "DT":[], "RF":[]}

# read data
data = pd.read_csv("salaries_hw.csv")
columns = ["Id", "Title", "Title Level", "Experience", "Points", "Salary"]
data.columns = columns

# Multiple Linear Regression
#Preprocessing
x = data.iloc[:,2:5]
y = data.iloc[:,5:]
X=x.values
Y=y.values

#Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#model summary
model = sm.OLS(lin_reg.predict(X), X)
result = model.fit()
#print(result.summary())


X=X[:,0].reshape(-1,1)
lin_reg.fit(X,Y) # key part
model = sm.OLS(lin_reg.predict(X), X)
result = model.fit()
#print(result.summary())
r2_score_mlr=r2_score(Y, lin_reg.predict(X))

errors["MLP"].append([result.rsquared,result.rsquared_adj, r2_score_mlr])

# Polynomial Regression

#Polynomial Regression model
poly_reg2 = PolynomialFeatures(degree=2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, Y)

#model summary
model2 = sm.OLS(lin_reg2.predict(poly_reg2.fit_transform(X)), X)
result2 = model2.fit()
print(result.summary())
r2_score_pr=r2_score(Y, lin_reg2.predict(poly_reg2.fit_transform(X)))
errors["PR"].append([result2.rsquared,result2.rsquared_adj, r2_score_pr])

# Support Vector Regression

#preprocessing
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(X)
sc2=StandardScaler()
y_scaled=sc2.fit_transform(Y)

#Support Vector Regression Model
svr = SVR(kernel='rbf')
svr.fit(x_scaled, y_scaled)

#model summary
model3 = sm.OLS(svr.predict(x_scaled), x_scaled)
result3=model.fit()
print(result3.summary())
r2_score_svr=r2_score(y_scaled, svr.predict(x_scaled))
errors["SVR"].append([result3.rsquared, result3.rsquared_adj, r2_score_svr])

# Decision Tree
#Decision tree model
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X,Y)

#model summary
model4 = sm.OLS(dt_reg.predict(X), X)
result4 = model4.fit()
print(result4.summary())
r2_score_dt=r2_score(Y, dt_reg.predict(X))
errors["DT"].append([result4.rsquared, result4.rsquared_adj, r2_score_dt])

# Random Forest 
#Random Forest Model
rf = RandomForestRegressor()
rf.fit(X,y)

#model
model5=sm.OLS(rf.predict(X), X)
result5=model5.fit()
print(result5.summary())
r2_score_rf=r2_score(Y, rf.predict(X))
errors["RF"].append([result5.rsquared, result5.rsquared_adj, r2_score_rf])


"""
Correlation Matrix: It shows the relation between columns
"""

print(data.corr())

"""
                   Id  Title Level  Experience    Points    Salary
Id           1.000000     0.331847    0.206278 -0.251278  0.226287
Title Level  0.331847     1.000000   -0.125200  0.034948  0.727036
Experience   0.206278    -0.125200    1.000000  0.322796  0.117964
Points      -0.251278     0.034948    0.322796  1.000000  0.201474
Salary       0.226287     0.727036    0.117964  0.201474  1.000000
From the correlation matrix, we can see that Salary column mostly depends on Title Level, and nearly no dependence
to Experience.

"""
