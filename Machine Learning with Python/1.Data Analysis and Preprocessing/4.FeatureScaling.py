# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 23:51:48 2022

@author: anilk
"""

"""
Feature Scaling
"""
#function
 


#modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

#read data
data = pd.read_csv("dataset.csv")
countries = data.iloc[:, 0:1].values
genders = data.iloc[:, 4:5].values

#preprocessing
le = LabelEncoder()
countries[:,0] = le.fit_transform(data.iloc[:,0])
genders[:,0] = le.fit_transform(data.iloc[:,4])
data.iloc[:,4] = genders
data.iloc[:,0] = countries

#train-test split
y = data["yas"]
x = data.drop("yas", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

#scale features
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
print(X_train)
