# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:37:37 2022

@author: anilk
"""

"""
How to handle missing(NaN) values in python.
"""

#modules
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

#codes

#loading data
data = pd.read_csv("missingvalues.csv")
#print(data)

#filling missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
age = data.iloc[:, 1:4].values
imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])
print(age)