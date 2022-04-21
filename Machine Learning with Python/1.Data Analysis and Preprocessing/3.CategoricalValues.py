# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:52:18 2022

@author: anilk
"""

"""
How to process categorical values.
How to translate categorical values into numerical values.
"""

#modules
import pandas as pd
from sklearn import preprocessing

#codes
#reading data
data = pd.read_csv("dataset.csv")
countries = data.iloc[:,0:1].values

#preprocessing
#label encoding
labelEncoding = preprocessing.LabelEncoder()
countries[:,0] = labelEncoding.fit_transform(data.iloc[:,0])
print(countries)

#one hot encoding

ohe = preprocessing.OneHotEncoder()
countries = ohe.fit_transform(countries).toarray()
print(countries)