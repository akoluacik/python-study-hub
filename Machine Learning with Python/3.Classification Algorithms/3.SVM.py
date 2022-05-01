# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:18:47 2022

@author: anilk
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Read data
data = pd.read_csv("dataset.csv")

x=data.iloc[:,1:4].values
y=data.iloc[:,4:].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=0)

#KNN Model
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

#prediction
predictions=svm.predict(X_test)

#Confusion matrix
cm=confusion_matrix(y_test, predictions)
print(cm)