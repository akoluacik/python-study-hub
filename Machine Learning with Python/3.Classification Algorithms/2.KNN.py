# -*- coding: utf-8 -*-
"""
Created on Sun May  1 02:04:21 2022

@author: anilk
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Read data
data = pd.read_csv("dataset.csv")

x=data.iloc[:,1:4].values
y=data.iloc[:,4:].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=0)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Prediction
predictions = knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print(cm)