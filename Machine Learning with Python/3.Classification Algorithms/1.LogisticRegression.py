# -*- coding: utf-8 -*-
"""
Created on Sun May  1 01:22:38 2022

@author: anilk
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# Read data
data = pd.read_csv("dataset.csv")

x=data.iloc[:,1:4].values
y=data.iloc[:,4:].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=0)

# Preprocessing
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Logistic Regression Model
log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

predictions=log_reg.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print(cm)