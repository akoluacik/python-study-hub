# -*- coding: utf-8 -*-
"""
Created on Sun May  1 23:50:51 2022

@author: anilk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  1 23:46:21 2022

@author: anilk
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Read data
data = pd.read_csv("dataset.csv")

x=data.iloc[:,1:4].values
y=data.iloc[:,4:].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=0)

#Random Forest Classifier Model
rfc=RandomForestClassifier(n_estimators=6,criterion='entropy')
rfc.fit(X_train, y_train)

#Predict
y_pred=rfc.predict(X_test)

#Confusion Matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)