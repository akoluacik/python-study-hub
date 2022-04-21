# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:33:01 2022

@author: anilk
"""

"""
How to load data from an external file
by using pandas.
"""

# modules
import pandas as pd


# codes
# data loading

data = pd.read_csv("dataset.csv")
print(data)
print(data["boy"])
print(data[["boy", "kilo"]])