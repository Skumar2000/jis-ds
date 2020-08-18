# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:34:54 2020

@author: ANKIT
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


mydataset = pd.read_csv('preprocess.csv')


X = mydataset.iloc[:, :-1].values
Y = mydataset.iloc[:, 3].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)


from sklearn.impute import SimpleImputer


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')


imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labelencoder_X = LabelEncoder()

X[:,0] = labelencoder_X.fit_transform(X[:,0])


onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(Y)
