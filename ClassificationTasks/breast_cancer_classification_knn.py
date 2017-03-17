# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:08:52 2017

@author: NAHID
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)


X = np.array(df.drop(['class'], 1))

y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#print(X_train, y_train)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)


example_mesures = np.array([4,2,1,1,1,2,3,2,1])
example_mesures = example_mesures.reshape(1,-1)

print(example_mesures)

prediction = clf.predict(example_mesures)

print(prediction)


example_mesures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1],[4,2,1,2,2,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
#example_mesures = example_mesures.reshape(4,-1)
example_mesures = example_mesures.reshape(len(example_mesures), -1)

#print(example_mesures)

prediction = clf.predict(example_mesures)

print(prediction)
