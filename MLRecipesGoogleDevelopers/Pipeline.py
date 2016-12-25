# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:20:59 2016

@author: NAHID
"""


#Loading Dataset

from sklearn import datasets
iris = datasets.load_iris() 


X = iris.data
y = iris.target

#print X, y


#Splitting the training and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)


#Classify using DecissionTee
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)

#print prediction


from sklearn.metrics import accuracy_score

print "DecissionTree Accuracy : ", accuracy_score(y_test, prediction)


#Classify using KNN
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier();
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)

#print prediction


from sklearn.metrics import accuracy_score

print "KNN Accuracy  : ", accuracy_score(y_test, prediction)



def classify(features):
    pass

    