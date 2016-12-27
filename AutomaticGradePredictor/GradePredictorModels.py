# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 17:21:55 2016

@author: NAHID
"""

from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

# data train [attendance, tutorial, assaignment, midterm]
X_train = [[10.00,4.25,8.00,13.00],
           [9.00,7.33,8.00,25.00], 
           [3.00,6.00,8.00,16.00], 
           [3.00,6.67,8.00,19.00], 
    [10.00,7.00,8.00,28.50], 
    [10.00,9.00,8.00,24.00], 
    [8.00,6.25,8.00,28.50], 
    [9.00,8.25,8.00,28.50], 
    [9.00,6.50,8.00,23.50], 
    [5.00,5.75,8.00,20.50], 
    [8.00,7.50,8.00,24.00],
    [9.00,8.25,8.00,18.00], 
    [3.00,6.67,8.00,15.50],
    [10.00,7.50,8.00,23.00], 
    [10.00,9.50,8.00,12.00],
    [10.00,9.50,8.00,22.00],
    [4.00,9.50,5.00,20.50],
    [4.00,5.00,8.00,22.00],
    [10.00,9.25,8.00,26.00],
    [10.00,9.50,8.00,20.50],
    [8.00,8.00,8.00,25.50],
    [7.00,4.00,8.00,16.00], 
    [10.00,6.25,8.00,26.50],
    [4.00,8.75,8.00,21.00],
    [7.00,6.00,8.00,17.00],
    [5.00,8.50,8.00,13.50],
    [8.00,9.50,8.00,26.50],
    [8.00,5.00,8.00,15.00],
    [3.00,6.14,8.00,17.00],
    [9.00,7.00,8.00,22.50],
    [9.00,8.67,8.00,26.00],
    [10.00,9.50,8.00,25.50]]

#y_train = [23.00, 31.50, 22.00, 24.50, 34.00, 33.50, 30.50,34.00 ,35.00 ,24.50 , 32.00,28.50 , 25.50, 32.50]

y_train = ['B-','A+','B-','B','A+','A+','A+','A+', 'A+', 'B', 'A', 'A-', 'B-', 
'A+', 'B','A-','B','B','A+','A-','A+','B-', 'A','B','B','B+','A','B-','B-','A', 'A+', 'A+']


#classifiers
clf1 = tree.DecisionTreeClassifier()
clf2 = svm.SVC()
clf3 = neighbors.KNeighborsClassifier()
clf4 = GaussianNB()
#train model
clf1 = clf1.fit(X_train,y_train)
clf2 = clf2.fit(X_train,y_train)
clf3 = clf3.fit(X_train,y_train)
clf4 = clf4.fit(X_train,y_train)


X_test = [[10.00,9.50,8.00,27.00],
          [9.00,5.00,8.00,19.00],
    [8.00,5.00,8.00,15.00],
    [10.00,9.25,8.00,25.00],
    [8.00,8.75,8.00,16.00]]
y_test = ['A','A-','B-','A+','B+']

#prediction
prediction1 = clf1.predict(X_test)
prediction2 = clf2.predict(X_test)
prediction3 = clf3.predict(X_test)
prediction4 = clf4.predict(X_test)

print("DT:", prediction1)
print("SVM:", prediction2)
print("KNN:", prediction3)
print("GNBayes:", prediction4)


#results 
r1 = accuracy_score(y_test,prediction1)
r2 = accuracy_score(y_test,prediction2)
r3 = accuracy_score(y_test,prediction3)
r4 = accuracy_score(y_test,prediction4)

print ("DT: ", r1)
print ("SVM: ", r2)
print ("KNN: ", r3)
print ("GNBayes: ", r4)



