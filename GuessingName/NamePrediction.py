# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 10:14:57 2016

@author: NAHID
"""
import numpy as np
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPClassifier
#from sklearn.neural_network import BernoulliRBM
from sklearn import tree

#Step 1: let's define the taining data 
train_path_male = r"E:\Projects and Codes\ML\MachineLearningRecipes\GuessingName\train\male"
train_path_female =  r"E:\Projects and Codes\ML\MachineLearningRecipes\GuessingName\train\female"

train_path = r"E:\Projects and Codes\ML\MachineLearningRecipes\GuessingName\train"
test_path = r"E:\Projects and Codes\ML\MachineLearningRecipes\GuessingName\test"

categories = [
   'male',
   'female'
]
train_data = sklearn.datasets.load_files(train_path, 
                                         description=None, categories=None, load_content=True, shuffle=True,
                                         encoding='utf-8', decode_error='strict', random_state=42)
#print("data loading complete")

count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(train_data.data)
tfidf_transformer = TfidfTransformer(use_idf=True).fit(x_train_counts)
x_train_tfidf = tfidf_transformer.transform(x_train_counts)
x_train = x_train_tfidf.toarray()
y_train = train_data.target
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

#print("Tarin Data : ", train_data.data)

train_data = -1
x_train_counts = -1
x_train_tfidf = -1

# ============================= loading test data and normalizing it ======================================= #

test_data = sklearn.datasets.load_files(test_path, 
                                         description=None, categories=None, load_content=True, shuffle=True,
                                         encoding='utf-8', decode_error='strict', random_state=42)


x_test_counts = count_vect.transform(test_data.data)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)
x_test = x_test_tfidf.toarray()
y_test = test_data.target
x_test = scaler.transform(x_test)

#print("test Data : ", test_data.data)

#print("Test data : ", x_test)
test_data = -1 
x_test_counts = -1
x_test_tfidf = -1

# ========================= training multilayer perceptron and testing it ================================ #

#clf = MLPClassifier(alpha = 0.005, activation='logistic', max_iter=10, verbose = True)
#clf.fit(x_train, y_train)
#clf.score(x_test, y_test)


#clf = BernoulliRBM(n_components=2)
#clf.fit(x_train, y_train)
#clf.score_samples(x_test)
#
#from sklearn.metrics import accuracy_score
##print y_test
#
#clf = tree.DecisionTreeClassifier()
#
#clf = clf.fit(x_train, y_train)
#
#prediction = clf.predict(x_test)
#print "DT Accuracy  : ", accuracy_score(y_test, prediction)
#--------------------------------Loading Text Files-----------------------------
#
menData = []
with open('inputMen.txt') as inputfile:
    for line in inputfile:
        menData.append(line.strip().split('\n'))

#print("data: ", menData)

womenData = []
with open('inputWomen.txt') as inputfile:
    for line in inputfile:
        womenData.append(line.strip().split('\n'))

#print("data: ", menData)

trainData = menData + womenData
X_train = trainData
#print("data: ", X_train)

y_train = []

for i in menData:
    y_train.append(1)
for i in womenData:
    y_train.append(0)
#print("label: ", y_train)    

testData = []
with open('input.txt') as inputfile:
    for line in inputfile:
        testData.append(line.strip().split('\n'))
#print("label: ", testData)    


#-----------------------------------------------------------------------------------
x_train_list = []

for data in X_train[355]: 
    print("data: ", data)
    x_test_counts = count_vect.transform(data)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    x_test = x_test_tfidf.toarray()
    x_test = scaler.transform(x_test)
    x_train_list.append(x_test)

print("x_train_list: ", x_train_list)
print("y_train: ", y_train[355])
x_test_list = []

for data in testData[5]: 
    print("data: ", data)
    x_test_counts = count_vect.transform(data)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    x_test = x_test_tfidf.toarray()
    x_test = scaler.transform(x_test)
    x_test_list.append(x_test)

print("x_test_list: ", x_test_list)
