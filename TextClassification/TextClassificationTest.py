# -*- coding: utf-8 -*-
"""
Created on Sat 25 13:40:13 2016

@author: Nahid
"""

#from nltk.stem.lancaster import LancasterStemmer
#st = LancasterStemmer()
#st.stem('maximum')

# ==================== loading some important libraries ========================== #
import numpy as np
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPClassifier
#from sklearn.neural_network import BernoulliRBM
from sklearn import tree



categories = [
   'sports',
   'politics'
]

train_path = r"E:\Projects and Codes\ML\NeuralNet\NeuralNetPractice\train"
test_path =  r"E:\Projects and Codes\ML\NeuralNet\NeuralNetPractice\test"

# ============================= loading training data and transforming it ================================== #

train_data = sklearn.datasets.load_files(train_path, 
                                         description=None, categories=None, load_content=True, shuffle=True,
                                         encoding='utf-8', decode_error='strict', random_state=42)
print("data loading complete")
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(train_data.data)
tfidf_transformer = TfidfTransformer(use_idf=True).fit(x_train_counts)
x_train_tfidf = tfidf_transformer.transform(x_train_counts)
x_train = x_train_tfidf.toarray()
y_train = train_data.target
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

#print("x_train.size: ", x_train.size, " y_train.size: ", y_train.size)

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

#print("x_test.size: ", x_test.size, " y_test.size: ", y_test.size, "x_test: ")

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


#print y_test

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)

print clf.predict(x_test)

print y_test

print clf.score
