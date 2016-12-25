# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:51:54 2016

@author: NAHID
"""

"""
Visualising Decission Tree 

"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


iris = load_iris()

#print iris.feature_names
#print iris.target_names
print iris.data[0]
print iris.target[0]
print iris.data[10]
print iris.target[10]


#for i in range (len(iris.target)):
#    print "Example %d : Label %s, Feature %s" % (i, iris.target[i], iris.data[i])


test_idx = [0, 50, 100]

# training data 
training_target = np.delete(iris.target, test_idx)
training_data = np.delete(iris.data, test_idx, axis = 0)

# testing data 
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


# Classifier

clf = tree.DecisionTreeClassifier()

clf = clf.fit(training_data, training_target)

print clf.predict(test_data)
print test_target



# viz code for visualizing the decission tree 

#from sklearn.externals.six import StringIO
#import pydot 
#
#dot_data = StringIO()
#tree.export_graphviz(clf, 
#                     out_file = dot_data,
#                     feature_names = iris.feature_names, 
#                     class_name = iris.target_names, 
#                     filled = True, rounded = True, 
#                     impurity = False)
#
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("irisTree.pdf")
#
#
#
