# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:23:55 2016

@author: NAHID
"""

"""
In this program we will build a classifier that can classify whether an image is 
Apple or Orange base on two features Weight and Texure 

Here is the training data:::

Weight    Texture    Label
---------------------------
150g      Bumpy      Oragne 
170g      Bumpy      Orange 
140g      Smooth     Apple 
130g      Smooth     Apple 
----      ------     -----

only two classes :::
--------------------
Apple ------- Orange 
"""
from sklearn import tree

#Step 1: let's define the taining data 
features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
labels = ["apple", "apple", "orange", "orange"]

#we can use numerical values rather than string .. sklearn uses real valued features
#let's say smoot === 1, and bumpy === 0 and apple ==== 0, orange === 1
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
#labels = [0, 0, 1, 1]

#step2: Define a classifier. in this case we use Decission Tree Classifier 
clf = tree.DecisionTreeClassifier()

#Step 3: Train the data using the classifier
clf = clf.fit(features, labels)

#Step 4: Now our classifier is ready to work

print clf.predict([[150, 0]])

#the output is [1]. that mean s it is an Apple 
#cool Yah :D 

