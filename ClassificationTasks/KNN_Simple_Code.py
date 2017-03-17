# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:34:29 2017

@author: NAHID
KNN Simplified code
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k' : [[1,2],[2,3],[3,1]], 'r' : [[6,5],[7,7],[8,6]] } 

new_features = [5,7]


plt.scatter(new_features[0],new_features[1])

plt.show()