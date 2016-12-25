# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 12:58:46 2016

@author: NAHID
"""

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500 

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)


#plt.hist([grey_height, lab_height])
#plt.hist([grey_height, lab_height], stacked = True)

plt.hist([grey_height, lab_height], stacked = True, color=['r', 'b'])
plt.show()

