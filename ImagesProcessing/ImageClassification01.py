# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 20:17:00 2017

@author: NAHID
"""

import numpy as np
import numpy as np
import cv2

img = cv2.imread('nahid.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('nahidgray.png',img)
    cv2.destroyAllWindows()

    