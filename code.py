import numpy as np
import cv2
from scipy import sparse
import scipy.sparse.linalg
import sys

import matplotlib.pyplot as plt

from Algorithms.exact import optimize_T
from Algorithms.LIME import SpedUp_Solver
from Algorithms.illumination import basic

## SAMPLE-1
print("Please wait for Output-1, once the output images are displayed (close the output windows after they are displyed to see the next output).")
img = cv2.imread('Data/9.bmp')

output = SpedUp_Solver(img, alpha = 0.5, epsilon = 0.1, weight_strategy = 2)
cv2.imshow('Input', img)
cv2.imshow('output', output)
cv2.waitKey()
cv2.destroyAllWindows()

## SAMPLE-2
print("Please wait for Output-2.")
img = cv2.imread('Data/7.bmp')

output = SpedUp_Solver(img, alpha = 0.5, epsilon = 0.1, weight_strategy = 2)
cv2.imshow('Input', img)
cv2.imshow('output', output)
cv2.waitKey()
cv2.destroyAllWindows()