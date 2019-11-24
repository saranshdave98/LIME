import numpy as np
import cv2
from scipy import sparse
import scipy.sparse.linalg
import sys

from algorithms import LIME

import matplotlib.pyplot as plt

from algorithms.LIME import spedup

raw_image = cv2.imread("Data/t4.jpg")
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(16,4))
plt.imshow(raw_image)
plt.show()

output = spedup(raw_image, alpha = 0.1, epsilon = 0.2, weight_strategy = 1)
plt.imshow(output)
plt.show()