import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

"""
Paper: http://www.dabi.temple.edu/~hbling/publication/LIME-tip.pdf
"""
#Read Input Image and implement Equation 2 of the Paper

def basic(img):
    # This is the basic illumination map method for image illumination
    img = img/255
    img2 = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    T_hat = np.max(img, axis = 2)
    img2[:, :, 0] = T_hat
    img2[:, :, 1] = T_hat
    img2[:, :, 2] = T_hat

    img3 = img/(img2 + 0.1)
    return img3