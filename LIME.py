import numpy as np
import cv2

img = cv2.imread('img3.png') / 255

cv2.imshow('output', img)
#cv2.waitKey()
#cv2.destroyAllWindows()

img2 = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        temp = img[i, j]
        t = np.max(temp)
        #arg = np.argmax()
        img2[i, j, :] = [t,t,t]

img3 = img/(img2 + 0.1)
cv2.imshow('output2', img3)

cv2.waitKey()
cv2.destroyAllWindows()
