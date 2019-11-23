import numpy as np
import matplotlib.pyplot as plt

def rotate(l, n):
    return l[-n:] + l[:-n]

def Create_Toeplitz_Matrix(img,fil):
    
    img = np.array(img)
    l,b = np.array(img).shape
    fil = np.array(fil)
    lf,bf = np.array(fil).shape
    newlf,newbf = [l+lf-1,b+bf-1]
    
    # padding filter
    fil = np.pad(fil, ((newlf-lf, 0), (0, newbf-bf)), 'constant')

    H = []
    
    for i in range(0,newlf):
        li = []
        for j in range(0,b):
            li = li + [np.roll(fil[newlf-1-i],j).tolist()]
        H = H + np.array(li).T.tolist()
    R = np.array(H)
    
    for i in range(0,l-1):
        H = rotate(H,newbf)
        R = np.concatenate((R,np.array(H)),axis=1)
        
    return R


img = Create_Toeplitz_Matrix([[1,4,1],[2,5,3]],[[1,1],[1,-1]])
print(img)
