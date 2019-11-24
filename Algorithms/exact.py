import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt


def Create_Toeplitz_Matrix(img, fil):
    """
    Create Toeplitz Matrix based on the Filter
    @Params:
    img: Image on which Convolution needs to be performed
    fil: Filter to apply to img
    """

    # Calcuate the size of the image after applying filter
    img = np.array(img)
    l, b = np.array(img).shape
    fil = np.array(fil)
    lf, bf = np.array(fil).shape
    newlf, newbf = [l + lf - 1, b + bf - 1]

    # padding filter
    fil = np.pad(fil, ((newlf - lf, 0), (0, newbf - bf)), 'constant')

    # Calculate Toeplitz Matrix
    H = []

    for i in range(0, newlf):
        li = []
        for j in range(0, b):
            li = li + [np.roll(fil[newlf - 1 - i], j).tolist()]
        H = H + np.array(li).T.tolist()
    R = np.array(H)

    for i in range(0, l - 1):
        H = rotate(H, newbf)
        R = np.concatenate((R, np.array(H)), axis=1)

    return R, newlf, newbf

def Update_G(W, delT, Z, mu, alpha = 0.1):
    """
    Function to update G based on equation 15 of the Paper
    @Params:
    W: Weight of the derivatives
    delT: Directives along X
    """
    Z = Z / mu
    X = delT + Z
    epsilon = alpha * W / mu
    G = np.sign(X) * (np.max(np.abs(X) - epsilon, 0))
    return G

def calculate_delT(T, Dx, Dy, newlf, newbf):
    """
    Function to calculate derivative of an image
    @Params:
    T: Input Image
    Dx: Topelitz Matrix
    Dy: Toplitz Matrix
    newlf, newbf: Size of the image after applying filter
    """
    T = vectorize(T)
    delX = Dx @ T
    delY = Dy @ T

    delX = delX.reshape(newlf, newbf)
    delX = delX[1:-1, 1: -1]

    delY = delY.reshape(newlf, newbf)
    delY = delY[1:-1, 1: -1]

    delT = np.vstack([delX, delY])
    return delT

def calculate_numerator(T_hat, Dx, Dy, G, Z, mu, newlf, newbf):
    """
    Function to calculate Numerator of equation 13
    """
    m, n = T_hat.shape
    DT = np.hstack([Dx, Dy])
    g = vectorize(G - Z / mu)
    result = DT @ g
    result = result.reshape(newlf, newbf)
    result = result[1:-1, 1:-1]
    numerator = 2 * T_hat + mu * result
    return numerator


def Update_T(T_hat, G, Z, mu, Dx, Dy, newlf, newbf, filX, filY):
    """
    Function to update T based on equation 13 of the paper
    @Params:
    T_hat: Initial Illumination Map
    G: Gradients
    Z: Lagrange's Multiplier Vector
    mu: Parameter to balance
    Dx, Dy: Topelitz Matrices
    newlf, newbf: Size of the image after applying filter
    """

    l, b = T_hat.shape

    vector_2 = 2 * np.ones((T_hat.shape[0], T_hat.shape[1]))

    FDx = np.fft.fft2(np.pad(filX, ((l - filX.shape[0], 0), (0, b - filX.shape[1])), 'constant'))
    FDy = np.fft.fft2(np.pad(filY, ((l - filY.shape[0], 0), (0, b - filY.shape[1])), 'constant'))

    Denominator = vector_2 + mu * (np.conjugate(FDx) * FDx + np.conjugate(FDy) * FDy)
    result = calculate_numerator(T_hat, Dx, Dy, G, Z, mu, newlf, newbf)
    Numerator = np.fft.fft2(result)

    temp = np.abs(np.fft.ifft2(Numerator / Denominator))
    return temp


def optimize_T(img2,T_hat, alpha):
    """
    Algorithm 1 of the paper
    @Params:
    T_hat: Initial Illumination Map
    alpha: Scalar Weights on the Gradients
    """

    W = np.ones((img2.shape[0] * 2, img2.shape[1]))
    T = np.zeros((img2.shape[0], img2.shape[1]))
    Z = 2 * np.zeros((img2.shape[0] * 2, img2.shape[1]))
    G = 2 * np.zeros((img2.shape[0] * 2, img2.shape[1]))
    t = 0
    mu = 2
    ro = 1.5
    max_iter = 1

    filX = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    filY = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    Dx, newlf, newbf = Create_Toeplitz_Matrix(T, filX)
    Dy, newlf, newbf = Create_Toeplitz_Matrix(T, filY)

    while (t < max_iter):
        T = Update_T(T_hat, G, Z, mu, Dx, Dy, newlf, newbf, filX, filY)
        delT = calculate_delT(T, Dx, Dy, newlf, newbf)
        G = Update_G(W, delT, Z, mu, alpha)
        Z = Z - mu * (delT - G)
        mu = mu * ro
        t += 1
    return T

def rotate(l, n):
    """
    Function to rotate an array
    @Params:
    l: Array to be rotated
    n: Number of times/positions
    """
    return l[-n:] + l[:-n]

def Reverse(lst):
    """
    Function to reverse a list
    """
    return [ele for ele in reversed(lst)]

def vectorize(X):
    """
    Function to reverse the rows and vectorize an array
    @Params:
    X: Matrix that needs to be vectorized
    """
    m,n = X.shape
    X = Reverse(X.tolist())
    X = np.reshape(np.array(X),(m*n,1))
    return X