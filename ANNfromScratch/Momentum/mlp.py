import numpy as np
from scipy.special import expit

def forward(X,W1,W2,b1,b2):

    # Z = 1 / (1+np.exp(-(X.dot(W1)+b1))) # sigmoid activation
    Z = X.dot(W1) + b1
    Z[Z<0] = 0 # relu activation
    # A = np.exp(Z.dot(W2)+b2)
    A = expit(Z.dot(W2) + b2)
    Y = A / A.sum(axis=1,keepdims=True) # softmax activation
    return Y, Z

def derivative_w2(Z,Y,predY):
    return Z.T.dot(predY-Y)

def derivative_b2(Y,predY):
    return (predY-Y).sum(axis=0)

def derivative_w1(X,Z,Y,predY,W2):
    # return X.T.dot(((predY-Y).dot(W2.T)*(Z*(1-Z)))) # sigmoid activaion
    return X.T.dot(((predY-Y).dot(W2.T)*np.sign(Z))) # relu activation

def derivative_b1(Z,Y,predY,W2):
    # return ((predY - Y).dot(W2.T) * (Z * (1 - Z))).sum(axis=0) # sigmoid activation
    return ((predY - Y).dot(W2.T) * np.sign(Z)).sum(axis=0) # relu activation






