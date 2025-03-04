"""
Utility functions used to generate Kernel matrices for training and test datasets.

@author: Ankur Kumar @ MLforPSE.com
"""

import numpy as np

def rbf_kernel(X, Y, gamma):
    """
    Compute the Gaussian RBF kernel between two matrices X and Y::
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (m, d) NumPy array (m datapoints each with d features)
        gamma - the gamma parameter of gaussian function (scalar)

    Returns:
        kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    n, d = X.shape
    m = Y.shape[0]

    kernel_matrix = X**2 @ np.ones((d,m)) + np.ones((n,d)) @ Y.T**2 - 2*(X @ Y.T)
    kernel_matrix = np.exp(-gamma*kernel_matrix)
    
    return kernel_matrix


def Kernel(X, width):
    """

    Args:
        X - N x dim matrix of input data (number of samples  x dimension)
        width (float) : width of the Gaussian Kernel

    Returns:
        K - N x N  kernel matrix  
        
    """

    N, m = X.shape
    K = np.zeros((N, N))
    
    for i in range(N):
        K[i,i] = 0
        for j in range(i+1, N):
            vec_diff = X[i,:]-X[j,:]
            K[i, j] = np.sum(vec_diff**2)
            K[j,i] = K[i,j]   
    K = np.exp(-K/width)

    return K


def Kernel_test(X, Xt, width):
    """

    Args:
        X -  N  x dim matrix of training input data (number of samples  x dimension)
        Xt - Nt x dim matrix of testing  input data (number of samples  x dimension)
        width (float) : width of the Gaussian Kernel

    Returns:
        K_tst - Nt x N  kernel matrix  
        
    """

    N, m = X.shape
    Nt, m = Xt.shape
    K_tst = np.zeros((Nt, N))
    
    for i in range(Nt):
        for j in range(N):
            vec_diff = Xt[i,:]-X[j,:]
            K_tst[i, j] = np.sum(vec_diff**2)
    K_tst = np.exp(-K_tst/width)

    return K_tst
