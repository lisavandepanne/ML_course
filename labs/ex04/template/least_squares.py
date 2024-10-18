# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    # least squares: TODO
    # returns optimal weights, MSE
    N = y.shape[0]
    
    # Compute optimal weights using least squares formula
    XTX = np.dot(tx.T, tx)  # Compute X^T * X
    XTy = np.dot(tx.T, y)   # Compute X^T * y
    w = np.linalg.solve(XTX, XTy)  # Solve for optimal weights

    # Compute predictions based on optimal weights
    y_pred = np.dot(tx, w)  # X * w
    
    # Compute mean squared error (MSE)
    mse = (1 / N) * np.sum((y - y_pred) ** 2)  # Mean squared error
    
    return w, mse
    # ***************************************************
    raise NotImplementedError
