# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    # ridge regression: TODO

    N = y.shape[0]
    D = tx.shape[1]
    #calculate w
    XTX = np.dot(tx.T, tx)
    lambdaI = lambda_*2*N*np.eye(D)
    XTX_lambdaI = XTX+lambdaI
    XTy = np.dot(tx.T,y)
    w = np.linalg.solve(XTX_lambdaI, XTy)
    return w
    # ***************************************************
    #raise NotImplementedError
