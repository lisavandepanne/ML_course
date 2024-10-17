# -*- coding: utf-8 -*-
"""Exercise 2.

Grid Search
"""

import numpy as np
from costs import compute_loss


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search
#       here when it is done.

# from costs import *


def grid_search(y, tx, grid_w0, grid_w1):
    """Algorithm for grid search.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        grid_w0: numpy array of shape=(num_grid_pts_w0, ). A 1D array containing num_grid_pts_w0 values of parameter w0 to be tested in the grid search.
        grid_w1: numpy array of shape=(num_grid_pts_w1, ). A 1D array containing num_grid_pts_w1 values of parameter w1 to be tested in the grid search.

    Returns:
        losses: numpy array of shape=(num_grid_pts_w0, num_grid_pts_w1). A 2D array containing the loss value for each combination of w0 and w1
    """

    losses = np.zeros((len(grid_w0), len(grid_w1)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.

        # Create a meshgrid for w0 and w1
    W0, W1 = np.meshgrid(grid_w0, grid_w1)
    
    # Stack W0 and W1 to create a (2, num_grid_pts_w0 * num_grid_pts_w1) array of weights
    W = np.vstack([W0.ravel(), W1.ravel()]).T  # Shape will be (num_grid_pts_w0 * num_grid_pts_w1, 2)

    # Compute the predictions for all weights
    predictions = tx @ W.T  # Shape will be (N, num_grid_pts_w0 * num_grid_pts_w1)

    # Compute the errors
    errors = y[:, np.newaxis] - predictions  # Shape will be (N, num_grid_pts_w0 * num_grid_pts_w1)

    # Compute the MSE for all combinations and reshape it back to (num_grid_pts_w0, num_grid_pts_w1)
    losses = 0.5 * np.mean(errors ** 2, axis=0).reshape(W0.shape)

    return losses
    # ***************************************************
    raise NotImplementedError
    return losses
# ***************************************************
