# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np


def load_data():
    """load data."""
    data = np.loadtxt("dataEx3.csv", delimiter=",", skiprows=1, unpack=True)
    x = data[0]
    y = data[1]
    return x, y


import numpy as np

import numpy as np

def load_data_from_ex02(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    
    # Load height and weight data
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1, 2], encoding='utf-8')
    height = data[:, 0]
    weight = data[:, 1]
    
    # Load gender data
    gender = np.genfromtxt(
        path_dataset,
        delimiter=",",
        skip_header=1,
        usecols=[0],
        converters={0: lambda x: 0 if x == "Male" else 1},  # No need to decode
        encoding='utf-8'  # Ensure strings are read as UTF-8
    )

    # Convert to metric system
    height *= 0.025  # Convert height from inches to meters
    weight *= 0.454  # Convert weight from pounds to kg

    # Sub-sample if needed
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # Outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])  # Adding outliers in meters
        weight = np.concatenate([weight, [51.5 / 0.454, 55.3 / 0.454]])  # Adding outliers in kg

    return height, weight, gender



def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
