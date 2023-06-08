# emodels.py
# ethanscothamilton@gmail.com
# My own implementations of metrics for machine learning models. 

import numpy as np

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)