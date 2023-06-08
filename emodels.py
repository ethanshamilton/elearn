# emodels.py
# ethanscothamilton@gmail.com
# My own implementations of machine learning models so I can learn by doing. 

import numpy as np
import pandas as pd

# TODO: implement linear regression
class LinearRegression:
    # implemented with the assistance of the following tutorial: 
    # https://www.youtube.com/watch?v=ltXSoduiVwY
    """
    Create a linear regression model. 

    Attributes:
        - learning_rate: desired learning rate of the model
        - n_iters: desired number of iterations for fitting the model
    """
    def __init__(self, learning_rate:float=0.001, n_iters:int=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred