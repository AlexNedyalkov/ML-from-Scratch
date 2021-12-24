import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import datasets
class BaseRegression:

    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _predict(self, X, w, b):
        raise NotImplementedError

    def _approximation(self, X, w, b):
        raise NotImplementedError

        
class LinearRegression(BaseRegression):


class LogisticRegression(BaseRegression):


if __main__ == '__name__':
    print('main')