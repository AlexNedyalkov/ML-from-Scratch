import numpy as np
from sklearn.datasets.samples_generator import make_classification


# Utils
def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class BaseRegression:

    def __init__(self, lr=0.01, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        # implement gradien t descent
        for _ in range(self.n_iters):
            # make prediction
            y_predicted = self._approximation(X, self.weights, self.bias)

            # calcualte gradients
            dw = (1 / self.n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / self.n_samples) * np.sum(y_predicted - y)

            # update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict(self, X, w, b):
        raise NotImplementedError

    def _approximation(self, X, w, b):
        raise NotImplementedError


class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b
        
    def _predict(self, X, w, b):
        return np.dot(X, w) + b

class LogisticRegression(BaseRegression):
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def _approximation(self, X, w, b):
        linear_model =  np.dot(X, w) + b
        return self.sigmoid(linear_model)

    def _predict(self, X, w, b):
        linear_model =  np.dot(X, w) + b
        return np.array([0 if x < 0.5 else 1 for x in self.sigmoid(linear_model)])



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets


    # linear regresion
    X,y = datasets.make_regression(n_samples=100,n_features=5, noise = 20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
        )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    accu = r2_score(y_test, predictions)
    print("Linear reg Accuracy:", accu)

    # logistic regression

    X, y = make_classification(n_samples = 1000, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
        )

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    accu = accuracy(y_test, predictions)
    print("Logistic reg Accuracy:", accu)