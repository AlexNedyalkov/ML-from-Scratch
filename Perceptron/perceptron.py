import numpy as np

class Perceptron:

    def __init__(self, lr = 0.01, n_iters = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.w, self.b = None, None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # initialize parameters
        self.b = 0
        self.w = np.zeros(n_features)

        # fix y
        y_ = np.array(1 if i > 0 else 1 for i in y)

        for _ in range(self.n_iters):
    
            for idx, x_i in enumerate(X):

                # make predictions
                prediction = self.activation_func(x_i @ self.w + self.b)

                # perceptron update rule
                update = self.lr * (y[idx] - prediction)
                self.w += update * x_i
                self.b += update

    def predict(self, X):
        return  np.array(self.activation_func(X  @ self.w  + self.b))

    def _unit_step_func(self, x):
        return np.where(x > 0, 1, 0)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from utils.accuracy_metrics import accuracy

    X, y = datasets.make_classification(n_samples = 1000, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
        )

    percept = Perceptron()
    percept.fit(X_train, y_train)
    predictions = percept.predict(X_test)

    accu = accuracy(y_test, predictions)
    print(f"Perceptron Accuracy: {accu:.3f}")