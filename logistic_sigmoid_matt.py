import numpy as np
import matplotlib.pyplot as plt
import util
from sklearn.utils import shuffle


class LogisticModel(object):

    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=10e-2, reg=10e-15,
            epochs=120000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        N, D = X.shape
        self.w = np.random.randn(D) / np.sqrt(D)
        self.b = 0

        costs = []
        best_validation_error = 1

        for i in range(epochs):
            pY = self.forward(X)
            self.w -= learning_rate * (X.T.dot(pY - Y) + (reg * self.w))
            self.b -= learning_rate * ((pY - Y).sum() + (reg * self.b))

            if i % 20 == 0:
                pYvalid = self.forward(Xvalid)
                cost = util.sigmoid_cost(Yvalid, pYvalid)
                costs.append(cost)
                error_rate = util.error_rate(Yvalid, pYvalid.round())
                print("i:", i, "cost:", cost, "error:", error_rate)
                if error_rate < best_validation_error:
                    best_validation_error = error_rate

        print("best validation error:", best_validation_error)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        return util.sigmoid(X.dot(self.w + self.b))

    def predict(self, X, Y):
        pY = self.forward(X)
        return np.round(pY)

    def score(self, X, Y):
        predictions = self.predict(X)
        return 1 - util.error_rate(Y, predictions)


if __name__ == '__main__':
    X, Y = util.getBinaryData()
    # balance classes (class 1 has 9x less occurrences than class 0)
    X0 = X[Y == 0, :]
    X1 = X[Y == 1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0] * len(X0) + [1] * len(X1))

    model = LogisticModel()
    model.fit(X, Y, show_fig=True)
    model.score(X, Y)
