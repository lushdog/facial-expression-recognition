import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from util import *


class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=10e-7, mu=0.99,
            decay=0.999, reg=10e-3, epochs=400, batch_sz=100, show_fig=False):
        K = len(set(Y))

        # make validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        #Yvalid_flat same as orginal Y before changed to indicator(?)
        Yvalid_flat = np.argmax(Yvalid, axis=1)
        X, Y = X[:-1000], Y[:-1000]

        # init hidden layers
        N, D = X.shape
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

        for h in self.hidden_layers:
            self.params += h.params

        tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
        tfT = tf.placeholder(tf.float32, shape=(None, K), name='T')

        # output at activation
        act = self.forward(tfX)

        # regularization cost
        rcost = reg * sum([tf.nn.l2_loss(p) for p in self.params])
        # total cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=tfT)) + rcost

        prediction = self.predict(tfX)
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)

        n_batches = N // batch_sz
        costs = []
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                    Ybatch = Y[j * batch_sz:(j * batch_sz + batch_sz)]

                    session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})

                    if j % 20 == 0:
                        c = session.run(cost,
                                        feed_dict={tfX: Xvalid, tfT: Yvalid})
                        costs.append(c)
                        p = session.run(prediction,
                                        feed_dict={tfX: Xvalid, tfT: Yvalid})
                        e = error_rate(Yvalid_flat, p)
                        print('i:', i, 'j:', j, 'nb:', n_batches,
                              'cost:', c, 'error_rate:', e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        act = self.forward(X)
        return tf.argmax(act, 1)


def main():
    X, Y = getData()

    model = ANN([2000, 1000, 500])
    model.fit(X, Y, show_fig=True)


if __name__ == '__main__':
    main()
