import numpy as np
import random
import math

class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.wt = None # 10个数字多分类权重

    """
    sigmoid函数
    """
    def _sigmoid(self, t):
        return 1./ (1. + np.exp(-t))

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        def J(theta, X, y):
            y_hat = self._sigmoid(X.dot(theta))
            return -(np.sum(y * np.log(y_hat) + (1 - y) * np.log(y_hat)) / len(y))

        def dJ(theta, X, y):
            return X.T.dot(self._sigmoid(X.dot(theta)) - y) / len(X)


        return J(self.w, X_batch, y_batch), dJ(self.w, X_batch, y_batch)
        #########e################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = [] # 损失函数历史值

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################

            # 可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回
            # 相当于从num_train里面选batch_size个数字来着
            # replace = True的意思是抽样放回 下次选可能会选出之前的数字
            sample_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_index]
            y_batch = y[sample_index]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            # print(loss)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################

            self.w -= learning_rate * grad # 学习率 * 梯度
            # print(self.w)

            # pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 10000 == 0:
                print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################

        score = self._sigmoid(np.dot(X, self.w)) # 相乘预测
        y_pred = np.array(score >= 0.5, dtype='int')

        # pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred


    def predict_one_vs_all(self, X):
        # lables = self._sigmoid(X.dot(self.wt)) # 这里就是计算各个lables了
        # y_pred = np.array(np.argmax(lables[i, :]) for i in range(X.shape[0]))
        # return y_pred

        lables=self._sigmoid(X.dot(self.wt))#这个地方就是用不同数字对应的函数去计算
        # 初始化
        mul_y_pred = np.zeros(X.shape[0])
        for i in range(len(mul_y_pred)):
            mul_y_pred[i]=np.argmax(lables[i,:])
        #选出一个概率最大的出来，例如一组数据（x1,x2,x3,x4,...x10）会与数字0到9的函数计算得出来那哪个的概率最大，哪个概率最大，
        #就能预测他应该是哪一个数字
        return mul_y_pred



    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose = True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        """
        num_train, dim = X.shape

        # 指定权重(theta)
        if self.wt is None:
            self.wt = 0.001 * np.random.randn(dim, 10) # 因为有10个数字 每个数字都要训练


        for i in range(10):
            y_temp = np.array(y == i, dtype='int')
            print(y_temp)
            self.w = None # 每一次都归零 因为train里面会给我们初始化得说
            self.train(X, y, learning_rate, num_iters, batch_size)
            self.wt[:, i] = self.w
            pass
