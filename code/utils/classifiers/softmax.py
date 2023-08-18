import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    # lne=1 ln以e为底 log默认10为底
    # 关于Softmax求导，详见:
    # <https://stackoverflow.com/questions/41663874/cs231n-how-to-calculate-gradient-for-softmax-loss-function>
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]  # C
    num_train = X.shape[0]  # N
    for i in range(num_train):
        scores = X[i].dot(W)
        # shift values for 'scores' for numeric reasons (over-flow cautious)
        # scores -= scores.max()
        loss_i = -scores[y[i]]+np.log(np.sum(np.exp(scores)))
        loss += loss_i  # 2.376615
        # loss -= np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))  # 2.354158
        for j in range(num_classes):
            j_probablity = np.exp(scores[j])/np.sum(np.exp(scores))
            if j == y[i]:
                dW[:, j] += (-1+j_probablity)*X[i]
            else:
                dW[:, j] += j_probablity*X[i]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    loss = loss/num_train+0.5*reg*np.sum(W*W)
    dW = dW/num_train+reg*W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]  # C
    num_train = X.shape[0]  # N
    scores = X.dot(W)
    prob = np.exp(scores[range(num_train), y])/np.sum(np.exp(scores), axis=1)
    # print(prob.shape)
    loss = -np.sum(np.log(prob))
    # print(f"minus before: {-np.sum(np.log(prob))}")
    # print(f"minus after:{np.sum(-np.log(prob))}")

    coeff_mat = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(-1, 1)
    # print(coeff_mat.shape)
    coeff_mat[range(num_train), y] += -1
    dW = X.T.dot(coeff_mat)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #

    loss = loss/num_train+0.5*reg*np.sum(W*W)
    dW = dW/num_train+reg*W

    return loss, dW
