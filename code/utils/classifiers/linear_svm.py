import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    非向量化实现, 使用的是for循环
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data. like: x_train
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C. like: y_train
    - reg: (float) regularization strength 正则项强度, 这里默认使用L2正则

    Returns a tuple of:
    - loss as single float 损失值 浮点数
    - gradient with respect to weights W; an array of same shape as W 对应于W维度的梯度, shape和W一致
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]  # C
    num_train = X.shape[0]  # N
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # 不要和神经网络的激活函数搞混，这里得到的就是每个类别的分数
        correct_class_score = scores[y[i]]  # 正确类别的得分
        for j in range(num_classes):  # multiclass svm loss, 这个损失函数对y的梯度不是0就是-1
            if j == y[i]:  # 正确类别,  loss=0
                dW[i, j] += 0  # 这行其实可以不写, 因为默认初始化就是0
                continue
            margin = scores[j] - correct_class_score + 1
            # 正确类别和其他类别的差+安全阈值 note delta = 1
            if margin > 0:
                # 大于0, 如果不大于0就是等于0, 而后者对于loss的计算没有影响
                # 同时, 如果等于0, 则后续梯度计算也为0, 所以只有对margin>0, 才有计算梯度的必要
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] += -X[i]
                # dW[:, j] += X[i].T
                # dW[:, y[i]] += -X[i].T # 结果是一样的
                # 大于0, 求导详见jupyter文件

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train  # 因为是多个样本, 所以dw也需要根据样本数进行平均, 就好像lr会根据batch_size平均一样

    # Add regularization to the loss.
    # 因为loss会加上这个正则项, 所以优化时, 也会希望降低w^2, 这样就可以限制W的大小了
    # 同时为了方便求导, 一般这里会给一个1/2 w^2求导之后刚好就是w了
    loss += 0.5*reg * np.sum(W * W)
    dW += reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data. like: x_train
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C. like: y_train
    - reg: (float) regularization strength 正则项强度, 这里默认使用L2正则
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    num_train = X.shape[0]

    scores = X.dot(W)  # 矩阵乘法
    # without reshape, correct_class_score.shape is (500, )
    # will error in scores - correct_class_score, can not broadcast (500,10)-(500,)
    correct_class_score = scores[range(num_train), y].reshape(-1, 1)
    # print(f"correct_class_score.shape: {correct_class_score.shape}")

    # margin = scores-correct_class_score+1.0
    # loss = np.sum(margin[margin > 0])/num_train + 0.5*reg * np.sum(W * W)
    # 考虑到计算dw，所以选用下面这种写法
    margin = np.maximum(0, scores - correct_class_score + 1)  # note delta = 1
    margin[np.arange(num_train), y] = 0
    loss = np.sum(margin)/num_train + 0.5*reg * np.sum(W * W)

    # W:(D,C),Score:(N,C) margin:(N,C)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    coeff_mat = np.zeros_like(margin)  # margin.shape = score.shape
    coeff_mat[margin > 0] = 1
    coeff_mat[np.arange(num_train), y] = -np.sum(coeff_mat, axis=1)
    # coeff_mat[np.arange(num_train), y] = -9
    # 不一定每一个都是9个1,1个-9，有可能有些在max取值是0，就没有对w求导了
    dW = X.T.dot(coeff_mat)/num_train+reg*W
    return loss, dW
