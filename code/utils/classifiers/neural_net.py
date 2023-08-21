import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.

    一个两层的全连接神经网络, 网络的输入是N维(输入样本的数量是N), 隐藏层的维度是H, C个类别
    使用Softmax损失函数,对权重施加L2正则来训练网络
    第一个全连接层后面使用ReLU非线性函数
    即: 网络具有以下结构
    input - fully connected layer - ReLU - fully connected layer - softmax
    (N,D) - (D,H)                           (H,C)                -(N,C)
    第二个全连接层的输出就是每个类别的分数
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        权重使用很小的随机数初始化,偏置项被初始化为0

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data. 每个输入样本有D维
        - hidden_size: The number of neurons H in the hidden layer. 每个隐藏层有H个神经元
        - output_size: The number of classes C.

        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        计算原理详见: https://cs231n.github.io/neural-networks-case-study/

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        hidden_layer1_output = X.dot(W1)+b1
        relu_output = np.maximum(0, hidden_layer1_output)
        # maximum 逐元素比较两个列表中的值 max 取某个数组中沿着某个轴的最大值
        hidden_layer2_output = relu_output.dot(W2)+b2
        scores = hidden_layer2_output

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # input - fully connected layer - ReLU - fully connected layer - softmax
        # (N,D) - (D,H)                           (H,C)                -(N,C)
        probability = np.exp(scores[range(N), y]) / \
            np.sum(np.exp(scores), axis=1)
        loss = -np.sum(np.log(probability))

        # 注意，这里没对正则项前面加上0.5，因此求dW的时候要记得乘2,
        # 如果加上0.5的话,jupyter中检查loss计算正确的部分会输出差异为：0.01896541960606335
        # https://github.com/oubindo/cs231n-cnn/blob/master/assignment1/two_layer_net.ipynb
        # 不加的话，就是符合要求的<1e-12
        # https://github.com/amanchadha/stanford-cs231n-assignments-2020/blob/master/assignment1/two_layer_net.ipynb
        loss = loss/N+reg*np.sum(W1*W1)+reg*np.sum(W2*W2)

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################

        # hidden_layer1_output = X.dot(W1)+b1
        # relu_output = np.maximum(0, hidden_layer1_output)
        # hidden_layer2_output = relu_output.dot(W2)+b2
        # scores = hidden_layer2_output
        # probability = np.exp(scores[range(N), y]) /np.sum(np.exp(scores), axis=1)
        all_probs = np.exp(scores)/np.sum(np.exp(scores),
                                          axis=1, keepdims=True)
        all_probs[range(N), y] -= 1
        dW2 = relu_output.T.dot(all_probs)
        # 类似以前的 X.T.dot(coeff_mat) X是W的local gradient，coeff_mat是上游梯度
        # relu_output是 W2*()+b2的本地梯度，all_probs是上游梯度,即▽W2 L
        dB2 = np.sum(all_probs, axis=0)/N

        # all_probs.shape=score.shape=(N,C) W2.shape=(H,C)
        dw1_hidden_layer1 = all_probs.dot(W2.T)
        dw1_hidden_layer1[relu_output <= 0] = 0  # dw1_relu
        dW1 = X.T.dot(dw1_hidden_layer1)
        dB1 = np.sum(dw1_hidden_layer1, axis=0)/N

        dW1 = dW1/N+2*reg*W1
        dW2 = dW2/N+2*reg*W2
        grads = {'W1': dW1, 'b1': dB1, 'W2': dW2, 'b2': dB2}
        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
          pytorch的调整学习率以及支持的函数:
          <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        # 49000/200=245,  1000/245=4
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        for it in range(num_iters):
            batch_indexes = np.random.choice(
                range(num_train), batch_size, replace=True)
            X_batch = X[batch_indexes]
            y_batch = y[batch_indexes]

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params["W1"] -= learning_rate*grads["W1"]
            self.params["b1"] -= learning_rate*grads["b1"]
            self.params["W2"] -= learning_rate*grads["W2"]
            self.params["b2"] -= learning_rate*grads["b2"]

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # 49000/200=245,  1000/245=4  一共4个epoch，加上初始化时候计算的结果，一共5个值
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                val_loss, grads = self.loss(X_val, y=y_val, reg=reg)
                val_loss_history.append(val_loss)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            'val_loss_history': val_loss_history
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        hidden1_layer = np.maximum(
            0, X.dot(self.params["W1"])+self.params["b1"])
        hidden2_layer = hidden1_layer.dot(self.params["W2"])+self.params["b2"]
        y_pred = np.argmax(hidden2_layer, axis=1)
        # print(y_pred.shape)
        return y_pred
