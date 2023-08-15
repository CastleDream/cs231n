import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                dists[i, j] = np.linalg.norm(self.X_train[j]-X[i], ord=None)
                # dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j] - X[i]), dtype=float))
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            dists[i] = np.linalg.norm(self.X_train-X[i], axis=1)
            # dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1, dtype=float))
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        sx = np.sum(X**2, axis=1, keepdims=True, dtype=float)
        sy = np.transpose(np.sum(self.X_train**2, axis=1,
                          keepdims=True, dtype=float))
        dists = np.sqrt(-2 * X.dot(self.X_train.T) + sx + sy)
        # 使用二重循环，可以看到， √(a-b)^2 从单个元素的平方差，变成多个元素的平方差
        # dists[i, j] = np.linalg.norm(self.X_train[j]-X[i], ord=None)
        # 多个展开分别求每项的值，再相加，其实就等于每个元素的单项求值

        # 写法2
        # a = -2 * np.dot(X, self.X_train.T)
        # b = np.sum(np.square(self.X_train), axis=1)
        # c = np.transpose([np.sum(np.square(X), axis=1)])
        # dists = np.sqrt(a + b + c)

        # 方法2 会内存溢出
        # X = X[:, np.newaxis, :]
        # dists = np.linalg.norm(X-self.X_train, axis=2)
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=np.uint8)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            closest_y_index = np.argsort(dists[i])[:k]
            closest_y = self.y_train[closest_y_index]
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            values, counts = np.unique(closest_y, return_counts=True)
            ind = np.argmax(counts)
            y_pred[i] = values[ind]
            # count = np.bincount(closest_y)
            # y_pred[i] = np.argmax(count)
        return y_pred


if __name__ == "__main__":
    import time
    x_train = np.random.normal(size=(5000, 3072))
    y_train = np.random.choice(range(11), 10)
    print(y_train)
    x_test = np.random.normal(size=(500, 3072))
    KNearestNeighbor_test = KNearestNeighbor()
    KNearestNeighbor_test.train(x_train, y_train)
    start_time = time.time()
    dists_two = KNearestNeighbor_test.compute_distances_two_loops(x_test)
    print(f"two_loops execution time is {time.time()-start_time}")
    # 21.9s  500X5000个距离计算
    start_time = time.time()
    dists_one = KNearestNeighbor_test.compute_distances_one_loop(x_test)
    print(f"one_loops execution time is {time.time()-start_time}")
    # 36.8s 单循环竟然更慢。。
    start_time = time.time()
    difference = np.linalg.norm(dists_two - dists_one, ord='fro')
    print(
        f'Difference was: {difference}, calulation time is {time.time()-start_time}')
