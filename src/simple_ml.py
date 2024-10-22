from cProfile import label
import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # use python gzip to read file https://docs.python.org/3/library/gzip.html
    images = []
    labels = []
    with gzip.open(image_filename) as f:
        chunk = f.read(4) # read 4 bytes
        # python struct https://docs.python.org/3/library/struct.html
        examples = struct.unpack('>i', f.read(4))[0] # examples 数据以大端格式存储
        rows = struct.unpack('>i', f.read(4))[0]
        cols = struct.unpack('>i', f.read(4))[0]
        print(examples, rows, cols)
        while True:
            # every single pixel is one bytes
            single_image_bytes = f.read(rows*cols)
            if not single_image_bytes:
                break
            # print(np.frombuffer(single_image_bytes, dtype=np.uint8).shape)
            single_image = np.frombuffer(single_image_bytes, dtype=np.uint8)
            single_image = single_image / 255.0
            images.append(single_image)
    images = np.stack(images, axis = 0).astype(np.float32)
    
    with gzip.open(label_filename, "rb") as f:
        f.read(4)
        examples = struct.unpack('>i', f.read(4))[0]
        print(examples)
        while True:
            chunk = f.read(1)
            if not chunk:
                break
            single_label = struct.unpack('>B', chunk)
            labels.append(single_label)
    labels = np.array(labels, dtype = np.uint8).flatten()
    return (images, labels)
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    Z_exp = np.exp(Z)
    Z_exp_sum_row = np.sum(Z_exp, axis = 1)
    Z_exp_sum_row_log = np.log(Z_exp_sum_row)
    row_index = np.array(list(range(Z.shape[0])))
    col_index = y
    p = Z[row_index, col_index]
    loss = Z_exp_sum_row_log - p
    loss = np.sum(loss) / Z.shape[0]
    return loss
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # print(X.shape, theta.shape)
    iters = X.shape[0] // batch # need multi_step update
    eye = np.eye(np.max(y) + 1)
    for i in range(iters):
        X_batch = X[i*batch : (i+1)*batch]
        y_batch = y[i*batch : (i+1)*batch]
        Z = X_batch @ theta # [num_examples, input_dim] * [input_dim, num_classes]
        Z_normalize = np.exp(Z) / np.sum(np.exp(Z), axis = 1).reshape(-1,1)
        I_y = eye[y_batch] # [num_examples, num_classes] one-hot
        theta -= lr / batch * (np.transpose(X_batch) @ (Z_normalize - I_y))
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    i = 0
    while i + batch < num_examples:
        inputs = X[i : i + batch]
        labels = y[i : i + batch]
        nn_epoch_batch(inputs, labels, W1, W2, lr)
        i += batch
    if i < num_examples:
        inputs = X[i : num_examples]
        labels = y[i : num_examples]
        nn_epoch_batch(inputs, labels, W1, W2, lr)
    
    ### END YOUR CODE

def nn_epoch_batch(X, y, W1, W2, lr = 0.1):
    z1 = np.matmul(X, W1) # (num_examples, input_dim) * (input_dim, hidden_dim)
    z1_relu = np.where(z1 >= 0.0, z1, 0.0) # (num_examples, hidden_dim)
    z2 = np.matmul(z1_relu, W2) # (num_examples, hidden_dim) * (hidden_dim, num_classes)
    z2_exp = np.exp(z2)
    z2_normalize = z2_exp / (np.sum(z2_exp, axis = 1)[:, None])
    I = np.eye(W2.shape[1])
    I_y = I[y]
    H = z2_normalize - I_y
    dW2 = np.transpose(z1_relu) @ H / X.shape[0]
    dW1 = np.transpose(X) @ (np.where(z1 > 0.0, 1.0, 0.0) * (H @ W2.T)) / X.shape[0]
    W1 -= dW1 * lr
    W2 -= dW2 * lr

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
