import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

###https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0###
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
###########################################################
 
def generate_data(m, d):
    # generate an array for a grid of m equidistant points per dimension for the hypercube [0, 4pi]^d
    data = np.zeros((m**d, d))
    for i in range(0, d):
        data[:, i] = np.tile(np.repeat(np.linspace(0, 4*np.pi, m), m**(d-(i+1))), m**i)
    return(data)


def generate_label(data):
    return(np.apply_along_axis(lambda x: sum(np.sin(x)), 1, data))


"""https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0
I utilized this with minor edits"""

def init_weights(shape):
	weights = tf.random_normal(shape, stddev=0.1)
	return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
	h = tf.nn.sigmoid(tf.matmul(X, w_1))
	yhat = tf.matmul(h, w_2) #yhat is not softmax; that's done internally
	return yhat

#Not yet working with main()
def get_split_data(m, d):
	data = generate_data(m, d)
	label = generate_label(data)
    
    # Prepend the column of 1s for bias
	N, M = data.shape
	all_X = np.ones((N,M+1))
	all_X[:,1:] = data

	# Convert into one-hot vectors
	num_labels = len(np.unique(label))
	all_Y = np.eye(num_labels)[label]
	return train_test_split(all_X, all_Y, test_size=0.2, random_state=RANDOM_SEED)

#functional with main() as it  is from example
def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    train_X, test_X, train_y, test_y = get_iris_data() #get_split_data(10,3)

    # Layer's sizes
    x_size = train_X.shape[1] # Number of input nodes: 4 features and 1 bias
    h_size = 256              # Number of hidden nodes
    y_size = train_y.shape[1] # Number of outcomes 

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

     # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()
