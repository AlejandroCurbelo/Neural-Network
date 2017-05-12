import gzip
import cPickle

import tensorflow as tf
import numpy as np

# Alumno: Alejandro Curbelo Fontelos

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

x_train, y_train = train_set
x_valid, y_valid = valid_set
x_test, y_test = test_set

#784 data_dim
#train_set 50000
#valid_set 10000
#test_set 10000

y_train = one_hot(y_train.astype(int), 10)
y_valid = one_hot(y_valid.astype(int), 10)
y_test = one_hot(y_test.astype(int), 10)

"""
print "\nSome samples..."
for i in range(20):
    print x_train[i], " -> ", y_train[i]
print
"""

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 13)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(13)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(13, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
old_error = 10000
new_error = 9999
epoch = 0

while new_error > 1:
    epoch+=1
    for jj in xrange(len(x_train) / batch_size):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    if epoch > 50:
        old_error = new_error
    new_error = sess.run(loss, feed_dict={x: x_valid, y_: y_valid})
    print "Epoch #:", epoch, "Error: ", new_error

    """
    result = sess.run(y, feed_dict={x: x_valid})
    for i in range(10):
        print y_valid[i], "-->"
        for j in range(len(result[i])):
            print "%.3f" % result[i][j], " ",
        print
    print "----------------------------------------------------------------------------------"
    """

    if 1 < new_error-old_error:
        break;

print "----------------------"
print "     Start test...    "
print "----------------------"
result = sess.run(y, feed_dict={x: x_test})
for i in range(10):
    print y_test[i], "-->"
    for j in range(len(result[i])):
        print "%.3f" % result[i][j],
    print

print "(...)"

n_error=0
print "Errores: "
for line_y, line_result in zip(y_test, result):
    if np.argmax(line_y) != np.argmax(line_result):
        n_error+=1

print "Numero de errores: ", n_error, "/10000 -> ", n_error/100, "%"
print

"""
# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(x_train[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print y_train[57]


# TODO: the neural net!!
"""
