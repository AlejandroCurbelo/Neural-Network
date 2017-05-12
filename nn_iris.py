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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

x_validation = data[0:23, 0:4].astype('f4')
y_validation = one_hot(data[0:23, 4].astype(int), 3)

x_test = data[23:46, 0:4].astype('f4')
y_test = one_hot(data[23:46, 4].astype(int), 3)

x_train = data[46:, 0:4].astype('f4')
y_train = one_hot(data[46:, 4].astype(int), 3)

"""
print "\nSome samples..."
for i in range(20):
    print x_train[i], " -> ", y_train[i]
print
"""

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

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
old_error = 100
new_error = 99
epoch = 0

while new_error > 0.5:
    epoch+=1
    for jj in xrange(len(x_train) / batch_size):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    if  epoch > 50:
        old_error = new_error
    new_error = sess.run(loss, feed_dict={x: x_validation, y_: y_validation})
    print "Epoch #:", epoch, "Error: ", new_error

    """
    result = sess.run(y, feed_dict={x: x_validation})
    for b, r in zip(y_validation, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"
    """

    if old_error < new_error:
        break;

print "----------------------"
print "     Start test...    "
print "----------------------"
result = sess.run(y, feed_dict={x: x_test})
for b, r in zip(y_test, result):
    print b, "-->", r

n_error=0
print "Errores: "
for line_y, line_result in zip(y_test, result):
    if np.argmax(line_y) != np.argmax(line_result):
        n_error+=1
        print line_y, "-->", line_result

print "Numero de errores: ", n_error, "/23 -> ", (n_error*100)/23, "%"
print
