__author__ = 'alexromriell'

import tensorflow as tf
import numpy as np

# sess = tf.InteractiveSession()


# placeholders
x = tf.placeholder(tf.float32, shape=[None, 2304])  # 48 x 48 flattened image
y_ = tf.placeholder(tf.float32, shape=[None, 7])  # 7 classes of emotion


# Create weight and bias functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])  # creates 32 features from a 5x5 kernel
b_conv1 = bias_variable([32])


# flatten the image to a 1-D tensor. last argument tells TF that there is 1 color channel, greyscale
image = tf.random_uniform(shape=[2304], maxval=256)

x_image = tf.reshape(image, [-1,48,48,1])


# convolve x_image
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# second convolutional layer. getting DEEP
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# third convolutional layer. getting DEEPER!
W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)  # this should give us a 6 x 6 image

print h_pool3

# densely connected layer
W_fc1 = weight_variable([6 * 6 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 6*6*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


# dropout - keep from overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# readout-layer -- make the final prediction
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



# train and evaluate
# How well does this model do? To train and evaluate it we will use code
# that is nearly identical to that for the simple one layer SoftMax network above.
# The differences are that: we will replace the steepest gradient descent optimizer
# with the more sophisticated ADAM optimizer; we will include the additional parameter
# keep_prob in feed_dict to control the dropout rate; and we will add logging to
# every 100th iteration in the training process.

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



